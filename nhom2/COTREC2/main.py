"""COTREC2 — run original COTREC on CatSA-exported RetailRocket data."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
COTREC_DIR = REPO_ROOT / "nhom2" / "COTREC"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COTREC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import COTREC, train_test, trans_to_cuda  # noqa: E402
from util import Data  # noqa: E402

from ncs_logging import format_best_k_summary, write_run_summary  # noqa: E402
from ncs_paths import data_dir, log_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train COTREC on CatSA-exported data.")
    parser.add_argument("--dataset", default="retailrocket")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batchSize", type=int, default=100)
    parser.add_argument("--embSize", type=int, default=100)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--lam", type=float, default=0.005)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="1 epoch, subsampled data (quick check)")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Process log root → <log-dir>/<dataset>/DD-MM-YYYY.log",
    )
    parser.add_argument(
        "--log-mins-dir",
        type=Path,
        default=None,
        help="Summary log root → <log-mins-dir>/<dataset>/DD-MM-YYYY.log",
    )
    return parser.parse_args()


def resolve_log_paths(
    args: argparse.Namespace,
    ds_name: str,
    *,
    now: datetime | None = None,
) -> tuple[Path, Path]:
    now = now or datetime.now()
    date_name = f"{now:%d-%m-%Y}.log"
    if args.log_dir is not None:
        process_log = Path(args.log_dir) / ds_name / date_name
    else:
        process_log = log_file("COTREC2", ds_name, now)
    if args.log_mins_dir is not None:
        summary_log = Path(args.log_mins_dir) / ds_name / date_name
    else:
        from ncs_paths import log_mins_file

        summary_log = log_mins_file("COTREC2", ds_name, now)
    return process_log, summary_log


def _subsample_cotrec_data(
    data: tuple[list, list],
    limit: int,
) -> tuple[list, list]:
    if limit <= 0 or limit >= len(data[0]):
        return data
    return data[0][:limit], data[1][:limit]


def load_meta(data_root: Path) -> dict:
    meta_path = data_root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run: python export_data.py --dataset retailrocket"
        )
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epoch = 1
        if args.max_train_samples == 0:
            args.max_train_samples = 5000
        if args.max_test_samples == 0:
            args.max_test_samples = 2000

    ds = args.dataset.lower()
    data_root = args.data_dir or data_dir("COTREC2", ds)
    meta = load_meta(data_root)
    n_node = int(meta["num_items"])

    print(args)
    print(f"Data: {data_root}")
    print(f"Meta: {meta}")

    train_data = pickle.load(open(data_root / "train.txt", "rb"))
    test_data = pickle.load(open(data_root / "test.txt", "rb"))
    all_train = pickle.load(open(data_root / "all_train_seq.txt", "rb"))
    train_data = _subsample_cotrec_data(train_data, args.max_train_samples)
    test_data = _subsample_cotrec_data(test_data, args.max_test_samples)
    if args.max_train_samples or args.max_test_samples:
        print(
            f"Subsampled: train={len(train_data[0])} test={len(test_data[0])} "
            f"(limits train={args.max_train_samples} test={args.max_test_samples})"
        )

    train_loader = Data(train_data, all_train, shuffle=True, n_node=n_node)
    test_loader = Data(test_data, all_train, shuffle=True, n_node=n_node)

    model = trans_to_cuda(
        COTREC(
            adjacency=train_loader.adjacency,
            n_node=n_node,
            lr=args.lr,
            l2=args.l2,
            beta=args.beta,
            lam=args.lam,
            eps=args.eps,
            layers=args.layer,
            emb_size=args.embSize,
            batch_size=args.batchSize,
            dataset=ds,
        )
    )

    top_k = [5, 10, 20]
    best_results: dict[str, list] = {}
    for k in top_k:
        best_results[f"epoch{k}"] = [0, 0]
        best_results[f"metric{k}"] = [0, 0]

    log_path, log_mins_path = resolve_log_paths(args, ds)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n{'=' * 60}\nCOTREC2 run {datetime.now()} args={args}\n")
        logf.write(f"meta={meta}\n")

        for epoch in range(args.epoch):
            print("-------------------------------------------------------")
            print("epoch:", epoch)
            metrics, total_loss = train_test(model, train_loader, test_loader, epoch)
            for k in top_k:
                metrics[f"hit{k}"] = np.mean(metrics[f"hit{k}"]) * 100
                metrics[f"mrr{k}"] = np.mean(metrics[f"mrr{k}"]) * 100
                if best_results[f"metric{k}"][0] < metrics[f"hit{k}"]:
                    best_results[f"metric{k}"][0] = metrics[f"hit{k}"]
                    best_results[f"epoch{k}"][0] = epoch
                if best_results[f"metric{k}"][1] < metrics[f"mrr{k}"]:
                    best_results[f"metric{k}"][1] = metrics[f"mrr{k}"]
                    best_results[f"epoch{k}"][1] = epoch
            print(metrics)
            line = (
                f"epoch={epoch} loss={total_loss:.4f} "
                f"HR@20={metrics['hit20']:.4f} MRR@20={metrics['mrr20']:.4f}"
            )
            logf.write(line + "\n")
            logf.flush()
            for k in top_k:
                print(
                    "train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d, %d"
                    % (
                        total_loss,
                        k,
                        best_results[f"metric{k}"][0],
                        k,
                        best_results[f"metric{k}"][1],
                        best_results[f"epoch{k}"][0],
                        best_results[f"epoch{k}"][1],
                    )
                )

    header = (
        f"COTREC2 on CatSA data | split_mode={meta.get('split_mode')} | "
        f"n_node={n_node} | {args}"
    )
    summary_lines = format_best_k_summary(ds, best_results, top_k, header=header)
    if args.log_mins_dir is not None:
        log_mins_path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(summary_lines)
        if log_mins_path.exists() and log_mins_path.stat().st_size > 0:
            body = f"\n{'=' * 60}\n{body}"
        with log_mins_path.open("a", encoding="utf-8") as handle:
            handle.write(body)
            handle.write("\n")
        print(f"Summary log: {log_mins_path}")
    else:
        write_run_summary("COTREC2", ds, summary_lines)
    print(f"Process log: {log_path}")


if __name__ == "__main__":
    main()
