"""CatSA2 training — implementation aligned with CatSA build guide (May 2026)."""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ncs_logging import format_best_k_summary, write_run_summary
from ncs_paths import data_dir, log_file

from data_utils import load_artifact, load_lookup_tables, remap_sessions_for_taxonomy
from evaluation import evaluate_topk
from model.catsa import CatSA2
from preprocess.augmentation import catsa_augment
from preprocess.session_graph_dataset import (
    CategorySessionGraphDataset,
    build_prefix_target_pairs,
    fit_taxonomy_encoders,
)

PROBLEM = "CatSA2"
DEFAULT_DATASET = "retailrocket"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "default.yaml"


def _parse_simple_yaml(text: str) -> dict:
    """Minimal YAML parser for flat key: value config (no extra deps)."""
    out: dict = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            out[key] = [] if not inner else [int(x.strip()) for x in inner.split(",")]
        elif val.lower() in ("true", "false"):
            out[key] = val.lower() == "true"
        elif val.replace(".", "", 1).isdigit() or (val.startswith("-") and val[1:].replace(".", "", 1).isdigit()):
            out[key] = float(val) if "." in val else int(val)
        else:
            out[key] = val
    return out


def load_yaml_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
        return data or {}
    except ImportError:
        return _parse_simple_yaml(text)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CatSA2 (guide-compliant) on RetailRocket.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--dataset", default=None)
    p.add_argument("--mode", choices=["a2", "full"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--cl-weight", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--eta-aug", type=float, default=None)
    p.add_argument("--k-min", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--metric-k", type=int, default=None, help="Primary K for early stopping")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-train-sessions", type=int, default=0)
    p.add_argument("--max-valid-sessions", type=int, default=0)
    p.add_argument("--max-test-sessions", type=int, default=0)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--log-mins-dir", type=Path, default=None)
    cli = p.parse_args()

    cfg = load_yaml_config(cli.config) if cli.config.exists() else {}
    metric_ks = cfg.get("metric_ks", [10, 20])
    if not isinstance(metric_ks, list):
        metric_ks = [10, 20]

    def pick(name: str, cli_val, default):
        if cli_val is not None:
            return cli_val
        if name in cfg and cfg[name] is not None:
            return cfg[name]
        return default

    args = argparse.Namespace(
        config=cli.config,
        dataset=str(pick("dataset", cli.dataset, DEFAULT_DATASET)).lower(),
        mode=pick("mode", cli.mode, "full"),
        epochs=int(pick("epochs", cli.epochs, 30)),
        batch_size=int(pick("batch_size", cli.batch_size, 100)),
        hidden_dim=int(pick("hidden_dim", cli.hidden_dim, 100)),
        num_layers=int(pick("num_layers", cli.num_layers, 2)),
        lr=float(pick("lr", cli.lr, 1e-3)),
        weight_decay=float(pick("weight_decay", cli.weight_decay, 1e-5)),
        dropout=float(pick("dropout", cli.dropout, 0.1)),
        cl_weight=float(pick("cl_weight", cli.cl_weight, 0.1)),
        temperature=float(pick("temperature", cli.temperature, 0.5)),
        eta_aug=float(pick("eta_aug", cli.eta_aug, 0.3)),
        k_min=int(pick("k_min", cli.k_min, 5)),
        patience=int(pick("patience", cli.patience, 5)),
        seed=int(pick("seed", cli.seed, 42)),
        device=pick("device", cli.device, "auto"),
        metric_k=int(pick("metric_k", cli.metric_k, max(metric_ks))),
        metric_ks=[int(k) for k in metric_ks],
        smoke=cli.smoke,
        max_train_sessions=cli.max_train_sessions,
        max_valid_sessions=cli.max_valid_sessions,
        max_test_sessions=cli.max_test_sessions,
        log_every=int(pick("log_every", cli.log_every, 50)),
        log_dir=cli.log_dir,
        log_mins_dir=cli.log_mins_dir,
    )
    return args


def resolve_log_paths(args: argparse.Namespace, ds_name: str, *, now: datetime | None = None):
    now = now or datetime.now()
    date_name = f"{now:%d-%m-%Y}.log"
    if args.log_dir is not None:
        process_log = Path(args.log_dir) / ds_name / date_name
    else:
        process_log = log_file(PROBLEM, ds_name, now)
    if args.log_mins_dir is not None:
        summary_log = Path(args.log_mins_dir) / ds_name / date_name
    else:
        from ncs_paths import log_mins_file

        summary_log = log_mins_file(PROBLEM, ds_name, now)
    return process_log, summary_log


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA không khả dụng. Dùng --device cpu.")
        return torch.device("cuda")
    if name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS không khả dụng. Dùng --device cpu hoặc cuda.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_device_info(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return str(device)


def build_dataset(sessions, item2leaf, leaf2parent, taxonomy):
    prefixes, targets = build_prefix_target_pairs(sessions)
    valid_p, valid_t = [], []
    for prefix, target in zip(prefixes, targets):
        if taxonomy.item_encoder.get(target) is None:
            continue
        if not prefix or not all(item_id in taxonomy.item_encoder for item_id in prefix):
            continue
        valid_p.append(prefix)
        valid_t.append(target)
    return CategorySessionGraphDataset(
        session_sequences=valid_p,
        item2leaf_dict=item2leaf,
        leaf2parent_dict=leaf2parent,
        targets=valid_t,
        taxonomy=taxonomy,
        raw_sessions=valid_p,
    )


def build_augmented_batch(batch, lookup, taxonomy, item2leaf, leaf2parent, eta_aug, k_min):
    item2cat = lookup["item2cat"]
    cat2items = lookup["cat2items"]
    siblings = lookup["siblings"]
    graphs = batch.to_data_list()
    aug_sessions = []
    for g in graphs:
        session = [int(x) for x in g.raw_session.tolist()]
        session_0 = [x - 1 for x in session]
        aug_0 = catsa_augment(session_0, item2cat, cat2items, siblings, eta_aug=eta_aug, k_min=k_min)
        aug_sessions.append([x + 1 for x in aug_0])
    aug_ds = CategorySessionGraphDataset(
        session_sequences=aug_sessions,
        item2leaf_dict=item2leaf,
        leaf2parent_dict=leaf2parent,
        targets=None,
        taxonomy=taxonomy,
        raw_sessions=aug_sessions,
    )
    return next(iter(DataLoader(aug_ds, batch_size=len(aug_ds))))


def _limit_sessions(sessions, limit, label):
    if limit <= 0 or limit >= len(sessions):
        return sessions
    print(f"  {label}: using {limit}/{len(sessions)} sessions", flush=True)
    return sessions[:limit]


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 32)
        if args.max_train_sessions == 0:
            args.max_train_sessions = 200
        if args.max_valid_sessions == 0:
            args.max_valid_sessions = 100
        if args.max_test_sessions == 0:
            args.max_test_sessions = 100

    set_seed(args.seed)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    device_info = format_device_info(device)
    print(f"CatSA2 | seed={args.seed} | device={device_info}", flush=True)

    ds_name = args.dataset
    ddir = data_dir(PROBLEM, ds_name)
    if not (ddir / "retailrocket.json").exists():
        fallback = data_dir("CatSA", ds_name)
        if (fallback / "retailrocket.json").exists():
            print(f"Data/CatSA2 not found; reusing {fallback}", flush=True)
            ddir = fallback
        else:
            raise FileNotFoundError(
                f"Missing artifact at {ddir}. Run: "
                f"python -m preprocess.retailrocket_preprocess"
            )

    artifact = load_artifact(ddir)
    lookup = load_lookup_tables(ddir)

    item2cat = {int(k): int(v) for k, v in artifact["item2cat"].items()}
    cat_parent = {int(k): int(v) for k, v in artifact["cat_parent"].items()}
    item2leaf_1b, leaf2parent = remap_sessions_for_taxonomy([], item2cat, cat_parent)

    # Item/leaf vocab from all splits (test items are from train pool per preprocess).
    # Lookup tables for augmentation stay train+valid-only in lookup_tables.pkl.
    vocab_sessions = []
    for split in ("train", "valid", "test"):
        vocab_sessions.extend(artifact["splits"][split])
    taxonomy = fit_taxonomy_encoders(vocab_sessions, item2leaf_1b, leaf2parent)
    print(
        f"Taxonomy: items={taxonomy.num_items} "
        f"leaf={taxonomy.num_leaf_cats} parent={taxonomy.num_parent_cats}",
        flush=True,
    )

    train_sessions = _limit_sessions(artifact["splits"]["train"], args.max_train_sessions, "train")
    valid_sessions = _limit_sessions(artifact["splits"]["valid"], args.max_valid_sessions, "valid")
    test_sessions = _limit_sessions(artifact["splits"]["test"], args.max_test_sessions, "test")

    train_ds = build_dataset(train_sessions, item2leaf_1b, leaf2parent, taxonomy)
    valid_ds = build_dataset(valid_sessions, item2leaf_1b, leaf2parent, taxonomy)
    test_ds = build_dataset(test_sessions, item2leaf_1b, leaf2parent, taxonomy)
    print(
        f"Datasets: train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)} graphs",
        flush=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)

    model = CatSA2(
        num_items=taxonomy.num_items,
        num_leaf_cats=taxonomy.num_leaf_cats,
        num_parent_cats=taxonomy.num_parent_cats,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        cl_weight=args.cl_weight,
        temperature=args.temperature,
        mode=args.mode,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    primary_k = args.metric_k
    best_hr, best_epoch = 0.0, 0
    patience_ctr = 0

    log_path, log_mins_path = resolve_log_paths(args, ds_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    n_train_batches = max(len(train_loader), 1)

    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n{'='*60}\nCatSA2 run {datetime.now()} mode={args.mode}\n")
        logf.write(f"config={args.config} seed={args.seed} args={args}\n")
        logf.write(f"Device: {device_info}\n")

        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            total_loss = total_rec = total_cl = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = batch.to(device, non_blocking=pin_memory)
                graph_aug = None
                if args.mode == "full":
                    graph_aug = build_augmented_batch(
                        batch, lookup, taxonomy, item2leaf_1b, leaf2parent, args.eta_aug, args.k_min
                    ).to(device)

                optimizer.zero_grad()
                _, loss_rec, loss_cl = model(batch, graph_aug)
                loss = model.total_loss(loss_rec, loss_cl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                total_loss += float(loss.item())
                total_rec += float(loss_rec.item())
                if loss_cl is not None:
                    total_cl += float(loss_cl.item())
                n_batches += 1

                if args.log_every > 0 and (
                    n_batches % args.log_every == 0 or n_batches == n_train_batches
                ):
                    msg = f"  batch {n_batches}/{n_train_batches} loss={total_loss/n_batches:.4f}"
                    print(msg, flush=True)
                    logf.write(msg + "\n")

            val_metrics = {k: evaluate_topk(model, valid_loader, device, k) for k in args.metric_ks}
            val_hr, val_ndcg, val_mrr = val_metrics[primary_k]
            msg = (
                f"epoch={epoch+1} loss={total_loss/max(n_batches,1):.4f} "
                f"rec={total_rec/max(n_batches,1):.4f} cl={total_cl/max(n_batches,1):.4f} "
                f"val_HR@{primary_k}={val_hr:.4f} val_NDCG@{primary_k}={val_ndcg:.4f} "
                f"val_MRR@{primary_k}={val_mrr:.4f} time={time.time()-t0:.1f}s"
            )
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()

            if val_hr > best_hr:
                best_hr, best_epoch = val_hr, epoch
                patience_ctr = 0
            else:
                patience_ctr += 1
            if args.patience > 0 and patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break

        test_results = {}
        for k in args.metric_ks:
            hr, ndcg, mrr = evaluate_topk(model, test_loader, device, k)
            test_results[k] = (hr, ndcg, mrr)
            line = f"TEST HR@{k}={hr:.4f} NDCG@{k}={ndcg:.4f} MRR@{k}={mrr:.4f}"
            print(line, flush=True)
            logf.write(line + "\n")

        phr, pndcg, pmrr = test_results[primary_k]
        logf.write(
            f"BEST_VAL HR@{primary_k}={best_hr:.4f} epoch={best_epoch+1}\n"
        )

    best_results: dict = {}
    for k in (5, 10, 20):
        if k in test_results:
            hr, _, mrr = test_results[k]
            best_results[f"metric{k}"] = [hr, mrr]
            best_results[f"epoch{k}"] = [best_epoch, best_epoch]
        else:
            best_results[f"metric{k}"] = [0.0, 0.0]
            best_results[f"epoch{k}"] = [0, 0]
    summary_lines = format_best_k_summary(ds_name, best_results, args.metric_ks, header=f"CatSA2 {args.mode}")
    if args.log_mins_dir is not None:
        log_mins_path.parent.mkdir(parents=True, exist_ok=True)
        with log_mins_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(summary_lines) + "\n")
    else:
        write_run_summary(PROBLEM, ds_name, summary_lines)


if __name__ == "__main__":
    main()
