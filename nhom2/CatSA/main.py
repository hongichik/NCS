"""CatSA training entry point — extends COTREC baseline with category graph + CL."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ncs_logging import format_best_k_summary, write_run_summary
from ncs_paths import data_dir, log_file

from data_utils import load_artifact, load_lookup_tables, remap_sessions_for_taxonomy
from model.catsa import CatSA
from preprocess.augmentation import catsa_augment
from preprocess.session_graph_dataset import (
    CategorySessionGraphDataset,
    build_prefix_target_pairs,
    fit_taxonomy_encoders,
)

PROBLEM = "CatSA"
DEFAULT_DATASET = "retailrocket"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CatSA on RetailRocket.")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--mode", choices=["a2", "full"], default="full", help="a2=Module1 only, full=+CL")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=100)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--cl-weight", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--eta-aug", type=float, default=0.3)
    p.add_argument("--k-min", type=int, default=5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--metric-k", type=int, default=20)
    p.add_argument("--smoke", action="store_true", help="Quick run: 1 epoch, small batch")
    p.add_argument("--max-train-sessions", type=int, default=0, help="Limit train sessions (0=all)")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(
    sessions: list[list[int]],
    item2leaf: dict[int, int],
    leaf2parent: dict[int, int],
    taxonomy,
) -> CategorySessionGraphDataset:
    prefixes, targets = build_prefix_target_pairs(sessions)
    valid_p, valid_t = [], []
    for prefix, target in zip(prefixes, targets):
        if taxonomy.item_encoder.get(target) is None:
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


@torch.no_grad()
def evaluate(model: CatSA, loader: DataLoader, device: torch.device, k: int) -> tuple[float, float]:
    model.eval()
    hits, rr_sum, total = 0, 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits, _, _ = model(batch, None)
        target = batch.y.view(-1)
        topk = torch.topk(logits, min(k, logits.size(-1)), dim=-1).indices
        matches = topk.eq(target.unsqueeze(1))
        hit = matches.any(dim=1)
        hits += int(hit.sum().item())
        if hit.any():
            rows = hit.nonzero(as_tuple=True)[0]
            pos = matches[rows].to(torch.int64).argmax(dim=1) + 1
            rr_sum += float((1.0 / pos.to(torch.float32)).sum().item())
        total += int(target.size(0))
    if total == 0:
        return 0.0, 0.0
    return 100.0 * hits / total, 100.0 * rr_sum / total


def build_augmented_batch(batch, lookup, taxonomy, item2leaf, leaf2parent, eta_aug, k_min):
    item2cat = lookup["item2cat"]
    cat2items = lookup["cat2items"]
    siblings = lookup["siblings"]
    graphs = batch.to_data_list()
    aug_sessions, targets = [], []
    for g in graphs:
        session = [int(x) for x in g.raw_session.tolist()]
        session_0 = [x - 1 for x in session]
        aug_0 = catsa_augment(session_0, item2cat, cat2items, siblings, eta_aug=eta_aug, k_min=k_min)
        aug_sessions.append([x + 1 for x in aug_0])
        targets.append(int(g.y.view(-1)[0].item()))
    aug_ds = CategorySessionGraphDataset(
        session_sequences=aug_sessions,
        item2leaf_dict=item2leaf,
        leaf2parent_dict=leaf2parent,
        targets=targets,
        taxonomy=taxonomy,
        raw_sessions=aug_sessions,
    )
    return next(iter(DataLoader(aug_ds, batch_size=len(aug_ds))))


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 32)

    device = resolve_device(args.device)
    ds_name = args.dataset.lower()
    ddir = data_dir(PROBLEM, ds_name)

    artifact = load_artifact(ddir)
    lookup = load_lookup_tables(ddir)

    item2cat = {int(k): int(v) for k, v in artifact["item2cat"].items()}
    cat_parent = {int(k): int(v) for k, v in artifact["cat_parent"].items()}
    item2leaf_1b, leaf2parent = remap_sessions_for_taxonomy([], item2cat, cat_parent)

    all_sessions = []
    for split in ("train", "valid", "test"):
        all_sessions.extend(artifact["splits"][split])
    taxonomy = fit_taxonomy_encoders(all_sessions, item2leaf_1b, leaf2parent)

    train_sessions = artifact["splits"]["train"]
    if args.smoke and args.max_train_sessions == 0:
        args.max_train_sessions = 500
    if args.max_train_sessions > 0:
        train_sessions = train_sessions[: args.max_train_sessions]
    train_ds = build_dataset(train_sessions, item2leaf_1b, leaf2parent, taxonomy)
    valid_ds = build_dataset(artifact["splits"]["valid"], item2leaf_1b, leaf2parent, taxonomy)
    test_ds = build_dataset(artifact["splits"]["test"], item2leaf_1b, leaf2parent, taxonomy)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = CatSA(
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
    k = args.metric_k
    best_hr, best_mrr, best_epoch = 0.0, 0.0, 0
    patience_ctr = 0

    log_path = log_file(PROBLEM, ds_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n{'='*60}\nCatSA run {datetime.now()} mode={args.mode} args={args}\n")

        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            total_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = batch.to(device)
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
                n_batches += 1

            val_hr, val_mrr = evaluate(model, valid_loader, device, k)
            msg = (
                f"epoch={epoch+1} loss={total_loss/max(n_batches,1):.4f} "
                f"val_HR@{k}={val_hr:.4f} val_MRR@{k}={val_mrr:.4f} time={time.time()-t0:.1f}s"
            )
            print(msg)
            logf.write(msg + "\n")
            logf.flush()

            if val_hr > best_hr:
                best_hr, best_mrr, best_epoch = val_hr, val_mrr, epoch
                patience_ctr = 0
            else:
                patience_ctr += 1
            if args.patience > 0 and patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        test_hr, test_mrr = evaluate(model, test_loader, device, k)
        summary = (
            f"TEST HR@{k}={test_hr:.4f} MRR@{k}={test_mrr:.4f} | "
            f"BEST_VAL HR@{k}={best_hr:.4f} epoch={best_epoch+1}"
        )
        print(summary)
        logf.write(summary + "\n")

    best_results = {
        "metric20": [test_hr, test_mrr],
        "epoch20": [best_epoch, best_epoch],
        "metric10": [0, 0],
        "epoch10": [0, 0],
        "metric5": [0, 0],
        "epoch5": [0, 0],
    }
    write_run_summary(
        PROBLEM,
        ds_name,
        format_best_k_summary(ds_name, best_results, [20], header=str(args)),
    )


if __name__ == "__main__":
    main()
