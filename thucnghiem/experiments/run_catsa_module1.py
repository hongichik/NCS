from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from experiments.models.category_enhanced_gnn import CategoryEnhancedGNN
from preprocessing.retailrocket_preprocess import (
    build_sessions_from_events,
    load_preprocessed_artifacts,
    load_item_to_leaf_mapping,
    load_leaf_to_parent_mapping,
    resolve_retailrocket_file_paths,
)
from preprocessing.session_graph_dataset import (
    CategorySessionGraphDataset,
    build_prefix_target_pairs,
    fit_taxonomy_encoders,
)


UNKNOWN_CATEGORY_ID = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CatSA Module 1 on toy data or RetailRocket.")
    parser.add_argument("--data-root", type=Path, default=None, help="Path to DATA/retailrocket")
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=None,
        help="Path to a preprocessed RetailRocket artifact created by preprocessing.retailrocket_preprocess.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="train",
        help="Which split to load from the processed artifact.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory used to store experiment logs. If omitted, logs are only printed to stdout.",
    )
    parser.add_argument(
        "--log-file-name",
        type=str,
        default="catsa_module1.log",
        help="Log file name created inside --log-dir.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 Regularization penalty")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--conv-type", choices=["sage", "gat"], default="sage")
    parser.add_argument(
        "--version",
        type=int,
        choices=[1, 2],
        default=1,
        help="Pipeline version. version=2 maps uncategorized items to UNK_CAT=0.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--metric-k",
        type=int,
        default=20,
        help="K used for ranking metrics (Recall@K and MRR@K).",
    )
    parser.add_argument(
        "--eval-split",
        choices=["auto", "none", "train", "valid", "test"],
        default="auto",
        help="Split used for metric evaluation each epoch. 'auto' uses valid when training on train split.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Early stopping patience in epochs. Training stops after this many non-improving epochs. Set 0 to disable.",
    )
    parser.add_argument(
        "--monitor-metric",
        choices=["mrr", "recall"],
        default="mrr",
        help="Metric to monitor for early stopping (mrr or recall).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum MRR@K improvement (in percentage points) required to reset early stopping.",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=500,
        help="Log training progress every N mini-batches within each epoch.",
    )
    return parser.parse_args()


def apply_versioned_category_mapping(
    session_sequences: list[list[int]],
    item2leaf_dict: dict[int, int],
    *,
    version: int,
    logger: logging.Logger,
) -> dict[int, int]:
    if version != 2:
        return item2leaf_dict

    all_item_ids = {int(item_id) for session in session_sequences for item_id in session}
    augmented_item2leaf = dict(item2leaf_dict)
    missing_item_ids = [item_id for item_id in all_item_ids if item_id not in augmented_item2leaf]

    for item_id in missing_item_ids:
        augmented_item2leaf[item_id] = UNKNOWN_CATEGORY_ID

    logger.info(
        "Version 2 enabled | UNK_CAT=%d | assigned_missing_items=%d",
        UNKNOWN_CATEGORY_ID,
        len(missing_item_ids),
    )
    return augmented_item2leaf


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_logging(log_dir: Optional[Path], log_file_name: str) -> logging.Logger:
    logger = logging.getLogger("catsa.module1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_file_name
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Logging to file: %s", log_path.resolve())

    return logger


def build_toy_example() -> tuple[list[list[int]], dict[int, int], dict[int, int]]:
    session_sequences = [
        [1, 2, 5, 2],
        [3, 8, 4],
        [1, 7, 9],
        [5, 10, 11, 12],
    ]
    item2leaf_dict = {
        1: 101,
        2: 101,
        3: 102,
        4: 102,
        5: 103,
        7: 104,
        8: 102,
        9: 104,
        10: 105,
        11: 105,
        12: 106,
    }
    leaf2parent_dict = {
        101: 201,
        102: 201,
        103: 202,
        104: 202,
        105: 203,
        106: 203,
    }
    return session_sequences, item2leaf_dict, leaf2parent_dict


def load_retailrocket_inputs(data_root: Path) -> tuple[list[list[int]], dict[int, int], dict[int, int]]:
    events_path, category_tree_path, property_paths, _ = resolve_retailrocket_file_paths(data_root)

    missing_paths = [path for path in [events_path, category_tree_path, *property_paths] if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing RetailRocket files: {missing}")

    session_sequences = build_sessions_from_events(events_path)
    item2leaf_dict = load_item_to_leaf_mapping(property_paths)
    leaf2parent_dict = load_leaf_to_parent_mapping(category_tree_path)
    return session_sequences, item2leaf_dict, leaf2parent_dict


def build_prefix_dataset(
    session_sequences: list[list[int]],
    *,
    taxonomy,
    item2leaf_dict: dict[int, int],
    leaf2parent_dict: dict[int, int],
    logger: logging.Logger,
    split_name: str,
) -> CategorySessionGraphDataset:
    prefixes, targets = build_prefix_target_pairs(session_sequences)

    valid_prefixes: list[list[int]] = []
    valid_targets: list[int] = []
    skipped = 0
    for prefix, target in zip(prefixes, targets):
        encoded_target = taxonomy.item_encoder.get(int(target))
        if encoded_target is None:
            skipped += 1
            continue
        valid_prefixes.append(prefix)
        valid_targets.append(target)

    if skipped:
        logger.warning("Split=%s | skipped_prefix_target_pairs=%d due to unknown target ids", split_name, skipped)

    return CategorySessionGraphDataset(
        session_sequences=valid_prefixes,
        item2leaf_dict=item2leaf_dict,
        leaf2parent_dict=leaf2parent_dict,
        targets=valid_targets,
        taxonomy=taxonomy,
    )


def resolve_eval_split(requested_eval_split: str, *, training_split: str, has_processed_artifact: bool) -> str | None:
    if requested_eval_split == "none":
        return None
    if requested_eval_split != "auto":
        return requested_eval_split

    if has_processed_artifact and training_split == "train":
        return "valid"
    return training_split


def evaluate_topk_metrics(
    model: CategoryEnhancedGNN,
    loader: DataLoader,
    *,
    device: torch.device,
    k: int,
) -> tuple[float, float, int]:
    model.eval()

    total_samples = 0
    total_hits = 0
    total_rr = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            logits = model(batch)
            target = batch.y.view(-1)

            current_k = min(k, logits.size(-1))
            topk_items = torch.topk(logits, k=current_k, dim=-1).indices
            matches = topk_items.eq(target.unsqueeze(1))

            hits = matches.any(dim=1)
            total_hits += int(hits.sum().item())

            reciprocal_ranks = torch.zeros(target.size(0), dtype=torch.float32, device=target.device)
            if hits.any():
                hit_rows = hits.nonzero(as_tuple=True)[0]
                hit_positions = matches[hit_rows].to(torch.int64).argmax(dim=1) + 1
                reciprocal_ranks[hit_rows] = 1.0 / hit_positions.to(torch.float32)
            total_rr += float(reciprocal_ranks.sum().item())
            total_samples += int(target.size(0))

    if total_samples == 0:
        return 0.0, 0.0, 0

    recall_pct = 100.0 * total_hits / total_samples
    mrr_pct = 100.0 * total_rr / total_samples
    return recall_pct, mrr_pct, total_samples


def select_monitor_split(eval_split: str | None) -> str:
    if eval_split is not None:
        return eval_split
    return "train"


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_dir, args.log_file_name)
    device = resolve_device(args.device)

    if device.type == "cuda":
        logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(device))
    else:
        logger.info("Using device: %s", device)

    if args.processed_path is not None:
        train_sequences, item2leaf_dict, leaf2parent_dict = load_preprocessed_artifacts(
            args.processed_path,
            split_name=args.split,
        )
        full_sequences, _, _ = load_preprocessed_artifacts(args.processed_path)
        logger.info("Loaded preprocessed artifact from %s | split=%s", args.processed_path.resolve(), args.split)
        session_sequences = train_sequences
        sequences_for_mapping = full_sequences
    elif args.data_root is None:
        session_sequences, item2leaf_dict, leaf2parent_dict = build_toy_example()
        logger.info("Using built-in toy example data")
        sequences_for_mapping = session_sequences
    else:
        session_sequences, item2leaf_dict, leaf2parent_dict = load_retailrocket_inputs(args.data_root)
        logger.info("Loaded RetailRocket data directly from raw CSV files: %s", args.data_root.resolve())
        sequences_for_mapping = session_sequences

    item2leaf_dict = apply_versioned_category_mapping(
        sequences_for_mapping,
        item2leaf_dict,
        version=args.version,
        logger=logger,
    )

    # Fit taxonomy từ FULL sessions trước để vocabulary chứa tất cả item
    # (bao gồm cả item chỉ xuất hiện ở cuối session làm target, chưa từng xuất hiện trong prefix)
    taxonomy = fit_taxonomy_encoders(sequences_for_mapping, item2leaf_dict, leaf2parent_dict)

    train_dataset = build_prefix_dataset(
        session_sequences,
        taxonomy=taxonomy,
        item2leaf_dict=item2leaf_dict,
        leaf2parent_dict=leaf2parent_dict,
        logger=logger,
        split_name=args.split,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    logger.info(
        "Dataset prepared | num_graphs=%d | num_items=%d | num_leaf_cats=%d | num_parent_cats=%d",
        len(train_dataset),
        train_dataset.taxonomy.num_items,
        train_dataset.taxonomy.num_leaf_cats,
        train_dataset.taxonomy.num_parent_cats,
    )

    eval_split = resolve_eval_split(
        args.eval_split,
        training_split=args.split,
        has_processed_artifact=args.processed_path is not None,
    )
    eval_loader: DataLoader | None = None
    if eval_split is not None:
        if args.processed_path is not None:
            eval_sequences, _, _ = load_preprocessed_artifacts(args.processed_path, split_name=eval_split)
        else:
            eval_sequences = session_sequences

        eval_dataset = build_prefix_dataset(
            eval_sequences,
            taxonomy=taxonomy,
            item2leaf_dict=item2leaf_dict,
            leaf2parent_dict=leaf2parent_dict,
            logger=logger,
            split_name=eval_split,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        logger.info(
            "Evaluation split prepared | split=%s | num_graphs=%d",
            eval_split,
            len(eval_dataset),
        )

    model = CategoryEnhancedGNN(
        num_items=train_dataset.taxonomy.num_items,
        num_leaf_cats=train_dataset.taxonomy.num_leaf_cats,
        num_parent_cats=train_dataset.taxonomy.num_parent_cats,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        conv_type=args.conv_type,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metric_k = max(args.metric_k, 1)
    logger.info("Metric config | Recall@%d and MRR@%d will be logged each epoch", metric_k, metric_k)

    early_stop_patience = max(args.early_stop_patience, 0)
    early_stop_min_delta = max(float(args.early_stop_min_delta), 0.0)
    monitor_split = select_monitor_split(eval_split)
    best_monitor_metric = float("-inf")
    best_epoch = 0
    no_improve_epochs = 0
    if early_stop_patience > 0:
        logger.info(
            "Early stopping enabled | monitor=%s_%s@%d | patience=%d | min_delta=%.6f",
            monitor_split,
            args.monitor_metric,
            metric_k,
            early_stop_patience,
            early_stop_min_delta,
        )
    else:
        logger.info("Early stopping disabled")

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        num_batches = max(len(train_loader), 1)
        log_every_steps = max(args.log_every_steps, 1)

        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = batch.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad()
            logits = model(batch)
            target = batch.y.view(-1)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

            if batch_idx % log_every_steps == 0 or batch_idx == num_batches:
                elapsed = max(time.time() - epoch_start, 1e-8)
                progress_pct = 100.0 * batch_idx / num_batches
                steps_per_sec = batch_idx / elapsed
                remaining_steps = max(num_batches - batch_idx, 0)
                eta_seconds = remaining_steps / max(steps_per_sec, 1e-8)
                running_loss = total_loss / batch_idx
                logger.info(
                    "epoch=%d progress=%d/%d (%.2f%%) running_loss=%.4f steps_per_sec=%.2f eta_sec=%.1f",
                    epoch + 1,
                    batch_idx,
                    num_batches,
                    progress_pct,
                    running_loss,
                    steps_per_sec,
                    eta_seconds,
                )

        mean_loss = total_loss / max(len(train_loader), 1)
        epoch_seconds = time.time() - epoch_start

        train_recall, train_mrr, train_metric_samples = evaluate_topk_metrics(
            model,
            train_loader,
            device=device,
            k=metric_k,
        )

        log_message = (
            "epoch=%d mean_loss=%.4f train_recall@%d=%.2f%% train_mrr@%d=%.2f%% "
            "train_metric_samples=%d epoch_time_sec=%.1f"
        )
        log_args: list[object] = [
            epoch + 1,
            mean_loss,
            metric_k,
            train_recall,
            metric_k,
            train_mrr,
            train_metric_samples,
            epoch_seconds,
        ]

        current_monitor_metric = train_mrr if args.monitor_metric == "mrr" else train_recall

        if eval_loader is not None and eval_split is not None:
            eval_recall, eval_mrr, eval_metric_samples = evaluate_topk_metrics(
                model,
                eval_loader,
                device=device,
                k=metric_k,
            )
            log_message += " %s_recall@%d=%.2f%% %s_mrr@%d=%.2f%% %s_metric_samples=%d"
            log_args.extend(
                [
                    eval_split,
                    metric_k,
                    eval_recall,
                    eval_split,
                    metric_k,
                    eval_mrr,
                    eval_split,
                    eval_metric_samples,
                ]
            )
            if monitor_split == eval_split:
                current_monitor_metric = eval_mrr if args.monitor_metric == "mrr" else eval_recall

        improved = current_monitor_metric > (best_monitor_metric + early_stop_min_delta)
        if improved:
            best_monitor_metric = current_monitor_metric
            best_epoch = epoch + 1
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        log_message += " monitor=%s_%s@%d=%.2f%% best=%.2f%% no_improve=%d"
        log_args.extend(
            [
                monitor_split,
                args.monitor_metric,
                metric_k,
                current_monitor_metric,
                best_monitor_metric,
                no_improve_epochs,
            ]
        )

        logger.info(log_message, *log_args)

        if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            logger.info(
                "Early stopping triggered at epoch=%d | best_epoch=%d | best_%s_%s@%d=%.2f%%",
                epoch + 1,
                best_epoch,
                monitor_split,
                args.monitor_metric,
                metric_k,
                best_monitor_metric,
            )
            break

    model.eval()
    sample_batch = next(iter(train_loader)).to(device, non_blocking=device.type == "cuda")
    with torch.no_grad():
        sample_logits = model(sample_batch)
        topk_scores, topk_items = torch.topk(sample_logits, k=min(5, sample_logits.size(-1)), dim=-1)

    logger.info("sample_topk_item_indices=%s", topk_items[0].tolist())
    logger.info("sample_topk_scores=%s", [round(score, 4) for score in topk_scores[0].tolist()])


if __name__ == "__main__":
    main()