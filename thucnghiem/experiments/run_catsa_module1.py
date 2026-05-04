from __future__ import annotations

import argparse
import logging
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
    events_path = data_root / "events.csv"
    category_tree_path = data_root / "category_tree.csv"
    property_paths = [
        data_root / "item_properties_part1.csv",
        data_root / "item_properties_part2.csv",
    ]

    missing_paths = [path for path in [events_path, category_tree_path, *property_paths] if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing RetailRocket files: {missing}")

    session_sequences = build_sessions_from_events(events_path)
    item2leaf_dict = load_item_to_leaf_mapping(property_paths)
    leaf2parent_dict = load_leaf_to_parent_mapping(category_tree_path)
    return session_sequences, item2leaf_dict, leaf2parent_dict


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_dir, args.log_file_name)
    device = resolve_device(args.device)

    if device.type == "cuda":
        logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(device))
    else:
        logger.info("Using device: %s", device)

    if args.processed_path is not None:
        session_sequences, item2leaf_dict, leaf2parent_dict = load_preprocessed_artifacts(
            args.processed_path,
            split_name=args.split,
        )
        logger.info("Loaded preprocessed artifact from %s | split=%s", args.processed_path.resolve(), args.split)
    elif args.data_root is None:
        session_sequences, item2leaf_dict, leaf2parent_dict = build_toy_example()
        logger.info("Using built-in toy example data")
    else:
        session_sequences, item2leaf_dict, leaf2parent_dict = load_retailrocket_inputs(args.data_root)
        logger.info("Loaded RetailRocket data directly from raw CSV files: %s", args.data_root.resolve())

    item2leaf_dict = apply_versioned_category_mapping(
        session_sequences,
        item2leaf_dict,
        version=args.version,
        logger=logger,
    )

    # Fit taxonomy từ FULL sessions trước để vocabulary chứa tất cả item
    # (bao gồm cả item chỉ xuất hiện ở cuối session làm target, chưa từng xuất hiện trong prefix)
    taxonomy = fit_taxonomy_encoders(session_sequences, item2leaf_dict, leaf2parent_dict)

    prefixes, targets = build_prefix_target_pairs(session_sequences)

    # Encode targets sang chỉ số đã biết trong taxonomy
    # Loại bỏ các cặp (prefix, target) mà target không có trong vocab (edge case)
    valid_prefixes = []
    valid_targets = []
    skipped = 0
    for prefix, target in zip(prefixes, targets):
        encoded_target = taxonomy.item_encoder.get(int(target))
        if encoded_target is None:
            skipped += 1
            continue
        valid_prefixes.append(prefix)
        valid_targets.append(target)
    if skipped:
        logger.warning("Bỏ qua %d cặp prefix-target vì target không có trong vocabulary", skipped)

    dataset = CategorySessionGraphDataset(
        session_sequences=valid_prefixes,
        item2leaf_dict=item2leaf_dict,
        leaf2parent_dict=leaf2parent_dict,
        targets=valid_targets,
        taxonomy=taxonomy,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    logger.info(
        "Dataset prepared | num_graphs=%d | num_items=%d | num_leaf_cats=%d | num_parent_cats=%d",
        len(dataset),
        dataset.taxonomy.num_items,
        dataset.taxonomy.num_leaf_cats,
        dataset.taxonomy.num_parent_cats,
    )

    model = CategoryEnhancedGNN(
        num_items=dataset.taxonomy.num_items,
        num_leaf_cats=dataset.taxonomy.num_leaf_cats,
        num_parent_cats=dataset.taxonomy.num_parent_cats,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        conv_type=args.conv_type,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad()
            logits = model(batch)
            target = batch.y.view(-1)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        mean_loss = total_loss / max(len(loader), 1)
        logger.info("epoch=%d mean_loss=%.4f num_graphs=%d", epoch + 1, mean_loss, len(dataset))

    model.eval()
    sample_batch = next(iter(loader)).to(device, non_blocking=device.type == "cuda")
    with torch.no_grad():
        sample_logits = model(sample_batch)
        topk_scores, topk_items = torch.topk(sample_logits, k=min(5, sample_logits.size(-1)), dim=-1)

    logger.info("sample_topk_item_indices=%s", topk_items[0].tolist())
    logger.info("sample_topk_scores=%s", [round(score, 4) for score in topk_scores[0].tolist()])


if __name__ == "__main__":
    main()