"""
Preprocess RetailRocket for CatSA.

Protocol aligned with SR-GNN / COTREC baseline:
- view events only
- visitorid = session id
- item frequency >= 5
- last 7 days = test split
- train sessions split 70/20/10 -> train / valid / (held in train pool)

Outputs to Data/CatSA/retailrocket/:
- retailrocket.json (sessions + category maps + splits)
- lookup_tables.pkl (item2cat, cat2items, cat_parent, siblings, aug dicts)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import operator
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBLEM_NAME = "CatSA"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "Data" / "datagoc" / "Retailrocket"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Data" / PROBLEM_NAME / "retailrocket"
DEFAULT_LOG_DIR = REPO_ROOT / "Log" / PROBLEM_NAME / "retailrocket"

from preprocess.lookup_tables import (
    build_augmentation_dicts,
    build_cat2items,
    build_cat_parent,
    compute_siblings,
)


def resolve_retailrocket_paths(data_root: Path) -> tuple[Path, Path, list[Path], Path]:
    candidates = [data_root]
    if data_root.name.lower() == "retailrocket":
        candidates.append(data_root.parent)
    else:
        candidates.append(data_root / "Retailrocket")
        candidates.append(data_root / "retailrocket")

    for root in candidates:
        events = root / "events.csv"
        tree = root / "category_tree.csv"
        props = [root / "item_properties_part1.csv", root / "item_properties_part2.csv"]
        if events.exists() and tree.exists() and all(p.exists() for p in props):
            return events, tree, props, root

    events = data_root / "events.csv"
    tree = data_root / "category_tree.csv"
    props = [data_root / "item_properties_part1.csv", data_root / "item_properties_part2.csv"]
    return events, tree, props, data_root


def load_item_to_leaf(property_paths: Iterable[Path]) -> dict[int, int]:
    latest: dict[int, tuple[int, int]] = {}
    for path in property_paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("property") != "categoryid":
                    continue
                item_id = int(row["itemid"])
                ts = int(row["timestamp"])
                leaf = int(row["value"])
                cur = latest.get(item_id)
                if cur is None or ts >= cur[0]:
                    latest[item_id] = (ts, leaf)
    return {item: leaf for item, (_, leaf) in latest.items()}


def load_leaf_to_parent(category_tree_path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    with category_tree_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            leaf = int(row["categoryid"])
            parent_raw = row.get("parentid", "")
            if parent_raw in ("", None):
                continue
            mapping[leaf] = int(float(parent_raw))
    return mapping


def build_sessions_srgnn_style(events_path: Path) -> tuple[dict[int, list[int]], dict[int, float]]:
    """SR-GNN/COTREC: visitorid = session, view only."""
    sess_clicks: dict[int, list[int]] = {}
    sess_date: dict[int, float] = {}

    with events_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("event") != "view":
                continue
            sess_id = int(row["visitorid"])
            item_id = int(row["itemid"])
            ts = int(row["timestamp"]) / 1000.0
            if sess_id in sess_clicks:
                sess_clicks[sess_id].append(item_id)
            else:
                sess_clicks[sess_id] = [item_id]
            sess_date[sess_id] = ts

    for sid in list(sess_clicks):
        if len(sess_clicks[sid]) == 1:
            del sess_clicks[sid]
            del sess_date[sid]

    item_counts: dict[int, int] = defaultdict(int)
    for seq in sess_clicks.values():
        for item in seq:
            item_counts[item] += 1

    for sid in list(sess_clicks):
        filtered = [i for i in sess_clicks[sid] if item_counts[i] >= 5]
        if len(filtered) < 2:
            del sess_clicks[sid]
            del sess_date[sid]
        else:
            sess_clicks[sid] = filtered

    return sess_clicks, sess_date


def split_train_test(
    sess_clicks: dict[int, list[int]],
    sess_date: dict[int, float],
    *,
    test_days: int = 7,
) -> tuple[list[tuple[int, list[int]]], list[tuple[int, list[int]]]]:
    dates = list(sess_date.items())
    maxdate = max(d for _, d in dates)
    splitdate = maxdate - 86400 * test_days

    tra = sorted(
        [(sid, sess_clicks[sid]) for sid, d in dates if d < splitdate],
        key=lambda x: sess_date[x[0]],
    )
    tes = sorted(
        [(sid, sess_clicks[sid]) for sid, d in dates if d > splitdate],
        key=lambda x: sess_date[x[0]],
    )
    return tra, tes


def renumber_sessions(
    tra: list[tuple[int, list[int]]],
    tes: list[tuple[int, list[int]]],
) -> tuple[list[list[int]], list[list[int]], dict[int, int], dict[int, int]]:
    """Map raw item ids to 1-based indices (COTREC convention)."""
    item_dict: dict[int, int] = {}
    counter = 1

    train_seqs: list[list[int]] = []
    for _, seq in tra:
        out = []
        for raw in seq:
            if raw not in item_dict:
                item_dict[raw] = counter
                counter += 1
            out.append(item_dict[raw])
        if len(out) >= 2:
            train_seqs.append(out)

    test_seqs: list[list[int]] = []
    for _, seq in tes:
        out = [item_dict[raw] for raw in seq if raw in item_dict]
        if len(out) >= 2:
            test_seqs.append(out)

    raw_to_idx = {raw: idx for raw, idx in item_dict.items()}
    idx_to_raw = {idx: raw for raw, idx in item_dict.items()}
    return train_seqs, test_seqs, raw_to_idx, idx_to_raw


def split_train_valid(
    train_seqs: list[list[int]],
    *,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.2,
) -> dict[str, list[list[int]]]:
    n = len(train_seqs)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return {
        "train": train_seqs[:train_end],
        "valid": train_seqs[train_end:valid_end],
        "test": [],  # filled by caller
    }


def remap_category_maps(
    raw_item2leaf: dict[int, int],
    raw_leaf2parent: dict[int, int],
    raw_to_idx: dict[int, int],
    train_items: set[int],
) -> tuple[dict[int, int], dict[int, int]]:
    """Remap to 0-based item idx; only train+valid items for lookup (no leakage)."""
    item2cat: dict[int, int] = {}
    for raw_item, leaf in raw_item2leaf.items():
        if raw_item not in raw_to_idx:
            continue
        idx = raw_to_idx[raw_item] - 1  # 0-based for embedding
        if idx in train_items:
            item2cat[idx] = leaf

    leaf_ids = sorted({c for c in item2cat.values()})
    leaf_remap = {leaf: i for i, leaf in enumerate(leaf_ids)}
    item2cat_idx = {item: leaf_remap[cat] for item, cat in item2cat.items()}

    parent_ids = sorted(
        {raw_leaf2parent[leaf] for leaf in leaf_ids if leaf in raw_leaf2parent}
    )
    parent_remap = {p: i for i, p in enumerate(parent_ids)}

    cat_parent_idx: dict[int, int] = {}
    for leaf, parent in raw_leaf2parent.items():
        if leaf in leaf_remap and parent in parent_remap:
            cat_parent_idx[leaf_remap[leaf]] = parent_remap[parent]

    return item2cat_idx, cat_parent_idx


def configure_logging(log_dir: Path, log_file_name: str) -> logging.Logger:
    logger = logging.getLogger("catsa.preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / log_file_name, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess RetailRocket for CatSA.")
    p.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument("--test-days", type=int, default=7)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--valid-ratio", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_name = datetime.now().strftime("%d-%m-%Y") + ".log"
    logger = configure_logging(args.log_dir, log_name)

    events_path, tree_path, prop_paths, resolved = resolve_retailrocket_paths(args.source_root)
    missing = [p for p in [events_path, tree_path, *prop_paths] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(str(p) for p in missing))

    logger.info("source_root=%s (resolved=%s)", args.source_root, resolved)

    sess_clicks, sess_date = build_sessions_srgnn_style(events_path)
    logger.info("sessions_after_filter=%d", len(sess_clicks))

    tra, tes = split_train_test(sess_clicks, sess_date, test_days=args.test_days)
    train_seqs, test_seqs, raw_to_idx, idx_to_raw = renumber_sessions(tra, tes)
    logger.info("train_sessions=%d test_sessions=%d num_items=%d", len(train_seqs), len(test_seqs), len(raw_to_idx))

    splits = split_train_valid(train_seqs, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    splits["test"] = test_seqs

    raw_item2leaf = load_item_to_leaf(prop_paths)
    raw_leaf2parent = load_leaf_to_parent(tree_path)

    train_valid_items = set()
    for split in ("train", "valid"):
        for seq in splits[split]:
            for idx in seq:
                train_valid_items.add(idx - 1)

    item2cat, cat_parent = remap_category_maps(
        raw_item2leaf, raw_leaf2parent, raw_to_idx, train_valid_items
    )
    cat2items = build_cat2items(item2cat)
    siblings = compute_siblings(cat_parent)
    dict_same, dict_sibling = build_augmentation_dicts(item2cat, cat2items, siblings)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "version": 1,
        "protocol": "SR-GNN/COTREC",
        "num_items": len(raw_to_idx),
        "splits": splits,
        "item2cat": item2cat,
        "cat_parent": cat_parent,
        "raw_to_idx": {str(k): v for k, v in raw_to_idx.items()},
    }
    json_path = args.output_dir / "retailrocket.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f)

    lookup_path = args.output_dir / "lookup_tables.pkl"
    with lookup_path.open("wb") as f:
        pickle.dump(
            {
                "item2cat": item2cat,
                "cat2items": cat2items,
                "cat_parent": cat_parent,
                "siblings": siblings,
                "dict_same_leaf": dict_same,
                "dict_sibling": dict_sibling,
                "idx_to_raw": idx_to_raw,
            },
            f,
        )

    logger.info("saved %s", json_path)
    logger.info("saved %s", lookup_path)
    logger.info("|C|=%d items_with_cat=%d", len(cat2items), len(item2cat))
    for name, seqs in splits.items():
        logger.info("split %s: %d sessions", name, len(seqs))


if __name__ == "__main__":
    main()
