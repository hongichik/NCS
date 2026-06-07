"""Convert CatSA RetailRocket artifact to COTREC pickle format."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def expand_session_samples(sessions: list[list[int]]) -> tuple[list[list[int]], list[int]]:
    """Prefix → next-item pairs (same logic as CatSA / SR-GNN preprocess)."""
    prefixes: list[list[int]] = []
    targets: list[int] = []
    for seq in sessions:
        if len(seq) < 2:
            continue
        for end in range(1, len(seq)):
            prefixes.append(list(seq[:end]))
            targets.append(int(seq[end]))
    return prefixes, targets


def pick_session_pools(
    splits: dict[str, list[list[int]]],
    *,
    split_mode: str,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """
    Return (train_sessions, test_sessions, all_train_graph_sessions).

    split_mode:
      - catsa: train on CatSA train split only (fair vs CatSA)
      - full: train on train+valid (closer to original COTREC train pool)
    """
    train_pool = list(splits["train"])
    valid_pool = list(splits.get("valid", []))
    test_pool = list(splits["test"])

    if split_mode == "catsa":
        train_sessions = train_pool
        graph_sessions = train_pool + valid_pool
    elif split_mode == "full":
        train_sessions = train_pool + valid_pool
        graph_sessions = train_sessions
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    return train_sessions, test_pool, graph_sessions


def build_cotrec_pickle(
    train_sessions: list[list[int]],
    test_sessions: list[list[int]],
    graph_sessions: list[list[int]],
) -> tuple[tuple[list, list], tuple[list, list], list[list[int]]]:
    train_prefixes, train_targets = expand_session_samples(train_sessions)
    test_prefixes, test_targets = expand_session_samples(test_sessions)
    train_data = (train_prefixes, train_targets)
    test_data = (test_prefixes, test_targets)
    return train_data, test_data, graph_sessions


def export_from_catsa_artifact(
    artifact_path: Path,
    output_dir: Path,
    *,
    split_mode: str = "catsa",
) -> dict[str, Any]:
    with artifact_path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)

    splits = artifact["splits"]
    train_sessions, test_sessions, graph_sessions = pick_session_pools(
        splits, split_mode=split_mode
    )
    train_data, test_data, all_train = build_cotrec_pickle(
        train_sessions, test_sessions, graph_sessions
    )

    num_items = int(artifact.get("num_items", 0))
    if num_items <= 0:
        all_ids = {item for seq in graph_sessions + test_sessions for item in seq}
        num_items = max(all_ids) if all_ids else 0

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "train.txt").open("wb") as handle:
        pickle.dump(train_data, handle)
    with (output_dir / "test.txt").open("wb") as handle:
        pickle.dump(test_data, handle)
    with (output_dir / "all_train_seq.txt").open("wb") as handle:
        pickle.dump(all_train, handle)

    meta = {
        "source": str(artifact_path),
        "protocol": artifact.get("protocol", "unknown"),
        "split_mode": split_mode,
        "num_items": num_items,
        "n_train_sessions": len(train_sessions),
        "n_test_sessions": len(test_sessions),
        "n_graph_sessions": len(graph_sessions),
        "n_train_samples": len(train_data[0]),
        "n_test_samples": len(test_data[0]),
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    return meta
