"""Load preprocessed CatSA artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def load_artifact(data_dir: Path) -> dict[str, Any]:
    json_path = data_dir / "retailrocket.json"
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_lookup_tables(data_dir: Path) -> dict[str, Any]:
    with (data_dir / "lookup_tables.pkl").open("rb") as f:
        return pickle.load(f)


def sessions_to_1based(sessions: list[list[int]]) -> list[list[int]]:
    """CatSA internal uses 0-based item idx; sessions stored as 1-based (COTREC)."""
    return sessions


def remap_item2cat_1based_to_0based(item2cat: dict[int, int]) -> dict[int, int]:
    return {int(k): int(v) for k, v in item2cat.items()}


def remap_sessions_for_taxonomy(
    prefixes: list[list[int]],
    item2cat_0based: dict[int, int],
    cat_parent: dict[int, int],
) -> tuple[dict[int, int], dict[int, int]]:
    """Build leaf/parent dicts keyed by 1-based item id for dataset encoder."""
    item2leaf_1based: dict[int, int] = {}
    for item_0, leaf in item2cat_0based.items():
        item2leaf_1based[item_0 + 1] = leaf
    return item2leaf_1based, cat_parent
