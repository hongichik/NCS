"""Build CatSA lookup tables: item2cat, cat2items, cat_parent, siblings."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping


def build_cat2items(item2cat: Mapping[int, int]) -> dict[int, list[int]]:
    cat2items: dict[int, list[int]] = defaultdict(list)
    for item_id, cat_id in item2cat.items():
        cat2items[int(cat_id)].append(int(item_id))
    return {cat_id: sorted(items) for cat_id, items in cat2items.items()}


def build_cat_parent(leaf2parent: Mapping[int, int]) -> dict[int, int]:
    return {int(leaf): int(parent) for leaf, parent in leaf2parent.items()}


def compute_siblings(cat_parent: Mapping[int, int]) -> dict[int, list[int]]:
    parent_to_children: dict[int, list[int]] = defaultdict(list)
    for cat, parent in cat_parent.items():
        parent_to_children[int(parent)].append(int(cat))

    siblings: dict[int, list[int]] = {}
    for cat, parent in cat_parent.items():
        children = parent_to_children[int(parent)]
        siblings[int(cat)] = [c for c in children if c != int(cat)]
    return siblings


def build_augmentation_dicts(
    item2cat: Mapping[int, int],
    cat2items: Mapping[int, list[int]],
    siblings_table: Mapping[int, list[int]],
    *,
    max_candidates: int = 20,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Precompute same-leaf and sibling-leaf candidate lists per item."""
    import random

    dict_same_leaf: dict[int, list[int]] = {}
    for item_id, cat_id in item2cat.items():
        candidates = [i for i in cat2items.get(int(cat_id), []) if i != int(item_id)]
        if candidates:
            if len(candidates) > max_candidates:
                candidates = random.sample(candidates, max_candidates)
            dict_same_leaf[int(item_id)] = candidates

    dict_sibling: dict[int, list[int]] = {}
    for item_id, cat_id in item2cat.items():
        sibling_cats = siblings_table.get(int(cat_id), [])
        sibling_items: list[int] = []
        for sib_cat in sibling_cats:
            sibling_items.extend(cat2items.get(int(sib_cat), []))
        if sibling_items:
            if len(sibling_items) > max_candidates:
                sibling_items = random.sample(sibling_items, max_candidates)
            dict_sibling[int(item_id)] = sibling_items

    return dict_same_leaf, dict_sibling
