"""Module 2: Category-Structure-Guided Augmentation."""

from __future__ import annotations

import random
from typing import Mapping, Sequence


def same_leaf_augment(
    session: Sequence[int],
    item2cat: Mapping[int, int],
    cat2items: Mapping[int, list[int]],
    *,
    eta_aug: float = 0.3,
    k_min: int = 5,
) -> list[int]:
    n = len(session)
    k = max(1, round(eta_aug * n))
    positions = random.sample(range(n), min(k, n))
    augmented = list(session)
    for pos in positions:
        item = int(session[pos])
        cat = item2cat.get(item)
        if cat is None:
            continue
        candidates = cat2items.get(int(cat), [])
        if len(candidates) < k_min:
            continue
        valid = [c for c in candidates if c != item]
        if not valid:
            continue
        augmented[pos] = random.choice(valid)
    return augmented


def sibling_leaf_augment(
    session: Sequence[int],
    item2cat: Mapping[int, int],
    cat2items: Mapping[int, list[int]],
    siblings_table: Mapping[int, list[int]],
    *,
    eta_aug: float = 0.3,
    k_min: int = 5,
) -> list[int]:
    n = len(session)
    k = max(1, round(eta_aug * n))
    positions = random.sample(range(n), min(k, n))
    augmented = list(session)
    for pos in positions:
        item = int(session[pos])
        cat = item2cat.get(item)
        if cat is None:
            continue
        sibling_cats = siblings_table.get(int(cat), [])
        if not sibling_cats:
            augmented[pos] = same_leaf_augment([item], item2cat, cat2items, eta_aug=1.0, k_min=k_min)[0]
            continue
        sib_cat = random.choice(sibling_cats)
        sib_candidates = cat2items.get(int(sib_cat), [])
        if len(sib_candidates) < k_min:
            continue
        augmented[pos] = random.choice(sib_candidates)
    return augmented


def hybrid_augment(
    session: Sequence[int],
    item2cat: Mapping[int, int],
    cat2items: Mapping[int, list[int]],
    *,
    eta_aug: float = 0.3,
    eta_crop: float = 0.75,
    k_min: int = 5,
) -> list[int]:
    augmented = same_leaf_augment(session, item2cat, cat2items, eta_aug=eta_aug, k_min=k_min)
    n = len(augmented)
    crop_len = max(1, int(n * eta_crop))
    start = random.randint(0, max(0, n - crop_len))
    return augmented[start : start + crop_len]


def catsa_augment(
    session: Sequence[int],
    item2cat: Mapping[int, int],
    cat2items: Mapping[int, list[int]],
    siblings_table: Mapping[int, list[int]] | None = None,
    *,
    eta_aug: float = 0.3,
    k_min: int = 5,
) -> list[int]:
    strategies = ["same", "hybrid"]
    if siblings_table:
        strategies.append("sibling")
    strategy = random.choice(strategies)
    if strategy == "same":
        return same_leaf_augment(session, item2cat, cat2items, eta_aug=eta_aug, k_min=k_min)
    if strategy == "sibling" and siblings_table:
        return sibling_leaf_augment(
            session, item2cat, cat2items, siblings_table, eta_aug=eta_aug, k_min=k_min
        )
    return hybrid_augment(session, item2cat, cat2items, eta_aug=eta_aug, k_min=k_min)
