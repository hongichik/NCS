import csv
import os
import pickle
from collections import defaultdict


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "retailrocket")
OUT_DIR = os.path.join(DATA_DIR, "id")

# 30 minutes in milliseconds.
SESSION_GAP_MS = 30 * 60 * 1000
MIN_SESSION_LEN = 2
MIN_ITEM_FREQ = 5
MAX_SESSION_LEN = 200


def read_item_to_category():
    """Build item -> category map using latest categoryid assignment by timestamp."""
    item_to_cat = {}
    item_to_ts = {}
    for filename in ("item_properties_part1.csv", "item_properties_part2.csv"):
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["property"] != "categoryid":
                    continue
                try:
                    ts = int(row["timestamp"])
                    item_id = int(row["itemid"])
                    cat_id = int(row["value"])
                except ValueError:
                    continue
                prev_ts = item_to_ts.get(item_id, -1)
                if ts >= prev_ts:
                    item_to_ts[item_id] = ts
                    item_to_cat[item_id] = cat_id
    return item_to_cat


def build_sessions(item_to_cat):
    """
    Build sessions from events.csv using visitor-based 30-min inactivity split.
    Keep events that have category information.
    """
    active = {}
    sessions = []
    categories = []
    path = os.path.join(DATA_DIR, "events.csv")
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = int(row["timestamp"])
                visitor = int(row["visitorid"])
                item_id = int(row["itemid"])
            except ValueError:
                continue

            cat_id = item_to_cat.get(item_id)
            if cat_id is None:
                continue

            if visitor not in active:
                active[visitor] = [ts, [item_id], [cat_id]]
                continue

            last_ts, items, cats = active[visitor]
            if ts - last_ts > SESSION_GAP_MS:
                if len(items) >= MIN_SESSION_LEN:
                    sessions.append(items)
                    categories.append(cats)
                active[visitor] = [ts, [item_id], [cat_id]]
            else:
                items.append(item_id)
                cats.append(cat_id)
                active[visitor][0] = ts

    for _, (_, items, cats) in active.items():
        if len(items) >= MIN_SESSION_LEN:
            sessions.append(items)
            categories.append(cats)
    return sessions, categories


def filter_rare_items(sessions, categories, min_freq=MIN_ITEM_FREQ):
    freq = defaultdict(int)
    for seq in sessions:
        for item in seq:
            freq[item] += 1

    filtered_sessions = []
    filtered_categories = []
    for seq, cat_seq in zip(sessions, categories):
        new_seq = []
        new_cat = []
        for item, cat in zip(seq, cat_seq):
            if freq[item] >= min_freq:
                new_seq.append(item)
                new_cat.append(cat)
        if len(new_seq) >= MIN_SESSION_LEN:
            filtered_sessions.append(new_seq)
            filtered_categories.append(new_cat)
    return filtered_sessions, filtered_categories


def split_train_test_one(item_sessions, cat_sessions, max_len=MAX_SESSION_LEN):
    train_item, train_item_t = [], []
    test_item, test_item_t = [], []
    train_cat, train_cat_t = [], []
    test_cat, test_cat_t = [], []

    for items, cats in zip(item_sessions, cat_sessions):
        if len(items) != len(cats) or len(items) < 2:
            continue

        item_prefixes, item_targets = [], []
        cat_prefixes, cat_targets = [], []

        n = len(items)
        if n < max_len:
            start_points = [0]
        else:
            start_points = list(range(0, n - max_len + 1))

        for start in start_points:
            if start == 0:
                upper = min(n, max_len)
                for i in range(1, upper):
                    item_prefixes.append(items[:i])
                    item_targets.append(items[i])
                    cat_prefixes.append(cats[:i])
                    cat_targets.append(cats[i])
            else:
                prefix = items[start : start + max_len - 1]
                target = items[start + max_len - 1]
                item_prefixes.append(prefix)
                item_targets.append(target)

                cat_prefix = cats[start : start + max_len - 1]
                cat_target = cats[start + max_len - 1]
                cat_prefixes.append(cat_prefix)
                cat_targets.append(cat_target)

        if not item_prefixes:
            continue

        for i in range(len(item_prefixes) - 1):
            train_item.append(item_prefixes[i])
            train_item_t.append(item_targets[i])
            train_cat.append(cat_prefixes[i])
            train_cat_t.append(cat_targets[i])

        test_item.append(item_prefixes[-1])
        test_item_t.append(item_targets[-1])
        test_cat.append(cat_prefixes[-1])
        test_cat_t.append(cat_targets[-1])

    return (
        [train_item, train_item_t],
        [test_item, test_item_t],
        [train_cat, train_cat_t],
        [test_cat, test_cat_t],
    )


def build_id_map(train_sessions, test_sessions, train_targets=None, test_targets=None):
    freq = defaultdict(int)
    for seq in train_sessions:
        for x in seq:
            freq[x] += 1
    for seq in test_sessions:
        for x in seq:
            freq[x] += 1

    if train_targets is not None:
        for x in train_targets:
            freq[x] += 1
    if test_targets is not None:
        for x in test_targets:
            freq[x] += 1

    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    mapping = {}
    idx = 1
    for key, _ in ranked:
        mapping[key] = idx
        idx += 1
    return mapping


def remap_sequences(seqs, mapping):
    out = []
    for seq in seqs:
        out.append([mapping[x] for x in seq])
    return out


def remap_targets(targets, mapping):
    return [mapping[x] for x in targets]


def main():
    print("[1/6] build item->category")
    item_to_cat = read_item_to_category()
    print("items with category:", len(item_to_cat))

    print("[2/6] build sessions")
    sessions, categories = build_sessions(item_to_cat)
    print("sessions before filter:", len(sessions))

    print("[3/6] filter rare items")
    sessions, categories = filter_rare_items(sessions, categories)
    print("sessions after filter:", len(sessions))

    print("[4/6] split train/test")
    train_i, test_i, train_c, test_c = split_train_test_one(sessions, categories)
    print("train samples:", len(train_i[0]), "test samples:", len(test_i[0]))

    print("[5/6] remap to dense ids")
    item_map = build_id_map(train_i[0], test_i[0], train_i[1], test_i[1])
    cat_map = build_id_map(train_c[0], test_c[0], train_c[1], test_c[1])

    train_i = [remap_sequences(train_i[0], item_map), remap_targets(train_i[1], item_map)]
    test_i = [remap_sequences(test_i[0], item_map), remap_targets(test_i[1], item_map)]
    train_c = [remap_sequences(train_c[0], cat_map), remap_targets(train_c[1], cat_map)]
    test_c = [remap_sequences(test_c[0], cat_map), remap_targets(test_c[1], cat_map)]

    print("item nodes:", len(item_map), "category nodes:", len(cat_map))

    print("[6/6] write pickles")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "train.txt"), "wb") as f:
        pickle.dump(train_i, f)
    with open(os.path.join(OUT_DIR, "test.txt"), "wb") as f:
        pickle.dump(test_i, f)
    with open(os.path.join(OUT_DIR, "category_train.txt"), "wb") as f:
        pickle.dump(train_c, f)
    with open(os.path.join(OUT_DIR, "category_test.txt"), "wb") as f:
        pickle.dump(test_c, f)

    print("done:", OUT_DIR)


if __name__ == "__main__":
    main()
