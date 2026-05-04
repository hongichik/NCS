from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable


@dataclass(frozen=True)
class SessionRecord:
    sequence: list[int]
    end_timestamp: int


def load_leaf_to_parent_mapping(category_tree_path: str | Path) -> dict[int, int]:
    """Load leaf -> parent edges from RetailRocket category_tree.csv."""
    leaf_to_parent: dict[int, int] = {}
    with Path(category_tree_path).open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            category_id = int(row["categoryid"])
            parent_raw = row.get("parentid", "")
            if parent_raw is None or parent_raw == "":
                continue
            parent_id = int(float(parent_raw))
            leaf_to_parent[category_id] = parent_id
    return leaf_to_parent


def load_item_to_leaf_mapping(property_paths: Iterable[str | Path]) -> dict[int, int]:
    """
    Load the latest known item -> leaf category mapping from RetailRocket item_properties files.

    RetailRocket stores category assignments as rows where property == "categoryid".
    If an item has multiple assignments over time, the newest timestamp is retained.
    """
    latest_assignment: dict[int, tuple[int, int]] = {}

    for property_path in property_paths:
        with Path(property_path).open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row.get("property") != "categoryid":
                    continue
                item_id = int(row["itemid"])
                timestamp = int(row["timestamp"])
                leaf_cat_id = int(row["value"])
                current = latest_assignment.get(item_id)
                if current is None or timestamp >= current[0]:
                    latest_assignment[item_id] = (timestamp, leaf_cat_id)

    return {item_id: leaf_cat_id for item_id, (_, leaf_cat_id) in latest_assignment.items()}


def build_sessions_from_events(
    events_path: str | Path,
    *,
    allowed_events: tuple[str, ...] = ("view",),
    min_session_length: int = 2,
    session_gap_minutes: int = 30,
) -> list[list[int]]:
    """
    Convert RetailRocket events.csv into session item sequences.

    Sessions are split per visitor when the timestamp gap exceeds `session_gap_minutes`.
    The returned sequences only contain item ids and are sorted chronologically within each user.
    """
    session_records = build_session_records_from_events(
        events_path,
        allowed_events=allowed_events,
        min_session_length=min_session_length,
        session_gap_minutes=session_gap_minutes,
    )
    return [record.sequence for record in session_records]


def build_session_records_from_events(
    events_path: str | Path,
    *,
    allowed_events: tuple[str, ...] = ("view",),
    min_session_length: int = 2,
    session_gap_minutes: int = 30,
) -> list[SessionRecord]:
    per_visitor_events: DefaultDict[int, list[tuple[int, int]]] = defaultdict(list)

    with Path(events_path).open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            event_type = row.get("event")
            if event_type not in allowed_events:
                continue
            visitor_id = int(row["visitorid"])
            timestamp = int(row["timestamp"])
            item_id = int(row["itemid"])
            per_visitor_events[visitor_id].append((timestamp, item_id))

    max_gap_ms = session_gap_minutes * 60 * 1000
    sessions: list[SessionRecord] = []

    for visitor_events in per_visitor_events.values():
        visitor_events.sort(key=lambda pair: pair[0])
        current_session: list[int] = []
        previous_timestamp: int | None = None

        for timestamp, item_id in visitor_events:
            if previous_timestamp is None or timestamp - previous_timestamp <= max_gap_ms:
                current_session.append(item_id)
            else:
                if len(current_session) >= min_session_length:
                    sessions.append(SessionRecord(sequence=current_session, end_timestamp=previous_timestamp))
                current_session = [item_id]
            previous_timestamp = timestamp

        if len(current_session) >= min_session_length and previous_timestamp is not None:
            sessions.append(SessionRecord(sequence=current_session, end_timestamp=previous_timestamp))

    return sessions


def split_session_records_by_time(
    session_records: list[SessionRecord],
    *,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> dict[str, list[SessionRecord]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in the interval (0, 1)")
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError("valid_ratio must be in the interval [0, 1)")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be smaller than 1.0")

    ordered_records = sorted(session_records, key=lambda record: record.end_timestamp)
    num_sessions = len(ordered_records)
    train_end = int(num_sessions * train_ratio)
    valid_end = int(num_sessions * (train_ratio + valid_ratio))

    return {
        "train": ordered_records[:train_end],
        "valid": ordered_records[train_end:valid_end],
        "test": ordered_records[valid_end:],
    }


def filter_session_records_by_item_clicks(
    session_records: list[SessionRecord],
    *,
    min_item_clicks: int,
    min_session_length: int,
) -> tuple[list[SessionRecord], dict[str, int]]:
    """
    Bước 1 – Lọc item hiếm:
      Đếm tần suất click toàn cục. Item nào có số click < min_item_clicks bị
      xóa khỏi chuỗi (chỉ xóa item đó, không xóa cả chuỗi). Thứ tự các item
      còn lại trong chuỗi được giữ nguyên.

    Bước 2 – Lọc session không hợp lệ:
      Sau khi xóa item hiếm, chuỗi nào còn độ dài < min_session_length sẽ bị
      loại hoàn toàn.

    Item không có danh mục: vẫn được GIỮ NGUYÊN trong chuỗi và không bị lọc
    ở bước này. Khi xây dựng đồ thị, các item đó chỉ có cạnh sequential.
    """
    if min_item_clicks < 1:
        raise ValueError("min_item_clicks must be >= 1")

    # Bước 1a: đếm tần suất click toàn cục trên toàn bộ dữ liệu
    item_click_counts: dict[int, int] = defaultdict(int)
    for record in session_records:
        for item_id in record.sequence:
            item_click_counts[item_id] += 1

    # Bước 1b: xác định tập item đủ điều kiện (click >= min_item_clicks)
    # item có count < min_item_clicks bị coi là "hiếm" và bị xóa khỏi chuỗi
    kept_item_ids = {
        item_id for item_id, count in item_click_counts.items() if count >= min_item_clicks
    }

    # Bước 2: lọc từng chuỗi, xóa item hiếm, giữ thứ tự; loại chuỗi quá ngắn
    filtered_records: list[SessionRecord] = []
    removed_sessions_due_to_length = 0
    for record in session_records:
        # Xóa item hiếm khỏi chuỗi; các item còn lại giữ nguyên vị trí
        filtered_sequence = [item_id for item_id in record.sequence if item_id in kept_item_ids]
        if len(filtered_sequence) < min_session_length:
            # Chuỗi bị ngắn hơn ngưỡng sau lọc → loại cả chuỗi
            removed_sessions_due_to_length += 1
            continue
        filtered_records.append(SessionRecord(sequence=filtered_sequence, end_timestamp=record.end_timestamp))

    stats = {
        "num_items_before_frequency_filter": len(item_click_counts),
        "num_items_after_frequency_filter": len(kept_item_ids),
        "num_removed_rare_items": len(item_click_counts) - len(kept_item_ids),
        "num_sessions_before_frequency_filter": len(session_records),
        "num_sessions_after_frequency_filter": len(filtered_records),
        "num_removed_sessions_after_item_filter": removed_sessions_due_to_length,
    }
    return filtered_records, stats


def save_preprocessed_artifacts(
    output_path: str | Path,
    *,
    session_sequences: list[list[int]],
    item2leaf_dict: dict[int, int],
    leaf2parent_dict: dict[int, int],
    split_session_sequences: dict[str, list[list[int]]] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": 2,
        "session_sequences": session_sequences,
        "item2leaf_dict": item2leaf_dict,
        "leaf2parent_dict": leaf2parent_dict,
    }
    if split_session_sequences is not None:
        payload["splits"] = split_session_sequences
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False)
    return output_path


def load_preprocessed_artifacts(
    input_path: str | Path,
    *,
    split_name: str | None = None,
) -> tuple[list[list[int]], dict[int, int], dict[int, int]]:
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if split_name is not None and "splits" in payload:
        try:
            split_payload = payload["splits"][split_name]
        except KeyError as exc:
            available = ", ".join(sorted(payload["splits"].keys()))
            raise KeyError(f"Unknown split '{split_name}'. Available splits: {available}") from exc
        session_sequences = [list(map(int, session)) for session in split_payload]
    else:
        session_sequences = [list(map(int, session)) for session in payload["session_sequences"]]

    item2leaf_dict = {int(item_id): int(leaf_id) for item_id, leaf_id in payload["item2leaf_dict"].items()}
    leaf2parent_dict = {int(leaf_id): int(parent_id) for leaf_id, parent_id in payload["leaf2parent_dict"].items()}
    return session_sequences, item2leaf_dict, leaf2parent_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess RetailRocket data for CatSA Module 1.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to DATA/retailrocket")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/processed/retailrocket_module1.json"),
        help="Path to the saved preprocessed artifact.",
    )
    parser.add_argument(
        "--allowed-events",
        nargs="+",
        default=["view"],
        help="RetailRocket event types used to build sessions.",
    )
    parser.add_argument("--min-session-length", type=int, default=2)
    parser.add_argument("--session-gap-minutes", type=int, default=30)
    parser.add_argument(
        "--min-item-clicks",
        type=int,
        default=5,
        help=(
            "Ngưỡng tần suất click tối thiểu để giữ một item. "
            "Item có số click < giá trị này bị coi là 'hiếm' và bị xóa khỏi chuỗi. "
            "Mặc định: 5 (giữ item có click >= 5)."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory used to store preprocessing logs. If omitted, logs are only printed to stdout.",
    )
    parser.add_argument(
        "--log-file-name",
        type=str,
        default="retailrocket_preprocess.log",
        help="Log file name created inside --log-dir.",
    )
    return parser.parse_args()


def configure_logging(log_dir: Path | None, log_file_name: str) -> logging.Logger:
    logger = logging.getLogger("catsa.preprocess")
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


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_dir, args.log_file_name)

    data_root = args.data_root
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

    session_records = build_session_records_from_events(
        events_path,
        allowed_events=tuple(args.allowed_events),
        min_session_length=args.min_session_length,
        session_gap_minutes=args.session_gap_minutes,
    )

    filtered_session_records, frequency_filter_stats = filter_session_records_by_item_clicks(
        session_records,
        min_item_clicks=args.min_item_clicks,
        min_session_length=args.min_session_length,
    )

    session_records = filtered_session_records
    session_sequences = [record.sequence for record in session_records]
    split_records = split_session_records_by_time(
        session_records,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )
    split_session_sequences = {
        split_name: [record.sequence for record in records]
        for split_name, records in split_records.items()
    }
    item2leaf_dict = load_item_to_leaf_mapping(property_paths)
    leaf2parent_dict = load_leaf_to_parent_mapping(category_tree_path)

    output_path = save_preprocessed_artifacts(
        args.output_path,
        session_sequences=session_sequences,
        item2leaf_dict=item2leaf_dict,
        leaf2parent_dict=leaf2parent_dict,
        split_session_sequences=split_session_sequences,
    )

    logger.info("Saved preprocessed artifact to %s", output_path.resolve())
    logger.info("allowed_events=%s", ",".join(args.allowed_events))
    logger.info("min_item_clicks=%d  (giữ item có click >= %d, hiếm là < %d)", args.min_item_clicks, args.min_item_clicks, args.min_item_clicks)
    logger.info("num_sessions=%d", len(session_sequences))
    logger.info("num_train_sessions=%d", len(split_session_sequences["train"]))
    logger.info("num_valid_sessions=%d", len(split_session_sequences["valid"]))
    logger.info("num_test_sessions=%d", len(split_session_sequences["test"]))
    logger.info("num_items_before_frequency_filter=%d", frequency_filter_stats["num_items_before_frequency_filter"])
    logger.info("num_items_after_frequency_filter=%d", frequency_filter_stats["num_items_after_frequency_filter"])
    logger.info("num_removed_rare_items=%d", frequency_filter_stats["num_removed_rare_items"])
    logger.info(
        "num_removed_sessions_after_item_filter=%d",
        frequency_filter_stats["num_removed_sessions_after_item_filter"],
    )
    logger.info("num_items_with_leaf=%d", len(item2leaf_dict))
    logger.info("num_leaf_to_parent=%d", len(leaf2parent_dict))


if __name__ == "__main__":
    main()