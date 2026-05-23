#!/usr/bin/env python3
"""Convert raw clickstream CSV to RecBole .inter format.

Expected input columns (header):
timestamp,visitorid,event,itemid,transactionid

Only rows with event == "view" are kept.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw CSV to RecBole .inter (filter event=view)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output .inter file.",
    )
    parser.add_argument(
        "--event",
        type=str,
        default="view",
        help="Event type to keep (default: view).",
    )
    parser.add_argument(
        "--sort-by-time",
        action="store_true",
        help="Sort output by timestamp ascending.",
    )
    return parser.parse_args()


def _norm_text(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def convert_csv_to_inter(
    input_path: Path,
    output_path: Path,
    keep_event: str = "view",
    sort_by_time: bool = False,
) -> tuple[int, int]:
    keep_event = keep_event.strip().lower()

    rows: list[tuple[str, str, float, float]] = []
    total_rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"timestamp", "visitorid", "event", "itemid"}
        missing = required_cols.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Input CSV missing required columns: " + ", ".join(sorted(missing))
            )

        for raw in reader:
            total_rows += 1
            event = _norm_text(raw.get("event")).lower()
            if event != keep_event:
                continue

            user_id = _norm_text(raw.get("visitorid"))
            item_id = _norm_text(raw.get("itemid"))
            timestamp_text = _norm_text(raw.get("timestamp"))

            if not user_id or not item_id or not timestamp_text:
                continue

            try:
                timestamp = float(timestamp_text)
            except ValueError:
                continue

            # Use implicit feedback score for view events.
            rating = 1.0
            rows.append((user_id, item_id, rating, timestamp))

    if sort_by_time:
        rows.sort(key=lambda x: x[3])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "user_id:token",
                "item_id:token",
                "rating:float",
                "timestamp:float",
            ]
        )
        writer.writerows(rows)

    return total_rows, len(rows)


def main() -> None:
    args = parse_args()
    total, kept = convert_csv_to_inter(
        input_path=args.input,
        output_path=args.output,
        keep_event=args.event,
        sort_by_time=args.sort_by_time,
    )
    print(f"Input rows: {total}")
    print(f"Rows kept (event={args.event}): {kept}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
