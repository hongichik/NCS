"""
Thống kê sơ bộ dữ liệu RetailRocket:
- Kiểm tra item nào không có danh mục
- Đếm item có click <= 5
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / 'Data' / 'thucnghiem2' / 'retailrocket'


def analyze_retailrocket(data_root: str | Path) -> None:
    data_root = Path(data_root)

    # ============================================================================
    # 1. Thống kê item không có danh mục
    # ============================================================================
    print("=" * 80)
    print("1. THỐNG KÊ ITEM KHÔNG CÓ DANH MỤC")
    print("=" * 80)

    items_with_category: set[int] = set()
    for property_file in ["item_properties_part1.csv", "item_properties_part2.csv"]:
        property_path = data_root / property_file
        if not property_path.exists():
            print(f"⚠️  File {property_file} không tồn tại, bỏ qua.")
            continue

        with property_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("property") == "categoryid":
                    item_id = int(row["itemid"])
                    items_with_category.add(item_id)

    print(f"Số item có danh mục: {len(items_with_category):,}")

    # ============================================================================
    # 2. Thống kê tần suất click item
    # ============================================================================
    print("\n" + "=" * 80)
    print("2. THỐNG KÊ TẦN SUẤT CLICK ITEM")
    print("=" * 80)

    events_path = data_root / "events.csv"
    if not events_path.exists():
        print(f"❌ File events.csv không tồn tại.")
        return

    item_click_counts: dict[int, int] = defaultdict(int)
    total_view_events = 0

    with events_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("event") == "view":
                item_id = int(row["itemid"])
                item_click_counts[item_id] += 1
                total_view_events += 1

    print(f"Tổng số view event: {total_view_events:,}")
    print(f"Số item được click (view): {len(item_click_counts):,}")

    # Thống kê theo ngưỡng click
    click_thresholds = [1, 2, 3, 5, 10, 20, 50, 100]
    print("\nPhân bố item theo số click:")
    print(f"{'Ngưỡng (clicks)':<20} {'Số item <= ngưỡng':<20} {'Phần trăm':<15}")
    print("-" * 55)

    for threshold in click_thresholds:
        count_le = sum(1 for click in item_click_counts.values() if click <= threshold)
        percent = 100.0 * count_le / len(item_click_counts) if item_click_counts else 0
        print(f"<= {threshold:<17} {count_le:<20,} {percent:>6.2f}%")

    # Thống kê chi tiết cho click <= 5
    items_rare = {
        item_id: click for item_id, click in item_click_counts.items() if click <= 5
    }
    print("\n" + "=" * 80)
    print("3. CHI TIẾT ITEM CÓ CLICK <= 5")
    print("=" * 80)
    print(f"Số item có click <= 5: {len(items_rare):,}")
    print(f"Phần trăm so với tổng item: {100.0 * len(items_rare) / len(item_click_counts):.2f}%")

    # Thống kê phân bố click <= 5
    clicks_distribution = defaultdict(int)
    for click in items_rare.values():
        clicks_distribution[click] += 1

    print("\nPhân bố trong nhóm click <= 5:")
    for clicks in sorted(clicks_distribution.keys()):
        count = clicks_distribution[clicks]
        print(f"  {clicks} click: {count:,} item")

    # ============================================================================
    # 4. Item không có danh mục nhưng có click
    # ============================================================================
    print("\n" + "=" * 80)
    print("4. ITEM KHÔNG CÓ DANH MỤC NHƯNG CÓ CLICK")
    print("=" * 80)

    items_without_category = set(item_click_counts.keys()) - items_with_category
    print(f"✓ Số item KHÔNG có danh mục nhưng được click: {len(items_without_category):,}")
    print(
        f"  Phần trăm so với item được click: {100.0 * len(items_without_category) / len(item_click_counts):.2f}%"
    )

    # Thống kê click của item không có danh mục
    if items_without_category:
        clicks_no_category = [
            item_click_counts[item_id] for item_id in items_without_category
        ]
        print(f"  - Min clicks (no category): {min(clicks_no_category)}")
        print(f"  - Max clicks (no category): {max(clicks_no_category)}")
        print(f"  - Avg clicks (no category): {sum(clicks_no_category) / len(clicks_no_category):.2f}")

    # ============================================================================
    # 5. Item không có danh mục ở tất cả các file properties
    # ============================================================================
    print("\n" + "=" * 80)
    print("5. ITEM KHÔNG CÓ DANH MỤC TỪ PROPERTIES")
    print("=" * 80)
    print(f"Tổng số item riêng biệt xuất hiện trong events (clicked): {len(item_click_counts):,}")
    print(f"Item có trong item_properties (categoryid): {len(items_with_category):,}")
    print(f"✓ Item KHÔNG TỒN TẠI trong item_properties: {len(items_without_category):,}")

    # ============================================================================
    # 6. Khuyến nghị lọc
    # ============================================================================
    print("\n" + "=" * 80)
    print("6. KHUYẾN NGHỊ (EFFECT SAU LỌC)")
    print("=" * 80)

    items_to_keep_min5 = {
        item_id for item_id, click in item_click_counts.items() if click > 5
    }
    items_with_cat_to_keep = items_to_keep_min5 & items_with_category

    print(f"Nếu áp dụng: click > 5 + có danh mục")
    print(f"  - Item giữ lại: {len(items_with_cat_to_keep):,}")
    print(f"  - Item bỏ đi: {len(item_click_counts) - len(items_with_cat_to_keep):,}")
    print(
        f"  - Phần trăm giữ lại: {100.0 * len(items_with_cat_to_keep) / len(item_click_counts):.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phân tích dữ liệu RetailRocket trước preprocess."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to Data/thucnghiem2/retailrocket")
    args = parser.parse_args()

    analyze_retailrocket(args.data_root)


if __name__ == "__main__":
    main()
