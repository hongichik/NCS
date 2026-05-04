import argparse
import json
from collections import defaultdict
from pathlib import Path

def create_augmentation_dicts(input_path: Path, output_path: Path) -> None:
    print(f"Loading data from {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # Note: JSON keys are strings, so we convert them back to integers
    item2leaf = {int(k): int(v) for k, v in payload["item2leaf_dict"].items()}
    leaf2parent = {int(k): int(v) for k, v in payload["leaf2parent_dict"].items()}

    # dict_same_leaf: item_id -> list of item_ids
    leaf2items = defaultdict(list)
    for item, leaf in item2leaf.items():
        leaf2items[leaf].append(item)

    import random

    dict_same_leaf = {}
    for leaf, items in leaf2items.items():
        if len(items) > 1:
            for item in items:
                same = [i for i in items if i != item]
                if len(same) > 20: same = random.sample(same, 20)
                dict_same_leaf[item] = same

    # dict_sibling: item_id -> list of item_ids
    parent2leafs = defaultdict(list)
    for leaf, parent in leaf2parent.items():
        parent2leafs[parent].append(leaf)

    dict_sibling = {}
    for parent, leafs in parent2leafs.items():
        if len(leafs) > 1:
            # build the union of all items from sibling leafs
            for leaf in leafs:
                sibling_items = []
                for other_leaf in leafs:
                    if other_leaf != leaf:
                        sibling_items.extend(leaf2items[other_leaf])
                if sibling_items:
                    if len(sibling_items) > 20:
                        sibling_items = random.sample(sibling_items, 20)
                    for item in leaf2items[leaf]:
                        dict_sibling[item] = sibling_items

    print("Saving augmentation dictionaries...")
    out_payload = {
        "dict_same_leaf": dict_same_leaf,
        "dict_sibling": dict_sibling
    }
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False)
        
    print(f"Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/processed/retailrocket_module1.json")
    parser.add_argument("--output", type=str, default="outputs/processed/retailrocket_augmentation.json")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    create_augmentation_dicts(input_path, output_path)

if __name__ == "__main__":
    main()
