"""Export CatSA RetailRocket artifact to Data/COTREC2/ for baseline runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cotrec_format import export_from_catsa_artifact
from ncs_paths import data_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CatSA retailrocket.json to COTREC pickle format."
    )
    parser.add_argument("--dataset", default="retailrocket")
    parser.add_argument(
        "--catsa-data-dir",
        type=Path,
        default=None,
        help="Default: Data/CatSA/<dataset>/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: Data/COTREC2/<dataset>/",
    )
    parser.add_argument(
        "--split-mode",
        choices=["catsa", "full"],
        default="catsa",
        help="catsa=train split only; full=train+valid for COTREC training",
    )
    parser.add_argument(
        "--max-prefix-len",
        type=int,
        default=0,
        help="Cap prefix length to COTREC pos_embedding (default: 300 retailrocket, 200 else)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = args.dataset.lower()
    catsa_dir = args.catsa_data_dir or data_dir("CatSA", ds)
    out_dir = args.output_dir or data_dir("COTREC2", ds)
    artifact = catsa_dir / "retailrocket.json"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing CatSA artifact: {artifact}")

    meta = export_from_catsa_artifact(
        artifact,
        out_dir,
        split_mode=args.split_mode,
        max_prefix_len=args.max_prefix_len,
        dataset=ds,
    )
    print(f"Exported to {out_dir}")
    for key, value in meta.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
