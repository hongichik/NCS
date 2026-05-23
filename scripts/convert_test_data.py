#!/usr/bin/env python3
"""Convert Data/Test raw bundle into per-project test datasets.

Output: Data/Test/<Project>/  (flat — no retailrocket subfolder)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ncs_paths import project_code_dir, test_project_data, test_source_dir  # noqa: E402

SESSION_PROJECTS = ("SR-GNN", "DHCN", "COTREC", "GCE-GNN")
ALL_PROJECTS = (*SESSION_PROJECTS, "DuoRec", "CSGNN", "thucnghiem", "thucnghiem2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Data/Test raw CSVs into project test data.")
    parser.add_argument("--source-root", type=Path, default=test_source_dir())
    parser.add_argument("--projects", nargs="+", choices=ALL_PROJECTS, default=list(ALL_PROJECTS))
    parser.add_argument("--min-item-freq", type=int, default=2)
    return parser.parse_args()


def _run(cmd: list[str], label: str) -> None:
    print(f"\n==> {label}\n    {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def convert_session_projects(source_root: Path, projects: list[str]) -> None:
    preprocess = project_code_dir("SR-GNN") / "datasets" / "preprocess.py"
    for project in projects:
        if project not in SESSION_PROJECTS:
            continue
        out_dir = test_project_data(project)
        out_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                str(preprocess),
                "--dataset",
                "retailrocket",
                "--data-path",
                str(source_root),
                "--output-path",
                str(out_dir),
            ],
            f"{project} -> train.txt / test.txt / all_train_seq.txt",
        )


def convert_duorec(source_root: Path) -> None:
    out_dir = test_project_data("DuoRec")
    out_dir.mkdir(parents=True, exist_ok=True)
    events = source_root / "testall.csv"
    if not events.exists():
        events = source_root / "events.csv"
    _run(
        [
            sys.executable,
            str(project_code_dir("DuoRec") / "recbole" / "data" / "csv_to_inter.py"),
            "--input",
            str(events),
            "--output",
            str(out_dir / "retailrocket.inter"),
            "--sort-by-time",
        ],
        "DuoRec -> retailrocket.inter",
    )


def convert_csgnn(source_root: Path, min_item_freq: int) -> None:
    out_dir = test_project_data("CSGNN") / "id"
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(project_code_dir("CSGNN") / "datasets" / "retailrocket_interest.py"),
            "--data-dir",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--min-item-freq",
            str(min_item_freq),
        ],
        "CSGNN -> id/*.txt",
    )


def convert_thucnghiem(project: str, source_root: Path) -> None:
    out_dir = test_project_data(project)
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(project_code_dir(project) / "preprocessing" / "retailrocket_preprocess.py"),
            "--data-root",
            str(source_root),
            "--output-path",
            str(out_dir / "retailrocket_module1.json"),
        ],
        f"{project} -> retailrocket_module1.json",
    )


def main() -> None:
    args = parse_args()
    source = args.source_root
    if not (source / "testall.csv").exists() and not (source / "events.csv").exists():
        raise FileNotFoundError(f"No events in {source}. Run scripts/build_test_source.py first.")

    projects = list(dict.fromkeys(args.projects))
    session = [p for p in projects if p in SESSION_PROJECTS]
    if session:
        convert_session_projects(source, session)
    if "DuoRec" in projects:
        convert_duorec(source)
    if "CSGNN" in projects:
        convert_csgnn(source, args.min_item_freq)
    for project in ("thucnghiem", "thucnghiem2"):
        if project in projects:
            convert_thucnghiem(project, source)
    print("\nAll conversions finished.")


if __name__ == "__main__":
    main()
