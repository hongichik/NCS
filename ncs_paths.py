"""Central path helpers for NCS_serve workspace conventions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Code location of each project (Data/Log/LogMins still use <problem> name at repo root).
PROJECT_DIRS: dict[str, str] = {
    "SR-GNN": "nhom1/SR-GNN",
    "GCE-GNN": "nhom1/GCE-GNN",
    "DHCN": "nhom2/DHCN",
    "COTREC": "nhom2/COTREC",
    "CSGNN": "nhom2/CSGNN",
    "DuoRec": "nhom3/DuoRec",
    "SelfContrastiveLearningRecSys": "nhom3/SelfContrastiveLearningRecSys",
    "thucnghiem": "MyProject/thucnghiem",
    "thucnghiem2": "MyProject/thucnghiem2",
}


def project_code_dir(problem: str) -> Path:
    rel = PROJECT_DIRS.get(problem, problem)
    return REPO_ROOT / rel


def data_dir(problem: str, dataset: str) -> Path:
    return REPO_ROOT / "Data" / problem / dataset.lower()


def data_root(problem: str) -> Path:
    return REPO_ROOT / "Data" / problem


def log_dir(problem: str, dataset: str) -> Path:
    """Full process log directory: every line of cmd output."""
    return REPO_ROOT / "Log" / problem / dataset.lower()


def log_file(problem: str, dataset: str, date: datetime | None = None) -> Path:
    date = date or datetime.now()
    return log_dir(problem, dataset) / f"{date:%d-%m-%Y}.log"


def log_mins_dir(problem: str, dataset: str) -> Path:
    """Summary log directory: final results only."""
    return REPO_ROOT / "LogMins" / problem / dataset.lower()


def log_mins_file(problem: str, dataset: str, date: datetime | None = None) -> Path:
    date = date or datetime.now()
    return log_mins_dir(problem, dataset) / f"{date:%d-%m-%Y}.log"


def test_source_dir() -> Path:
    """Raw shared test bundle: testall.csv + category_tree + item_properties."""
    return REPO_ROOT / "Data" / "Test"


def test_events_file() -> Path:
    return test_source_dir() / "testall.csv"


def test_project_data(problem: str) -> Path:
    """Converted test data for one project: Data/Test/<problem>/."""
    return test_source_dir() / problem
