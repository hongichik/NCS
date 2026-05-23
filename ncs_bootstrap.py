"""Bootstrap NCS repo root for code running inside nested project folders."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root(caller: str | Path) -> Path:
    start = Path(caller).resolve()
    base = start.parent if start.is_file() else start
    for candidate in [base, *base.parents]:
        if (candidate / "ncs_paths.py").is_file():
            root = str(candidate)
            if root not in sys.path:
                sys.path.insert(0, root)
            return candidate
    raise FileNotFoundError(f"NCS repo root not found from {start}")
