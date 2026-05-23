"""Shared logging helpers: Log/ (full process) vs LogMins/ (final summary)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from ncs_paths import log_file, log_mins_file


def write_run_summary(
    problem: str,
    dataset: str,
    lines: Iterable[str],
    *,
    date: datetime | None = None,
    append: bool = True,
) -> Path:
    """Write final-result lines to LogMins/<problem>/<dataset>/DD-MM-YYYY.log."""
    path = log_mins_file(problem, dataset, date)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(line.rstrip() for line in lines if line is not None)
    if not body:
        return path
    if append and path.exists() and path.stat().st_size > 0:
        body = f"\n{'=' * 60}\n{body}"
    with path.open("a" if append and path.exists() else "w", encoding="utf-8") as handle:
        handle.write(body)
        handle.write("\n")
    return path


def default_process_log(problem: str, dataset: str, date: datetime | None = None) -> Path:
    """Return default full-process log path under Log/."""
    path = log_file(problem, dataset, date)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def format_best_k_summary(
    dataset: str,
    best_results: dict,
    top_k: Sequence[int],
    *,
    header: str | None = None,
) -> list[str]:
    """Build LogMins lines from DHCN/COTREC-style best_results dict."""
    lines = [header or f"dataset={dataset}"]
    for k in top_k:
        lines.append(
            "best Recall@{k}={recall:.4f} epoch={e_r} | best MRR@{k}={mrr:.4f} epoch={e_m}".format(
                k=k,
                recall=best_results["metric%d" % k][0],
                mrr=best_results["metric%d" % k][1],
                e_r=best_results["epoch%d" % k][0],
                e_m=best_results["epoch%d" % k][1],
            )
        )
    return lines
