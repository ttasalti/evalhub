"""Implementations for the ``evalhub report`` Typer commands.

Kept separate from :mod:`evalhub.cli` so the top-level CLI module stays small
and the report logic is independently testable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from evalhub.report.aggregate import aggregate_results
from evalhub.report.plots import render_all
from evalhub.utils.logger import logger


def cmd_aggregate(results_root: Path, output: Path) -> Path:
    """Implementation of ``evalhub report aggregate``."""
    aggregate_results(results_root, output)
    return output


def cmd_plot(csv: Path, output_dir: Path, fmt: str) -> dict[str, list[Path]]:
    """Implementation of ``evalhub report plot``."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "evalhub report plot requires pandas. Install with `pip install evalhub[report]`."
        ) from e
    df = pd.read_csv(csv)
    formats: tuple[str, ...]
    fmt = fmt.lower().strip()
    if fmt == "both":
        formats = ("png", "pdf")
    else:
        formats = (fmt,)
    written = render_all(df, output_dir, formats=formats)
    total = sum(len(paths) for paths in written.values())
    logger.info(f"Wrote {total} plot file(s) under {output_dir}")
    return written


def cmd_dashboard(
    csv: Path,
    results_root: Path | None,
    port: int,
) -> int:
    """Implementation of ``evalhub report dashboard``.

    Spawns ``streamlit run -m evalhub.report.dashboard`` with the CSV path and
    optional results root exported via environment variables.
    """
    if shutil.which("streamlit") is None:
        raise RuntimeError(
            "streamlit executable not found on PATH. Install with `pip install evalhub[report]`."
        )
    if not csv.exists():
        raise FileNotFoundError(f"CSV not found: {csv}")

    env = {
        **os.environ,
        "EVALHUB_REPORT_CSV": str(csv.resolve()),
        "EVALHUB_REPORT_ROOT": str(results_root.resolve()) if results_root else "",
    }
    cmd = [
        "streamlit",
        "run",
        "-m",
        "evalhub.report.dashboard",
        "--server.port",
        str(port),
    ]
    logger.info(f"Launching: {' '.join(cmd)}")
    completed = subprocess.run(cmd, env=env, check=False)
    return completed.returncode


def main_streamlit_entry() -> None:  # pragma: no cover - executed by streamlit
    """Convenience entrypoint so ``python -m evalhub.report.dashboard`` works
    when ``streamlit run`` cannot resolve ``-m``."""
    from evalhub.report.dashboard import main

    sys.exit(main() or 0)
