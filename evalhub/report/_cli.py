"""Implementations for the ``evalhub report`` Typer commands.

Kept separate from :mod:`evalhub.cli` so the top-level CLI module stays small
and the report logic is independently testable.
"""

from __future__ import annotations

from pathlib import Path

from evalhub.report.aggregate import aggregate_results, upsert_summary
from evalhub.utils.logger import logger

# The master CSV lives inside the results tree by convention.
DEFAULT_CSV = Path("results/report.csv")
DEFAULT_PLOT_DIR = Path("results/report_plots")


def cmd_aggregate(results_root: Path, output: Path) -> Path:
    """Implementation of ``evalhub report aggregate`` — full wide-CSV rebuild."""
    aggregate_results(results_root, output)
    return output


def cmd_upsert(summary: Path, csv: Path, results_root: Path | None = None) -> Path:
    """Implementation of ``evalhub report upsert`` — add/replace one result row."""
    out = upsert_summary(summary, csv, results_root)
    logger.info(f"Upserted {summary} into {out}")
    return out


def cmd_plot(csv: Path, output_dir: Path) -> dict[str, list[Path]]:
    """Implementation of ``evalhub report plot`` — render the Pass@K vs CoT-Pass@K suite."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "evalhub report plot requires pandas + matplotlib + seaborn. "
            "Install with `pip install evalhub[report]`."
        ) from e
    from evalhub.report.plots import render_all

    if not csv.exists():
        raise FileNotFoundError(f"CSV not found: {csv} (run `evalhub report aggregate` first)")
    df = pd.read_csv(csv)
    written = render_all(df, output_dir)
    n = sum(len(v) for v in written.values())
    logger.info(f"Wrote {n} file(s) under {output_dir}")
    return written
