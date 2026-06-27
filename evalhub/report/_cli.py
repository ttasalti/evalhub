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
DEFAULT_HIGHLIGHTS_PDF = Path("results/report_highlights.pdf")
DEFAULT_ATLAS_PDF = Path("results/report_plots_atlas.pdf")


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


def cmd_highlights(csv: Path, output: Path) -> Path:
    """Implementation of ``evalhub report highlights`` — render the highlights PDF."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "evalhub report highlights requires pandas + matplotlib. "
            "Install with `pip install evalhub[report]`."
        ) from e
    from evalhub.report.highlights import build_highlights

    if not csv.exists():
        raise FileNotFoundError(f"CSV not found: {csv} (run `evalhub report aggregate` first)")
    df = pd.read_csv(csv)
    out = build_highlights(df, output)
    logger.info(f"Wrote highlights PDF -> {out}")
    return out


def cmd_atlas(plot_dir: Path, output: Path) -> Path:
    """Implementation of ``evalhub report atlas`` — curated visual index of the plot suite."""
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "evalhub report atlas requires matplotlib. Install with `pip install evalhub[report]`."
        ) from e
    from evalhub.report.atlas import build_atlas

    if not plot_dir.exists():
        raise FileNotFoundError(f"Plot dir not found: {plot_dir} (run `evalhub report plot` first)")
    out = build_atlas(plot_dir, output)
    logger.info(f"Wrote plot atlas PDF -> {out}")
    return out
