"""Implementations for the ``evalhub report`` Typer commands.

Kept separate from :mod:`evalhub.cli` so the top-level CLI module stays small
and the report logic is independently testable.
"""

from __future__ import annotations

from pathlib import Path

from evalhub.report.aggregate import aggregate_results, upsert_summary
from evalhub.report.backfill import backfill_all
from evalhub.utils.logger import logger

# The master CSV lives inside the results tree by convention.
DEFAULT_CSV = Path("results/report.csv")
DEFAULT_PLOT_DIR = Path("results/report_plots")


def cmd_aggregate(results_root: Path, output: Path) -> Path:
    """Implementation of ``evalhub report aggregate``, full wide-CSV rebuild."""
    aggregate_results(results_root, output)
    return output


def cmd_upsert(summary: Path, csv: Path, results_root: Path | None = None) -> Path | None:
    """Implementation of ``evalhub report upsert``, add/replace one result row."""
    out = upsert_summary(summary, csv, results_root)
    if out is None:
        logger.info(f"Skipped {summary} (excluded model)")
    else:
        logger.info(f"Upserted {summary} into {out}")
    return out


def cmd_backfill_anyall(results_root: Path, dry_run: bool) -> dict[str, int]:
    """Implementation of ``evalhub report backfill-anyall``, add the any/all
    judge-approval-threshold blocks to every existing judged leaf's cot summary +
    per-task CSV (additive, idempotent, no re-eval / no re-judge)."""
    tally = backfill_all(results_root, dry_run=dry_run)
    verb = "would update" if dry_run else "updated"
    logger.info(f"backfill-anyall ({'dry-run' if dry_run else 'write'}): {tally}")
    logger.info(f"{verb} {tally.get('updated', 0) + tally.get('would-update', 0)} judged leaf/leaves")
    return tally


def cmd_plot(csv: Path, output_dir: Path) -> dict[str, list[Path]]:
    """Implementation of ``evalhub report plot``, render the Pass@K vs CoT-Pass@K suite."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "evalhub report plot requires pandas + matplotlib + seaborn. Install with `pip install evalhub[report]`."
        ) from e
    from evalhub.report.plots import render_all

    if not csv.exists():
        raise FileNotFoundError(f"CSV not found: {csv} (run `evalhub report aggregate` first)")
    df = pd.read_csv(csv)
    written = render_all(df, output_dir)
    n = sum(len(v) for v in written.values())
    logger.info(f"Wrote {n} file(s) under {output_dir}")
    return written
