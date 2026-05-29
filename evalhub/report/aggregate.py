"""Long-form DataFrame and master-CSV writer for :class:`RunRecord` lists.

A single :class:`RunRecord` carries a ``pass_at_k`` mapping with multiple K
values. To make the data convenient for pandas/Plotly aggregation we explode
each record into one row per ``(run, k)`` pair. ``cons_at_k`` and the per-run
stats columns are repeated on every row of a given run (acceptable for
plotting; treat-as-key documented in ``docs/reporting.md``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from evalhub.report.scan import RunRecord, scan_results
from evalhub.utils.logger import logger

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


# Canonical column order for the long-form DataFrame. The first chunk
# identifies the run; the K-axis columns come next; then per-run stats.
LONG_COLUMNS: list[str] = [
    "model",
    "state",
    "temperature",
    "max_tokens",
    "n_samples",
    "benchmark",
    "eval_type",
    "judge_model",
    "judge_state",
    "judge_temperature",
    "judge_max_tokens",
    "judge_n_samples",
    "k",
    "pass_at_k",
    "cons_at_k",
    "total_tasks",
    "total_generations",
    "true_count",
    "false_count",
    "cot_false_count",
    "invalid_count",
    "run_dir",
]

_STATS_KEYS: tuple[str, ...] = (
    "total_tasks",
    "total_generations",
    "true_count",
    "false_count",
    "cot_false_count",
    "invalid_count",
)


def _record_to_rows(record: RunRecord) -> list[dict]:
    base = {
        "model": record.model,
        "state": record.state,
        "temperature": record.temperature,
        "max_tokens": record.max_completion_tokens,
        "n_samples": record.n_samples,
        "benchmark": record.benchmark,
        "eval_type": record.eval_type,
        "judge_model": record.judge_model,
        "judge_state": record.judge_state,
        "judge_temperature": record.judge_temperature,
        "judge_max_tokens": record.judge_max_completion_tokens,
        "judge_n_samples": record.judge_n_samples,
        "cons_at_k": record.cons_at_k,
        "run_dir": str(record.run_dir),
    }
    for key in _STATS_KEYS:
        base[key] = (record.stats or {}).get(key)
    if not record.pass_at_k:
        # Preserve runs that produced no Pass@K (e.g. the "no base-correct
        # samples" stub) as a single row with NaN k so the CSV still
        # documents that the run happened.
        row = dict(base)
        row["k"] = None
        row["pass_at_k"] = None
        return [row]
    rows: list[dict] = []
    for k in sorted(record.pass_at_k.keys()):
        row = dict(base)
        row["k"] = k
        row["pass_at_k"] = record.pass_at_k[k]
        rows.append(row)
    return rows


def build_dataframe(records: Iterable[RunRecord]) -> pd.DataFrame:
    """Explode an iterable of :class:`RunRecord` into a long-form DataFrame."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - explicit, actionable error
        raise ImportError(
            "evalhub.report.aggregate requires pandas. Install with "
            "`pip install evalhub[report]` or `pip install pandas`."
        ) from e

    rows: list[dict] = []
    for record in records:
        rows.extend(_record_to_rows(record))
    df = pd.DataFrame(rows, columns=LONG_COLUMNS)
    return df


def write_csv(df: pd.DataFrame, output_csv: Path | str) -> Path:
    """Write a long-form DataFrame to CSV, creating parent dirs as needed."""
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df)} rows -> {out}")
    return out


def aggregate_results(
    results_root: Path | str,
    output_csv: Path | str,
) -> pd.DataFrame:
    """Scan ``results_root``, build the long-form DataFrame, and write CSV."""
    records = scan_results(results_root)
    df = build_dataframe(records)
    write_csv(df, output_csv)
    return df
