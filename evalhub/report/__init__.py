"""Result aggregation for EvalHub.

Walks an ``OUTPUT_ROOT`` produced by ``evalhub eval`` and ``evalhub cot
finalize`` and normalises every summary file into one **wide** row per
evaluated ``(model, mode, benchmark, judge)`` — every K and τ in a single row.
The ``judge_model`` column is empty for the No-Judge reference and set for the
cot (judged) variant.

Two CLI surfaces:

* ``evalhub report aggregate`` — full rebuild of the master CSV.
* ``evalhub report upsert``    — append-or-replace a single result row.

The package is read-only over the artefacts the rest of the framework produces.
"""

from evalhub.report.aggregate import (
    META_COLUMNS,
    aggregate_results,
    build_wide_dataframe,
    upsert_record,
    upsert_summary,
    wide_row_from_record,
    write_csv,
)
from evalhub.report.scan import RunRecord, record_from_summary, scan_results

__all__ = [
    "META_COLUMNS",
    "RunRecord",
    "aggregate_results",
    "build_wide_dataframe",
    "record_from_summary",
    "scan_results",
    "upsert_record",
    "upsert_summary",
    "wide_row_from_record",
    "write_csv",
]
