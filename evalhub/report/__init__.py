"""Result aggregation, plotting, and interactive dashboard for EvalHub.

This subpackage walks an ``OUTPUT_ROOT`` directory produced by ``evalhub eval``
and ``evalhub cot finalize``, normalises every summary file into a flat row
schema, and exposes three CLI surfaces:

* ``evalhub report aggregate``  — produce a master long-form CSV.
* ``evalhub report plot``       — render publication-ready PNG/PDF figures.
* ``evalhub report dashboard``  — launch a Streamlit + Plotly UI.

The package is intentionally read-only: it consumes the artefacts that the
rest of the framework produces, never modifies them, and adds no runtime
dependency on a running model server.
"""

from evalhub.report.aggregate import (
    LONG_COLUMNS,
    aggregate_results,
    build_dataframe,
    write_csv,
)
from evalhub.report.scan import RunRecord, scan_results

__all__ = [
    "LONG_COLUMNS",
    "RunRecord",
    "aggregate_results",
    "build_dataframe",
    "scan_results",
    "write_csv",
]
