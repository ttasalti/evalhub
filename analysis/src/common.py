"""Shared paths for the analysis scripts.

Every script resolves its inputs and outputs through this module, so the
package runs from any working directory. The data layer under ``analysis/data``
is published with the repository; the raw experiment outputs under
``results/`` and the error-injection caches are not, and the scripts that need
them stop with a message when they are absent. ``EVALHUB_ANALYSIS_OUT``
redirects the figure and table output to another directory, which the tests use.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYSIS_ROOT.parent
DATA_DIR = ANALYSIS_ROOT / "data"
_OUT = Path(os.environ["EVALHUB_ANALYSIS_OUT"]) if os.environ.get("EVALHUB_ANALYSIS_OUT") else ANALYSIS_ROOT
FIG_DIR = _OUT / "figures"
TAB_DIR = _OUT / "tables"
RESULTS_ROOT = Path(os.environ.get("EVALHUB_RESULTS", REPO_ROOT / "results"))
ERROR_STUDY_OUTPUT = Path(os.environ.get("EVALHUB_ERROR_STUDY_OUTPUT", REPO_ROOT / "error_study" / "output"))
CACHE_DIR = Path(os.environ.get("EVALHUB_ANALYSIS_CACHE", ANALYSIS_ROOT / "cache"))
GLOTLID_MODEL = Path(os.environ.get("GLOTLID_MODEL", Path.home() / ".cache" / "glotlid" / "model_v3.bin"))

BENCHMARK_LANGUAGE = {
    "aime2026": "EN",
    "aime2026_pt": "PT",
    "aime2026_tr": "TR",
    "tubitak_math2026": "TR",
    "pt_exams_math": "PT",
}


def data(name: str) -> Path:
    """Path of a published data table, e.g. ``data("verdicts.csv")``."""
    return DATA_DIR / name


def ensure_output_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def require(*paths: Path, what: str = "input") -> None:
    """Exit with a clear message when an unpublished input is missing."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        sys.exit(f"This script needs {what} that is not part of the repository:\n  " + "\n  ".join(missing))
