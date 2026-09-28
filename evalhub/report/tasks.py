"""Problem-level CSV writer: one row per (run, task_id), sitting next to report.csv.

Each :class:`~evalhub.report.scan.RunRecord` that becomes one ``report.csv`` row
explodes into N ``report_tasks.csv`` rows (N = that benchmark's task count).
Source data already exists on disk per run (``<leaf>/<benchmark>_per_task.csv``
/ ``_cot_per_task.csv``, written by
:func:`evalhub.benchmarks.math.base.write_per_task_csv`). This module never
re-derives correctness from raw generations; it only reshapes what is already
there, plus a per-task version of ``aggregate.cot_false_invalid_stats``'
trunc-induced-veto split (which today only exists summed over a whole run).

Two entry points mirror ``aggregate.py``'s:

* :func:`aggregate_task_rows`: full rebuild from an already-scanned record list.
* :func:`upsert_task_rows`: append-or-replace one run's task rows, keyed like
  :func:`evalhub.report.aggregate.row_key` (model, state, benchmark,
  max_tokens, judge_model, judge_state, judge_max_tokens), so a pipeline can
  call it once per finished evaluation and the CSV grows run by run, exactly
  like ``upsert_record``.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from evalhub.report import labels
from evalhub.report.aggregate import _json_loads, _series_key, _verdict_class, row_key, write_csv
from evalhub.report.scan import RunRecord
from evalhub.utils.logger import logger

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


TASK_CSV_NAME = "report_tasks.csv"

# The first 14 columns are the exact set requested for this file. judge_state
# and judge_max_tokens (15th/16th) are internal disambiguators, not part of
# that list: upsert_task_rows must tell apart two judge configs that share
# judge_model but differ in judge_state (e.g. "think" vs "non-think") or in
# their own max_tokens when deciding which on-disk rows belong to the run
# being replaced, without them, upserting one config would silently delete
# another's task rows (this is also why max_tokens -- already one of the
# original 14 -- is part of the key: a 16k and a 32k re-run of the same
# model/state/benchmark/judge are different runs, not the same run twice).
TASK_COLUMNS: tuple[str, ...] = (
    "model",
    "model_short",
    "state",
    "benchmark",
    "language",
    "max_tokens",
    "judge_model",
    "task_id",
    "n_generations",
    "true_count",
    "false_count",
    "cot_false_count",
    "cot_false_trunc_induced",
    "cot_false_complete_verdict",
    # any/all judge-approval-threshold per-task counts (judged rows only; blank on
    # No-Judge rows). majority survivors/vetoes are the unsuffixed true_count/cot_false_count.
    "true_count_any",
    "cot_false_count_any",
    "true_count_all",
    "cot_false_count_all",
    "judge_state",
    "judge_max_tokens",
)

# The exact note evalhub's pipeline_write_empty_cot_summary (scripts/lib/
# pipeline_common.sh) writes when a benchmark's base run produced zero
# base-correct generations -- so a judge was never invoked at all for it.
NO_BASE_CORRECT_NOTE = "no base-correct samples"

# Legacy task_id prefixes found via a real-data audit of results/: at some
# point evalhub.benchmarks.math.aime2026_pt/aime2026_tr's PREFIX constant was
# renamed from "AIME2026-PT"/"AIME2026-TR" to today's "aime2026_pt"/
# "aime2026_tr", but the *_per_task.csv files from runs generated before the
# rename still carry the old prefix (task_id is baked in at generation time,
# never retroactively updated). The numeric suffix (problem_idx) is identical
# either way, so this is a pure string normalization, not a data change --
# without it the same problem shows up under two different task_id values
# depending on which code version produced the run, breaking the "same
# problem -> same id in every model/mode" guarantee this file exists for.
_LEGACY_TASK_ID_PREFIXES: dict[str, str] = {
    "AIME2026-PT": "aime2026_pt",
    "AIME2026-TR": "aime2026_tr",
}


def _normalize_task_id(task_id: str) -> str:
    prefix, sep, rest = task_id.partition("/")
    canonical = _LEGACY_TASK_ID_PREFIXES.get(prefix)
    return f"{canonical}{sep}{rest}" if canonical else task_id


_TRAILING_INT_RE = re.compile(r"(\d+)$")


def natural_task_id_key(task_id: str) -> tuple:
    """Sort key: numeric-aware on a trailing integer, else plain string.

    ``'AIME2026/2' < 'AIME2026/7' < 'AIME2026/10'``, not lexicographic, where
    ``'10'`` would sort before ``'7'``.
    """
    tid = task_id or ""
    m = _TRAILING_INT_RE.search(tid)
    if m:
        return (0, tid[: m.start()], int(m.group(1)))
    return (1, tid, 0)


def cot_false_stats_per_task(summary_path: Path | str) -> dict[str, dict[str, int]]:
    """Per-task variant of ``aggregate.cot_false_invalid_stats``'s bucket/flip loop.

    Returns ``{original_task_id: {"cot_false_complete_verdict": n,
    "cot_false_trunc_induced": n}}`` for every task with >=1 vetoed generation
    (original task id recovered the same way ``cot_false_invalid_stats`` does:
    splitting the generation id on its trailing ``"_gen_<idx>"``). Tasks with
    zero vetoed generations are simply absent from the returned dict, callers
    default them to 0, a legitimate "judge rejected nothing from this
    problem". Empty dict when the leaf has no ``*_cot_majority.jsonl`` at all
    (a No-Judge leaf).
    """
    leaf = Path(summary_path).parent
    majs = sorted(leaf.glob("*_cot_majority.jsonl"))
    if not majs:
        return {}

    per_task: dict[str, dict[str, int]] = defaultdict(
        lambda: {"cot_false_complete_verdict": 0, "cot_false_trunc_induced": 0}
    )
    for maj in majs:
        with open(maj, "rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = _json_loads(line)
                except ValueError:
                    continue
                if r.get("majority_correct"):
                    continue
                if r.get("invalid_count") is not None:
                    yc = int(r.get("yes_count", 0))
                    nc = int(r.get("no_count", 0))
                    ic = int(r.get("invalid_count", 0))
                else:
                    classes = [_verdict_class(str(s)) for s in (r.get("solutions") or [])]
                    yc = classes.count("yes")
                    nc = classes.count("no")
                    ic = classes.count("invalid")
                orig = _normalize_task_id(str(r.get("task_id", "")).rsplit("_gen_", 1)[0])
                if ic == 0:
                    per_task[orig]["cot_false_complete_verdict"] += 1
                if yc + ic > nc:
                    per_task[orig]["cot_false_trunc_induced"] += 1
    return dict(per_task)


def _identity_head(record: RunRecord) -> dict:
    """Identity columns preceding task_id, in TASK_COLUMNS order."""
    return {
        "model": record.model,
        "model_short": labels.short_model(record.model),
        "state": record.state,
        "benchmark": record.benchmark,
        "language": labels.language(record.benchmark),
        "max_tokens": record.max_completion_tokens,
        "judge_model": record.judge_model,
    }


def _identity_tail(record: RunRecord) -> dict:
    """Identity columns trailing the counters, in TASK_COLUMNS order."""
    return {
        "judge_state": record.judge_state,
        "judge_max_tokens": record.judge_max_completion_tokens,
    }


def _zero_fill_judged_rows(record: RunRecord) -> list[dict] | None:
    """Fill legitimate all-zero judge rows for a benchmark the judge never saw.

    When ``pipeline_write_empty_cot_summary`` fires (the base run produced
    zero base-correct generations for every task, so the judge was never
    invoked at all), there is no ``*_cot_per_task.csv`` to read -- but every
    task_id in the benchmark legitimately has 0 approved / 0 rejected / 0
    trunc-induced / 0 complete-verdict, not "unknown". This reads the task_id
    list from the sibling No-Judge run's ``*_per_task.csv`` (same
    model/state/benchmark, three directories up from the judge leaf, past
    ``<judge_leaf>/judged_by/``) and emits one legitimate-zero row per task.

    Returns ``None`` (caller falls back to skip+warn) unless the run's own
    ``*_cot_summary.json`` carries the exact ``NO_BASE_CORRECT_NOTE`` marker
    *and* the base per-task file can be found -- so a genuinely incomplete or
    broken run is never silently reported as all-zero.
    """
    try:
        summary = _json_loads(Path(record.summary_path).read_bytes())
    except (OSError, ValueError):
        return None
    if summary.get("note") != NO_BASE_CORRECT_NOTE:
        return None

    leaf = Path(record.summary_path).parent
    base_leaf = leaf.parent.parent.parent / leaf.name
    base_per_task_path = base_leaf / f"{record.benchmark}_per_task.csv"
    if not base_per_task_path.exists():
        return None

    head, tail = _identity_head(record), _identity_tail(record)
    zero_counters = {
        "n_generations": None,
        "true_count": 0,
        "false_count": None,
        "cot_false_count": 0,
        "cot_false_trunc_induced": 0,
        "cot_false_complete_verdict": 0,
    }
    rows: list[dict] = []
    with open(base_per_task_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            task_id = _normalize_task_id(r["task_id"])
            rows.append({**head, "task_id": task_id, **zero_counters, **tail})
    rows.sort(key=lambda r: natural_task_id_key(r["task_id"]))
    return rows


def _opt_int(value: object) -> int | None:
    """int() of a CSV cell, or None when the column is absent/blank (old files)."""
    if value in (None, ""):
        return None
    return int(value)


def task_rows_from_record(record: RunRecord) -> list[dict]:
    """Explode one ``RunRecord`` into its per-task ``report_tasks.csv`` rows.

    Reads the sibling ``<benchmark>_per_task.csv`` (No-Judge) or
    ``<benchmark>_cot_per_task.csv`` (judged) already written next to the
    run's summary, this never recomputes correctness from raw generations.
    If that file doesn't exist on a judged leaf because the judge was never
    invoked (zero base-correct generations), see :func:`_zero_fill_judged_rows`
    for the legitimate-all-zero case. Otherwise returns ``[]`` (logging a
    warning), e.g. a run predating this feature or a genuinely incomplete one.
    """
    jm = record.judge_model
    leaf = Path(record.summary_path).parent
    fname = f"{record.benchmark}_cot_per_task.csv" if jm else f"{record.benchmark}_per_task.csv"
    per_task_path = leaf / fname
    if not per_task_path.exists():
        if jm:
            zero_rows = _zero_fill_judged_rows(record)
            if zero_rows is not None:
                logger.info(
                    f"report_tasks: {record.model}/{record.state}/{record.benchmark} "
                    f"(judge={jm}) had 0 base-correct generations -- filled "
                    f"{len(zero_rows)} task row(s) with legitimate zero judge counts"
                )
                return zero_rows
        logger.warning(
            f"report_tasks: missing {fname} for "
            f"{record.model}/{record.state}/{record.benchmark} (judge={jm or 'none'}) "
            "-- skipping task rows for this run"
        )
        return []

    breakdown = cot_false_stats_per_task(record.summary_path) if jm else {}
    head, tail = _identity_head(record), _identity_tail(record)

    rows: list[dict] = []
    with open(per_task_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            task_id = _normalize_task_id(r["task_id"])
            true_ = int(r["true"])
            false_ = int(r["false"])
            cot_false_ = int(r["cot_false"])
            if jm:
                tb = breakdown.get(task_id, {})
                counters = {
                    "n_generations": None,
                    "true_count": true_,
                    "false_count": None,
                    "cot_false_count": cot_false_,
                    "cot_false_trunc_induced": tb.get("cot_false_trunc_induced", 0),
                    "cot_false_complete_verdict": tb.get("cot_false_complete_verdict", 0),
                    # any/all threshold per-task counts (present after the threshold
                    # suite / backfill; None on older cot_per_task.csv files).
                    "true_count_any": _opt_int(r.get("true_any")),
                    "cot_false_count_any": _opt_int(r.get("cot_false_any")),
                    "true_count_all": _opt_int(r.get("true_all")),
                    "cot_false_count_all": _opt_int(r.get("cot_false_all")),
                }
            else:
                counters = {
                    "n_generations": true_ + false_,
                    "true_count": true_,
                    "false_count": false_,
                    "cot_false_count": None,
                    "cot_false_trunc_induced": None,
                    "cot_false_complete_verdict": None,
                    "true_count_any": None,
                    "cot_false_count_any": None,
                    "true_count_all": None,
                    "cot_false_count_all": None,
                }
            # Built in TASK_COLUMNS order so callers reading these dicts
            # directly (not just via pd.DataFrame(columns=...)) see the
            # documented column order too.
            rows.append({**head, "task_id": task_id, **counters, **tail})

    rows.sort(key=lambda r: natural_task_id_key(r["task_id"]))
    return rows


def build_task_dataframe(records: Iterable[RunRecord]) -> pd.DataFrame:
    import pandas as pd

    rows: list[dict] = []
    for r in records:
        rows.extend(task_rows_from_record(r))
    return pd.DataFrame(rows, columns=list(TASK_COLUMNS))


def write_task_csv(df: pd.DataFrame, output_csv: Path | str) -> Path:
    return write_csv(df, output_csv)


def aggregate_task_rows(records: Iterable[RunRecord], output_csv: Path | str) -> pd.DataFrame:
    """Full rebuild: explode every (already excluded/capped) record into its
    task rows, in the same order the caller's ``report.csv`` rows are in.
    """
    df = build_task_dataframe(records)
    write_task_csv(df, output_csv)
    return df


def upsert_task_rows(record: RunRecord, csv_path: Path | str) -> Path | None:
    """Append-or-replace one run's task rows, keyed like ``aggregate.row_key``.

    Mirrors ``aggregate.upsert_record``'s filter-then-concat pattern, fanned
    out to N rows instead of one. Returns ``None`` (no-op) if the run has no
    per-task source file to read, an existing CSV is left untouched.
    """
    import pandas as pd

    new_rows = task_rows_from_record(record)
    if not new_rows:
        return None

    csv_path = Path(csv_path)
    new_key = row_key(record)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if len(df):
            keep = [i for i, r in df.iterrows() if _series_key(r) != new_key]
            df = df.loc[keep]
    else:
        df = pd.DataFrame(columns=list(TASK_COLUMNS))

    new_df = pd.DataFrame(new_rows, columns=list(TASK_COLUMNS))
    df = pd.concat([df, new_df], ignore_index=True)[list(TASK_COLUMNS)]
    write_task_csv(df, csv_path)
    return csv_path
