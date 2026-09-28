"""Backfill the any/all judge-approval-threshold metrics onto existing judged results.

For every judged leaf already on disk this adds the ``_any`` / ``_all`` threshold
blocks to ``<bench>_cot_summary.json`` and the ``true_any`` / ``cot_false_any`` /
``true_all`` / ``cot_false_all`` per-task columns to ``<bench>_cot_per_task.csv``,
WITHOUT re-running any model or judge, and WITHOUT touching the raw base
``_results.jsonl`` / ground truth. The majority (unsuffixed) values are left exactly as
they are. Additive + idempotent: re-running produces no net change.

The base "answer-correct" set is reconstructed from the leaf's own
``<bench>_cot_results.jsonl`` (a generation is answer-correct iff its post-veto label is
``True`` or ``"cot_false"``), so no base-leaf lookup is needed. As a safety net, the
recomputed majority metrics are checked against the leaf's existing summary; a leaf whose
majority would not reproduce is SKIPPED untouched.
"""

from __future__ import annotations

import csv as _csv
import glob
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import orjson

from evalhub.cot.metrics import COT_FALSE_LABEL, apply_cot_metrics
from evalhub.utils.logger import logger

_SUMMARY_SUFFIXES = ("_any", "_all")
_EXTRA_COLS = ["true_any", "cot_false_any", "true_all", "cot_false_all"]
# majority keys that reconstruct+recompute must reproduce for the leaf to be safe to touch
_MAJORITY_KEYS = (
    "pass_at_k",
    "g_pass_at_k",
    "mg_pass_at_k",
    "cons_at_k",
    "true_count",
    "false_count",
    "cot_false_count",
)


def _close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def _reconstruct_base(cot_results_path: Path, out_path: Path) -> None:
    """Write a base_results.jsonl reconstructed from a leaf's cot_results.jsonl."""
    with open(cot_results_path, "rb") as f_in, open(out_path, "wb") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            r = orjson.loads(line)
            base_correct = [(x is True or x == COT_FALSE_LABEL) for x in r.get("correct", [])]
            f_out.write(
                orjson.dumps(
                    {
                        "task_id": r["task_id"],
                        "solutions": r.get("solutions", []),
                        "ground_truth": r.get("ground_truth", ""),
                        "correct": base_correct,
                        "pass_at_k": r.get("pass_at_k", {}),
                    }
                )
                + b"\n"
            )


def backfill_leaf(leaf: Path, bench: str, dry_run: bool = False) -> str:
    """Backfill one judged leaf. Returns 'updated' | 'noop' | 'would-update' | 'skip:<reason>'."""
    maj = leaf / f"{bench}_cot_majority.jsonl"
    res = leaf / f"{bench}_cot_results.jsonl"
    summ = leaf / f"{bench}_cot_summary.json"
    per_task = leaf / f"{bench}_cot_per_task.csv"
    if not (maj.exists() and res.exists() and summ.exists()):
        return "skip:missing-files"

    cur = orjson.loads(summ.read_bytes())

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        base_path = tmp / f"{bench}_results.jsonl"
        _reconstruct_base(res, base_path)
        new_summary = apply_cot_metrics(
            base_results_path=base_path,
            majority_path=maj,
            output_results_path=tmp / f"{bench}_cot_results.jsonl",
            summary_path=tmp / f"{bench}_cot_summary.json",
        )
        # safety: majority must reproduce exactly, else leave the leaf untouched
        if not all(k in cur and _close(new_summary[k], cur[k]) for k in _MAJORITY_KEYS):
            return "skip:majority-mismatch"
        tmp_per_task = tmp / f"{bench}_cot_per_task.csv"
        thresh_by_task: dict[str, dict[str, str]] = {}
        if tmp_per_task.exists():
            with open(tmp_per_task, newline="", encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    thresh_by_task[row["task_id"]] = {c: row.get(c, "") for c in _EXTRA_COLS}

    # merge summary: add only the *_any / *_all keys
    add = {k: v for k, v in new_summary.items() if k.endswith(_SUMMARY_SUFFIXES)}
    merged = dict(cur)
    merged.update(add)
    summary_changed = merged != cur

    # merge per_task.csv: add/replace the 4 threshold columns
    per_task_changed = False
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    if per_task.exists():
        with open(per_task, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                vals = thresh_by_task.get(row["task_id"], dict.fromkeys(_EXTRA_COLS, ""))
                for c in _EXTRA_COLS:
                    nv = str(vals.get(c, ""))
                    if row.get(c) != nv:
                        per_task_changed = True
                    row[c] = nv
                rows.append(row)
        missing = [c for c in _EXTRA_COLS if c not in fieldnames]
        if missing:
            per_task_changed = True
            fieldnames = fieldnames + missing

    if not (summary_changed or per_task_changed):
        return "noop"
    if dry_run:
        return "would-update"

    if summary_changed:
        summ.write_bytes(orjson.dumps(merged))
    if per_task_changed and per_task.exists():
        with open(per_task, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    return "updated"


def backfill_all(results_root: Path, dry_run: bool = False) -> dict[str, int]:
    """Backfill every judged leaf under ``results_root``. Returns a status tally."""
    tally: Counter = Counter()
    logger.disable("evalhub.cot.metrics")  # apply_cot_metrics is chatty; run 299x quietly
    try:
        majs = sorted(glob.glob(str(results_root / "**" / "*_cot_majority.jsonl"), recursive=True))
        for m in majs:
            maj = Path(m)
            leaf = maj.parent
            bench = maj.name[: -len("_cot_majority.jsonl")]
            status = backfill_leaf(leaf, bench, dry_run=dry_run)
            tally[status.split(":")[0]] += 1
            if status.startswith("skip"):
                logger.warning(f"backfill skip {leaf}/{bench}: {status}")
    finally:
        logger.enable("evalhub.cot.metrics")
    return dict(tally)
