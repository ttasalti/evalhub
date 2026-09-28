"""Tests for the any/all judge-approval-threshold cot metric expansion.

Covers the survival rule, the tripled summary suite + monotonicity, and the
idempotent additive backfill.
"""

from __future__ import annotations

import csv

import orjson
import pytest

from evalhub.cot.ids import encode as gid
from evalhub.cot.metrics import _survives, apply_cot_metrics
from evalhub.report.backfill import backfill_leaf


def test_survives_thresholds():
    # (yes, no, invalid)
    assert _survives(1, 2, 0, "any")
    assert not _survives(0, 3, 0, "any")
    assert _survives(2, 1, 0, "majority")
    assert not _survives(1, 2, 0, "majority")
    assert not _survives(1, 1, 1, "majority")  # tie -> vetoed
    assert _survives(3, 0, 0, "all")
    assert not _survives(2, 1, 0, "all")  # a 'no' -> not unanimous
    assert not _survives(2, 0, 1, "all")  # an 'invalid' -> not unanimous


def _write_jsonl(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")


def _make_leaf(tmp_path):
    """One task, 3 answer-correct generations with a yes/no spread across thresholds."""
    base = tmp_path / "aime2026_results.jsonl"
    _write_jsonl(
        base,
        [
            {
                "task_id": "T/1",
                "solutions": ["5", "5", "5"],
                "ground_truth": "5",
                "correct": [True, True, True],
                "pass_at_k": {"1": 1.0, "2": 1.0, "4": 1.0},
            }
        ],
    )
    maj = tmp_path / "aime2026_cot_majority.jsonl"
    _write_jsonl(
        maj,
        [
            # gen0 unanimous yes -> survives all;  gen1 1y2n -> survives any only;
            # gen2 2y1n -> survives any + majority, not all.
            {"task_id": gid("T/1", 0), "yes_count": 3, "no_count": 0, "invalid_count": 0, "majority_correct": True},
            {"task_id": gid("T/1", 1), "yes_count": 1, "no_count": 2, "invalid_count": 0, "majority_correct": False},
            {"task_id": gid("T/1", 2), "yes_count": 2, "no_count": 1, "invalid_count": 0, "majority_correct": True},
        ],
    )
    return base, maj


def test_apply_cot_metrics_threshold_suite(tmp_path):
    base, maj = _make_leaf(tmp_path)
    s = apply_cot_metrics(base, maj, tmp_path / "aime2026_cot_results.jsonl", tmp_path / "aime2026_cot_summary.json")
    # majority survivors: gen0, gen2 -> true=2, cot_false=1
    assert s["true_count"] == 2 and s["cot_false_count"] == 1
    # any survivors: all three -> true=3, cot_false=0
    assert s["true_count_any"] == 3 and s["cot_false_count_any"] == 0
    # all survivors: gen0 only -> true=1, cot_false=2
    assert s["true_count_all"] == 1 and s["cot_false_count_all"] == 2
    # monotone pass@1 (= true/3): all <= majority <= any
    assert s["pass_at_k_all"]["1"] == pytest.approx(1 / 3)
    assert s["pass_at_k"]["1"] == pytest.approx(2 / 3)
    assert s["pass_at_k_any"]["1"] == pytest.approx(1.0)
    # the full suite is duplicated per threshold
    for key in ("g_pass_at_k", "mg_pass_at_k", "cons_at_k"):
        assert f"{key}_any" in s and f"{key}_all" in s


def test_apply_cot_metrics_old_schema_majority(tmp_path):
    """Old cot_majority.jsonl carries raw verdicts in `solutions` (no *_count fields);
    counts must be re-derived so any/all still work and stay monotone."""
    base = tmp_path / "aime2026_results.jsonl"
    _write_jsonl(
        base,
        [
            {
                "task_id": "T/1",
                "solutions": ["5", "5", "5"],
                "ground_truth": "5",
                "correct": [True, True, True],
                "pass_at_k": {"1": 1.0},
            }
        ],
    )
    maj = tmp_path / "aime2026_cot_majority.jsonl"
    _write_jsonl(
        maj,
        [
            {"task_id": gid("T/1", 0), "solutions": ["yes", "yes", "yes"], "majority_correct": True},
            # 1 yes / 2 non-yes (invalid_format + a stray number) -> survives any, not all
            {"task_id": gid("T/1", 1), "solutions": ["yes", "invalid_format", "277"], "majority_correct": True},
            {"task_id": gid("T/1", 2), "solutions": ["yes", "yes", "no"], "majority_correct": True},
        ],
    )
    s = apply_cot_metrics(base, maj, tmp_path / "aime2026_cot_results.jsonl", tmp_path / "aime2026_cot_summary.json")
    assert s["true_count_any"] == 3  # all three have >=1 yes
    assert s["true_count"] == 3  # stored majority_correct=True for all
    assert s["true_count_all"] == 1  # only gen0 is unanimous yes
    assert s["true_count_all"] <= s["true_count"] <= s["true_count_any"]


def test_backfill_is_additive_and_idempotent(tmp_path):
    base, maj = _make_leaf(tmp_path)
    # produce a full leaf, then strip the threshold blocks to simulate an OLD leaf
    apply_cot_metrics(base, maj, tmp_path / "aime2026_cot_results.jsonl", tmp_path / "aime2026_cot_summary.json")
    summ = tmp_path / "aime2026_cot_summary.json"
    full = orjson.loads(summ.read_bytes())
    stripped = {k: v for k, v in full.items() if not (k.endswith("_any") or k.endswith("_all"))}
    summ.write_bytes(orjson.dumps(stripped))
    pt = tmp_path / "aime2026_cot_per_task.csv"
    with open(pt, newline="") as f:
        rows = list(csv.DictReader(f))
        keep = [c for c in rows[0] if not (c.endswith("_any") or c.endswith("_all"))]
    with open(pt, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep)
        w.writeheader()
        w.writerows([{c: r[c] for c in keep} for r in rows])

    # backfill adds the threshold blocks back
    assert backfill_leaf(tmp_path, "aime2026") == "updated"
    got = orjson.loads(summ.read_bytes())
    assert got["true_count_any"] == 3 and got["true_count_all"] == 1
    # majority values are untouched by the backfill
    assert got["true_count"] == full["true_count"] and got["pass_at_k"] == full["pass_at_k"]
    hdr = open(pt).readline().strip().split(",")
    assert all(c in hdr for c in ("true_any", "cot_false_any", "true_all", "cot_false_all"))
    # second run is a no-op
    assert backfill_leaf(tmp_path, "aime2026") == "noop"
