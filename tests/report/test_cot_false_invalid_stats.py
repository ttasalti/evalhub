"""Tests for the cot_false "why no?" breakdown in :mod:`evalhub.report.aggregate`.

``cot_false_invalid_stats`` explains every vetoed (cot_false) generation by how
many of the judge's samples were invalid (0..3), whether invalidity *flipped* the
veto, how many distinct questions were touched, and, at the verdict level,
which invalid samples were truncated. The fixture below encodes one generation of
each bucket so the expected counts are hand-checkable.
"""

from __future__ import annotations

import json
from pathlib import Path

from evalhub.report.aggregate import (
    COT_FALSE_COLUMNS,
    cot_false_invalid_stats,
)

BENCH = "aime2026"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _raw(task_id: str, finish_reason: str) -> dict:
    return {"task_id": task_id, "response": {"choices": [{"finish_reason": finish_reason}]}}


def _build_leaf(
    tmp_path: Path, *, total_tasks: int = 4, with_verdicts: bool = True, misalign: bool = False, drop_raw: bool = False
) -> Path:
    """A judge leaf with seven base-correct generations covering every bucket.

    cot_false (majority_correct False): g1,g2 (inv0), g3 (inv1), g4 (inv2), g5 (inv3).
    Approved: g0, g6. Tasks 1,2,3 are vetoed (task 4 is not).
    """
    leaf = tmp_path / "leaf"
    leaf.mkdir(parents=True, exist_ok=True)

    # Per-generation majority records (task_id == generation id "<task>_gen_<i>").
    majority = [
        {
            "task_id": f"{BENCH}/1_gen_0",
            "yes_count": 3,
            "no_count": 0,
            "invalid_count": 0,
            "majority_correct": True,
        },  # g0 approved
        {
            "task_id": f"{BENCH}/1_gen_1",
            "yes_count": 0,
            "no_count": 3,
            "invalid_count": 0,
            "majority_correct": False,
        },  # g1 inv0, no-flip
        {
            "task_id": f"{BENCH}/2_gen_0",
            "yes_count": 1,
            "no_count": 2,
            "invalid_count": 0,
            "majority_correct": False,
        },  # g2 inv0, no-flip
        {
            "task_id": f"{BENCH}/2_gen_1",
            "yes_count": 1,
            "no_count": 1,
            "invalid_count": 1,
            "majority_correct": False,
        },  # g3 inv1, FLIP (1+1>1)
        {
            "task_id": f"{BENCH}/3_gen_0",
            "yes_count": 0,
            "no_count": 1,
            "invalid_count": 2,
            "majority_correct": False,
        },  # g4 inv2, FLIP (0+2>1)
        {
            "task_id": f"{BENCH}/3_gen_1",
            "yes_count": 0,
            "no_count": 0,
            "invalid_count": 3,
            "majority_correct": False,
        },  # g5 inv3, FLIP (0+3>0)
        {
            "task_id": f"{BENCH}/4_gen_0",
            "yes_count": 2,
            "no_count": 1,
            "invalid_count": 0,
            "majority_correct": True,
        },  # g6 approved
    ]
    _write_jsonl(leaf / f"{BENCH}_cot_majority.jsonl", majority)

    (leaf / f"{BENCH}_cot_summary.json").write_text(json.dumps({"total_tasks": total_tasks, "cot_false_count": 5}))

    if with_verdicts:
        # Verdict-level files (one line per judge sample), line-aligned sol<->raw.
        # Invalid samples inside cot_false gens: g3=1 (length), g4=1 length+1 stop,
        # g5=3 length  ->  trunc=5, other=1.
        sol = [
            {"task_id": f"{BENCH}/1_gen_0", "solution": "yes"},  # approved -> skipped
            {"task_id": f"{BENCH}/1_gen_1", "solution": "no"},  # cot_false but valid -> skipped
            {"task_id": f"{BENCH}/2_gen_1", "solution": "yes"},  # g3 valid -> skipped
            {"task_id": f"{BENCH}/2_gen_1", "solution": "no"},  # g3 valid -> skipped
            {"task_id": f"{BENCH}/2_gen_1", "solution": "invalid_format"},  # g3 invalid (length)
            {"task_id": f"{BENCH}/3_gen_0", "solution": "no"},  # g4 valid -> skipped
            {"task_id": f"{BENCH}/3_gen_0", "solution": ""},  # g4 invalid (length)
            {"task_id": f"{BENCH}/3_gen_0", "solution": "maybe"},  # g4 invalid (stop -> other)
            {"task_id": f"{BENCH}/3_gen_1", "solution": "invalid_format"},  # g5 invalid (length)
            {"task_id": f"{BENCH}/3_gen_1", "solution": "invalid_format"},  # g5 invalid (length)
            {"task_id": f"{BENCH}/3_gen_1", "solution": "invalid_format"},  # g5 invalid (length)
        ]
        raw = [
            _raw(f"{BENCH}/1_gen_0", "stop"),
            _raw(f"{BENCH}/1_gen_1", "stop"),
            _raw(f"{BENCH}/2_gen_1", "stop"),
            _raw(f"{BENCH}/2_gen_1", "stop"),
            _raw(f"{BENCH}/2_gen_1", "length"),
            _raw(f"{BENCH}/3_gen_0", "stop"),
            _raw(f"{BENCH}/3_gen_0", "length"),
            _raw(f"{BENCH}/3_gen_0", "stop"),
            _raw(f"{BENCH}/3_gen_1", "length"),
            _raw(f"{BENCH}/3_gen_1", "length"),
            _raw(f"{BENCH}/3_gen_1", "length"),
        ]
        if misalign:
            raw[0] = _raw(f"{BENCH}/9_gen_9", "stop")  # break task_id alignment
        _write_jsonl(leaf / "cot_judge.jsonl", sol)
        if not drop_raw:
            _write_jsonl(leaf / "cot_judge_raw.jsonl", raw)
    return leaf


def _summary(leaf: Path) -> Path:
    return leaf / f"{BENCH}_cot_summary.json"


def test_buckets_flip_tasks_and_truncation_split(tmp_path):
    leaf = _build_leaf(tmp_path)
    out = cot_false_invalid_stats(_summary(leaf))

    # generation-level: 5 cot_false split 2/1/1/1 across invalid 0..3.
    assert out["cotfalse_total"] == 5
    assert (
        out["cot_false_complete_verdict"],
        out["cot_false_invalid1"],
        out["cot_false_invalid2"],
        out["cot_false_invalid3"],
    ) == (2, 1, 1, 1)
    assert out["cot_false_complete_verdict_share"] == 0.4
    assert out["cot_false_invalid1_share"] == 0.2
    # shares over cot_false sum to 1.0
    inv_shares = [
        out["cot_false_complete_verdict_share"],
        out["cot_false_invalid1_share"],
        out["cot_false_invalid2_share"],
        out["cot_false_invalid3_share"],
    ]
    assert round(sum(inv_shares), 6) == 1.0

    # counterfactual flip: g3,g4,g5 -> 3 of 5.
    assert out["cot_false_trunc_induced"] == 3
    assert out["trunc_induced_share"] == 0.6

    # task-level: questions 1,2,3 touched, of 4 total.
    assert out["cotfalse_tasks"] == 3
    assert out["cotfalse_tasks_ratio"] == 0.75

    # verdict-level truncation split: 5 truncated, 1 other of 6 invalid verdicts.
    assert out["cot_false_invalid_by_trunc"] == 5
    assert out["cot_false_invalid_by_other"] == 1
    assert out["cot_false_invalid_by_trunc_share"] == round(5 / 6, 6)


def test_no_judge_row_is_all_blank(tmp_path):
    # A leaf with no *_cot_majority.jsonl mimics a No-Judge (base) row.
    leaf = tmp_path / "base"
    leaf.mkdir()
    (leaf / f"{BENCH}_summary.json").write_text(json.dumps({"total_tasks": 4}))
    out = cot_false_invalid_stats(leaf / f"{BENCH}_summary.json")
    assert set(out) == set(COT_FALSE_COLUMNS)
    assert all(v is None for v in out.values())


def test_zero_cot_false(tmp_path):
    leaf = tmp_path / "leaf0"
    leaf.mkdir()
    _write_jsonl(
        leaf / f"{BENCH}_cot_majority.jsonl",
        [{"task_id": f"{BENCH}/1_gen_0", "yes_count": 3, "no_count": 0, "invalid_count": 0, "majority_correct": True}],
    )
    (leaf / f"{BENCH}_cot_summary.json").write_text(json.dumps({"total_tasks": 4}))
    out = cot_false_invalid_stats(_summary(leaf))
    assert out["cotfalse_total"] == 0
    assert out["cot_false_complete_verdict"] == 0
    assert out["cot_false_complete_verdict_share"] is None  # 0/0 -> None, not a crash
    assert out["cot_false_trunc_induced"] == 0
    assert out["cotfalse_tasks"] == 0
    assert out["cotfalse_tasks_ratio"] == 0.0
    assert out["cot_false_invalid_by_trunc"] == 0
    assert out["cot_false_invalid_by_trunc_share"] is None


def test_unaligned_verdict_files_blank_the_split_only(tmp_path):
    leaf = _build_leaf(tmp_path, misalign=True)
    out = cot_false_invalid_stats(_summary(leaf))
    # generation/task level still computed from the majority file ...
    assert out["cotfalse_total"] == 5
    assert out["cot_false_trunc_induced"] == 3
    assert out["cotfalse_tasks"] == 3
    # ... but the truncation split is refused rather than reported wrong.
    assert out["cot_false_invalid_by_trunc"] is None
    assert out["cot_false_invalid_by_other"] is None
    assert out["cot_false_invalid_by_trunc_share"] is None


def test_missing_raw_blanks_the_split_only(tmp_path):
    leaf = _build_leaf(tmp_path, drop_raw=True)
    out = cot_false_invalid_stats(_summary(leaf))
    assert out["cotfalse_total"] == 5
    assert out["cot_false_invalid_by_trunc"] is None
    assert out["cot_false_invalid_by_other"] is None


def test_legacy_majority_schema_via_solutions_list(tmp_path):
    """Legacy majority files predate yes/no/invalid_count and carry only a raw
    ``solutions`` list, the buckets must be derived from it."""
    leaf = tmp_path / "legacy"
    leaf.mkdir()
    _write_jsonl(
        leaf / f"{BENCH}_cot_majority.jsonl",
        [
            {"task_id": f"{BENCH}/1_gen_0", "solutions": ["yes", "yes", "yes"], "majority_correct": True},  # approved
            {"task_id": f"{BENCH}/1_gen_1", "solutions": ["no", "no", "no"], "majority_correct": False},  # inv0
            {
                "task_id": f"{BENCH}/2_gen_0",
                "solutions": ["yes", "no", "invalid_format"],
                "majority_correct": False,
            },  # inv1, flip (1+1>1)
            {"task_id": f"{BENCH}/2_gen_1", "solutions": ["", "", ""], "majority_correct": False},  # inv3, flip
        ],
    )
    (leaf / f"{BENCH}_cot_summary.json").write_text(json.dumps({"total_tasks": 2}))
    out = cot_false_invalid_stats(leaf / f"{BENCH}_cot_summary.json")
    assert out["cotfalse_total"] == 3
    assert (
        out["cot_false_complete_verdict"],
        out["cot_false_invalid1"],
        out["cot_false_invalid2"],
        out["cot_false_invalid3"],
    ) == (1, 1, 0, 1)
    assert out["cot_false_trunc_induced"] == 2  # the inv1 (1,1) and the all-invalid one
    assert out["cotfalse_tasks"] == 2  # questions 1 and 2
