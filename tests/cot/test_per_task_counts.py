"""Tests for the per-task 4-way breakdown injected into result records.

`evalhub cot metrics` writes ``per_task_counts`` into every
``*_cot_results.jsonl`` row so drill-down tooling can answer
"how often was this specific question vetoed?" without recounting.
Base eval writes the same shape but with cot_false and invalid_format
fixed at 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import orjson

from evalhub.cot.ids import encode as encode_generation_id
from evalhub.cot.metrics import apply_cot_metrics


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")


def test_cot_metrics_writes_per_task_counts(tmp_path: Path) -> None:
    # 1 task, 4 generations: 2 base-correct, 2 base-wrong.
    base_results = [
        {
            "task_id": "TEST/1",
            "solutions": ["42", "42", "wrong", "wrong"],
            "ground_truth": "42",
            "correct": [True, True, False, False],
            "pass_at_k": {"1": 1.0, "2": 1.0, "4": 1.0},
            "majority_vote": "42",
            "is_correct_majority": True,
        }
    ]
    base_path = tmp_path / "test_results.jsonl"
    _write_jsonl(base_path, base_results)

    # Judge: approves the first base-correct (idx 0), rejects the second (idx 1).
    majority = [
        {"task_id": encode_generation_id("TEST/1", 0), "majority_correct": True},
        {"task_id": encode_generation_id("TEST/1", 1), "majority_correct": False},
    ]
    majority_path = tmp_path / "test_majority.jsonl"
    _write_jsonl(majority_path, majority)

    out_path = tmp_path / "test_cot_results.jsonl"
    summary_path = tmp_path / "test_cot_summary.json"
    stats_path = tmp_path / "test_cot_stats.json"

    apply_cot_metrics(base_path, majority_path, out_path, summary_path, stats_path)

    # Per-task breakdown should be present and correct on the output record.
    with out_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 1
    counts = records[0]["per_task_counts"]
    assert counts == {
        "true": 1,           # idx 0: still correct after judge approved
        "false": 2,          # idx 2, 3: base-wrong, never changed
        "cot_false": 1,      # idx 1: was True, judge vetoed
        "invalid_format": 0, # no unknown verdicts in this fixture
    }, counts


def test_cot_metrics_per_task_counts_no_judge_verdicts(tmp_path: Path) -> None:
    """When the judge produced no verdicts, every base-correct survives."""
    base_results = [
        {
            "task_id": "TEST/1",
            "solutions": ["x", "x", "x"],
            "ground_truth": "x",
            "correct": [True, False, True],
            "pass_at_k": {"1": 0.67, "2": 1.0, "4": 1.0},
            "majority_vote": "x",
            "is_correct_majority": True,
        }
    ]
    base_path = tmp_path / "r.jsonl"
    _write_jsonl(base_path, base_results)

    # Empty majority -> no vetoes applied.
    majority_path = tmp_path / "m.jsonl"
    _write_jsonl(majority_path, [])

    out_path = tmp_path / "out.jsonl"
    summary_path = tmp_path / "s.json"

    apply_cot_metrics(base_path, majority_path, out_path, summary_path)

    with out_path.open() as f:
        record = json.loads(f.readline())
    assert record["per_task_counts"] == {
        "true": 2,
        "false": 1,
        "cot_false": 0,
        "invalid_format": 0,
    }
