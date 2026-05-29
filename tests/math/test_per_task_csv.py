"""Per-task CSV emission alongside summary.json / results.jsonl.

These tests use the public ``write_per_task_csv`` helper directly (rather than
spinning up a full Dataset.evaluate run) so they don't depend on a model server.
"""

from __future__ import annotations

import csv

from evalhub.benchmarks.math.base import write_per_task_csv


def _sample_results():
    return [
        {
            "task_id": "tubitak_math2026/1",
            "pass_at_k": {"1": 0.5, "4": 0.9},
            "per_task_counts": {"true": 32, "false": 32, "cot_false": 0, "invalid_format": 0},
            "ground_truth": "105^\\circ",
            "majority_vote": "105^\\circ",
            "is_correct_majority": True,
        },
        {
            "task_id": "tubitak_math2026/2",
            "pass_at_k": {"1": 0.1, "4": 0.3},
            "per_task_counts": {"true": 8, "false": 50, "cot_false": 6, "invalid_format": 0},
            "ground_truth": "2",
            "majority_vote": "3",
            "is_correct_majority": False,
        },
    ]


def test_writes_one_row_per_task(tmp_path):
    out = tmp_path / "x_per_task.csv"
    write_per_task_csv(_sample_results(), out, has_cot=True)
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["task_id"] for r in rows] == [
        "tubitak_math2026/1",
        "tubitak_math2026/2",
    ]
    assert rows[1]["cot_false"] == "6"
    assert rows[1]["is_correct_majority"] == "False"


def test_columns_include_per_k_pass_rates(tmp_path):
    out = tmp_path / "x_per_task.csv"
    write_per_task_csv(_sample_results(), out, has_cot=False)
    with out.open(newline="") as f:
        header = next(csv.reader(f))
    # Counts + each K from pass_at_k + ground truth + majority.
    assert "pass@1" in header and "pass@4" in header
    assert header.index("true") < header.index("pass@1")
    assert header[-1] == "is_correct_majority"


def test_empty_results_is_safe(tmp_path):
    out = tmp_path / "x_per_task.csv"
    write_per_task_csv([], out, has_cot=False)
    # No file written when no rows — caller must not crash.
    assert not out.exists()
