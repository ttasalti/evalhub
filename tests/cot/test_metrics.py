import math

import orjson

from evalhub.cot import COT_FALSE_LABEL, apply_cot_metrics
from evalhub.cot.ids import encode
from evalhub.utils.metrics import compute_pass_at_k


def _write_jsonl(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")


def _read_jsonl(path):
    return [orjson.loads(line) for line in open(path, "rb") if line.strip()]


def test_cot_veto_downgrades_correct_to_cot_false(tmp_path):
    base_results = tmp_path / "base.jsonl"
    majority = tmp_path / "maj.jsonl"
    output = tmp_path / "out.jsonl"
    summary = tmp_path / "summary.json"

    # Task with 4 generations: original correct = [T, T, F, T]
    # Judge says: gen0=yes, gen1=no (downgrade), gen3=yes
    _write_jsonl(
        base_results,
        [
            {
                "task_id": "T1",
                "ground_truth": "x",
                "correct": [True, True, False, True],
                "solutions": ["x", "x", "y", "x"],
                "pass_at_k": {"1": 0.75, "4": 1.0},
            }
        ],
    )
    _write_jsonl(
        majority,
        [
            {"task_id": encode("T1", 0), "majority_correct": True},
            {"task_id": encode("T1", 1), "majority_correct": False},
            {"task_id": encode("T1", 3), "majority_correct": True},
        ],
    )

    result = apply_cot_metrics(base_results, majority, output, summary)
    rows = _read_jsonl(output)
    assert rows[0]["correct"] == [True, COT_FALSE_LABEL, False, True]
    # Surviving true count = 2 out of 4. Pass@1 = 0.5; Pass@4 = 1.0.
    assert math.isclose(rows[0]["pass_at_k"]["1"], compute_pass_at_k(4, 2, 1))
    assert math.isclose(rows[0]["pass_at_k"]["4"], compute_pass_at_k(4, 2, 4))
    assert math.isclose(result["pass_at_k"]["1"], 0.5)


def test_cot_pass_k_zero_when_all_vetoed(tmp_path):
    base_results = tmp_path / "base.jsonl"
    majority = tmp_path / "m.jsonl"
    output = tmp_path / "o.jsonl"
    summary = tmp_path / "s.json"

    _write_jsonl(
        base_results,
        [
            {
                "task_id": "Q",
                "ground_truth": "x",
                "correct": [True, True, True],
                "solutions": ["x", "x", "x"],
                "pass_at_k": {"1": 1.0, "3": 1.0},
            }
        ],
    )
    _write_jsonl(
        majority,
        [{"task_id": encode("Q", i), "majority_correct": False} for i in range(3)],
    )

    result = apply_cot_metrics(base_results, majority, output, summary)
    assert result["pass_at_k"]["1"] == 0.0
    assert result["pass_at_k"]["3"] == 0.0
    rows = _read_jsonl(output)
    assert rows[0]["correct"] == [COT_FALSE_LABEL, COT_FALSE_LABEL, COT_FALSE_LABEL]


def test_metrics_aggregate_across_multiple_tasks(tmp_path):
    base_results = tmp_path / "base.jsonl"
    majority = tmp_path / "m.jsonl"
    output = tmp_path / "o.jsonl"
    summary = tmp_path / "s.json"
    stats_path = tmp_path / "stats.json"

    _write_jsonl(
        base_results,
        [
            {
                "task_id": "A",
                "correct": [True, True],
                "solutions": ["a", "a"],
                "pass_at_k": {"1": 1.0, "2": 1.0},
            },
            {
                "task_id": "B",
                "correct": [True, False],
                "solutions": ["b", "x"],
                "pass_at_k": {"1": 0.5, "2": 1.0},
            },
        ],
    )
    _write_jsonl(
        majority,
        [
            {"task_id": encode("A", 0), "majority_correct": True},
            {"task_id": encode("A", 1), "majority_correct": False},
            {"task_id": encode("B", 0), "majority_correct": True},
        ],
    )

    result = apply_cot_metrics(base_results, majority, output, summary, stats_path=stats_path)
    # A: n=2, surviving correct=1 -> pass@1=0.5, pass@2=1.0
    # B: n=2, surviving correct=1 -> pass@1=0.5, pass@2=1.0
    assert math.isclose(result["pass_at_k"]["1"], 0.5)
    assert math.isclose(result["pass_at_k"]["2"], 1.0)
    assert math.isclose(result["cons_at_k"], 1.0)  # both tasks: majority answer is correct

    stats = orjson.loads(stats_path.read_bytes())
    assert stats["total_tasks"] == 2
    assert stats["cot_false_count"] == 1
    assert stats["true_count"] == 2
    assert stats["false_count"] == 1
