"""End-to-end test of the local-only stages of the CoT pipeline.

Stage exercised:
  base_results + base_raw  --extract-->  judge_input
                              (judge inference would run here)
  judge_solutions          --aggregate-> majority
  base_results + majority  --metrics-->  CoT results + summary

This is the same code path that ``evalhub cot finalize`` and the
``scripts/run_cot_pass_at_k.sh`` script exercise after the judge model
has produced its answers.
"""

import math

import orjson

from evalhub.cot import finalize_cot_pipeline
from evalhub.cot.ids import encode


def _write_jsonl(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")


def test_end_to_end_finalize(tmp_path):
    base_results = tmp_path / "bench_results.jsonl"
    base_raw = tmp_path / "bench_raw.jsonl"
    judge_solutions = tmp_path / "cot_judge.jsonl"
    output_dir = tmp_path / "out"

    # Two tasks, n=4 each.
    _write_jsonl(
        base_results,
        [
            {
                "task_id": "AIME2025/0",
                "ground_truth": "42",
                "correct": [True, True, False, True],
                "solutions": ["42", "42", "wrong", "42"],
                "pass_at_k": {"1": 0.75, "2": 0.875, "4": 1.0},
            },
            {
                "task_id": "AIME2025/1",
                "ground_truth": "7",
                "correct": [False, True, False, False],
                "solutions": ["x", "7", "y", "z"],
                "pass_at_k": {"1": 0.25, "2": 0.5, "4": 1.0},
            },
        ],
    )
    # 4 raw rows per task in order.
    _write_jsonl(
        base_raw,
        (
            [
                {"task_id": "AIME2025/0", "response": {"choices": [{"message": {"content": f"gen-{i}"}}]}}
                for i in range(4)
            ]
            + [
                {"task_id": "AIME2025/1", "response": {"choices": [{"message": {"content": f"gen-{i}"}}]}}
                for i in range(4)
            ]
        ),
    )

    # Judge sees 4 correct generations:
    #   AIME2025/0_gen_0 -> 2 yes, 1 no   (approved)
    #   AIME2025/0_gen_1 -> 1 yes, 2 no   (vetoed)
    #   AIME2025/0_gen_3 -> 3 yes          (approved)
    #   AIME2025/1_gen_1 -> 1 yes, 1 no   (vetoed: tie does not pass strict majority)
    judge_records = (
        [{"task_id": encode("AIME2025/0", 0), "solution": v} for v in ("yes", "yes", "no")]
        + [{"task_id": encode("AIME2025/0", 1), "solution": v} for v in ("yes", "no", "no")]
        + [{"task_id": encode("AIME2025/0", 3), "solution": v} for v in ("yes", "yes", "yes")]
        + [{"task_id": encode("AIME2025/1", 1), "solution": v} for v in ("yes", "no")]
    )
    _write_jsonl(judge_solutions, judge_records)

    result = finalize_cot_pipeline(
        base_results_path=base_results,
        base_raw_path=base_raw,
        judge_solutions_path=judge_solutions,
        output_dir=output_dir,
        benchmark="bench",
    )

    assert result.extracted_count == 4
    assert result.aggregated_count == 4

    # AIME2025/0: surviving correct = 2 (gens 0,3); n=4
    # AIME2025/1: surviving correct = 0; n=4
    from evalhub.utils.metrics import compute_pass_at_k

    expected_pass_at_1 = (compute_pass_at_k(4, 2, 1) + compute_pass_at_k(4, 0, 1)) / 2
    expected_pass_at_2 = (compute_pass_at_k(4, 2, 2) + compute_pass_at_k(4, 0, 2)) / 2
    expected_pass_at_4 = (compute_pass_at_k(4, 2, 4) + compute_pass_at_k(4, 0, 4)) / 2

    assert math.isclose(result.summary["pass_at_k"]["1"], expected_pass_at_1)
    assert math.isclose(result.summary["pass_at_k"]["2"], expected_pass_at_2)
    assert math.isclose(result.summary["pass_at_k"]["4"], expected_pass_at_4)
    # Cons@K: AIME2025/0 majority answer "42" still has a True survivor -> +1
    #         AIME2025/1 majority answer "7" survivor was vetoed -> 0
    assert math.isclose(result.summary["cons_at_k"], 0.5)

    # Files exist with the expected names.
    assert result.summary_path.exists()
    assert result.output_results.exists()
    assert result.majority_path.exists()
    assert result.judge_input_path.exists()
