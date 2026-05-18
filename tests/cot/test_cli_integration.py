"""Smoke test: the four ``evalhub cot ...`` CLI commands behave end-to-end.

We exercise them via Typer's ``CliRunner`` so the test is hermetic — no
subprocess, no shell, no network.
"""

import orjson
import pytest

typer_testing = pytest.importorskip("typer.testing")

from evalhub.cli import app  # noqa: E402
from evalhub.cot.ids import encode  # noqa: E402

runner = typer_testing.CliRunner()


def _write_jsonl(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")


def test_cot_finalize_cli(tmp_path):
    base_results = tmp_path / "b_results.jsonl"
    base_raw = tmp_path / "b_raw.jsonl"
    judge_solutions = tmp_path / "j.jsonl"
    out_dir = tmp_path / "out"

    _write_jsonl(
        base_results,
        [
            {
                "task_id": "T",
                "ground_truth": "x",
                "correct": [True, False, True],
                "solutions": ["x", "y", "x"],
                "pass_at_k": {"1": 0.66, "3": 1.0},
            }
        ],
    )
    _write_jsonl(
        base_raw,
        [{"task_id": "T", "response": {"choices": [{"message": {"content": str(i)}}]}} for i in range(3)],
    )
    _write_jsonl(
        judge_solutions,
        [
            {"task_id": encode("T", 0), "solution": "yes"},
            {"task_id": encode("T", 2), "solution": "no"},
        ],
    )

    result = runner.invoke(
        app,
        [
            "cot",
            "finalize",
            "--base-results", str(base_results),
            "--base-raw", str(base_raw),
            "--judge-solutions", str(judge_solutions),
            "--output-dir", str(out_dir),
            "--benchmark", "bench",
        ],
    )
    assert result.exit_code == 0, result.stdout
    summary = orjson.loads((out_dir / "bench_cot_summary.json").read_bytes())
    assert summary["pass_at_k"]["1"] > 0  # one survivor in three -> > 0
    assert summary["pass_at_k"]["3"] == 1.0


def test_cot_extract_then_aggregate_then_metrics_cli(tmp_path):
    base_results = tmp_path / "b_results.jsonl"
    base_raw = tmp_path / "b_raw.jsonl"
    judge_input = tmp_path / "judge_in.jsonl"
    judge_solutions = tmp_path / "judge_out.jsonl"
    majority = tmp_path / "majority.jsonl"
    cot_results = tmp_path / "cot_results.jsonl"
    summary = tmp_path / "summary.json"

    _write_jsonl(
        base_results,
        [
            {
                "task_id": "T",
                "ground_truth": "x",
                "correct": [True],
                "solutions": ["x"],
                "pass_at_k": {"1": 1.0},
            }
        ],
    )
    _write_jsonl(
        base_raw,
        [{"task_id": "T", "response": {"choices": [{"message": {"content": "x"}}]}}],
    )

    r1 = runner.invoke(
        app,
        ["cot", "extract",
         "--base-results", str(base_results),
         "--base-raw", str(base_raw),
         "--output", str(judge_input)],
    )
    assert r1.exit_code == 0, r1.stdout
    assert judge_input.exists() and judge_input.stat().st_size > 0

    _write_jsonl(judge_solutions, [{"task_id": encode("T", 0), "solution": "yes"}])

    r2 = runner.invoke(
        app,
        ["cot", "aggregate",
         "--judge-solutions", str(judge_solutions),
         "--output", str(majority)],
    )
    assert r2.exit_code == 0, r2.stdout

    r3 = runner.invoke(
        app,
        ["cot", "metrics",
         "--base-results", str(base_results),
         "--majority", str(majority),
         "--output", str(cot_results),
         "--summary", str(summary)],
    )
    assert r3.exit_code == 0, r3.stdout
    summary_data = orjson.loads(summary.read_bytes())
    assert summary_data["pass_at_k"]["1"] == 1.0
    assert summary_data["cons_at_k"] == 1.0
