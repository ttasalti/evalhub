"""Tests for the incremental (per-run) side of report_tasks.csv.

Exercises the full public entry points (`aggregate_results` / `upsert_summary`)
so these tests prove the actual integration hooks in aggregate.py, not just
evalhub.report.tasks in isolation (that's test_tasks.py's job).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from evalhub.report.aggregate import aggregate_results, upsert_summary  # noqa: E402
from evalhub.report.tasks import TASK_COLUMNS, TASK_CSV_NAME  # noqa: E402

BENCH = "aime2026"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _write_per_task_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "true",
        "false",
        "cot_false",
        "invalid_format",
        "pass@1",
        "ground_truth",
        "majority_vote",
        "is_correct_majority",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {
                "invalid_format": 0,
                "pass@1": "",
                "ground_truth": "",
                "majority_vote": "",
                "is_correct_majority": "",
            }
            row.update(r)
            w.writerow(row)


def _base_leaf(root: Path, model: str = "qwen-mini", max_tokens: int = 2048) -> Path:
    return root / "base" / model / f"{BENCH}__t0.6__max{max_tokens}__n64"


def _judge_leaf(
    root: Path,
    model: str = "qwen-mini",
    judge: str = "qwen-judge",
    judge_state: str = "think",
    max_tokens: int = 2048,
    judge_max_tokens: int = 16384,
) -> Path:
    return (
        root
        / "base"
        / model
        / "judged_by"
        / f"{judge}__state-{judge_state}__t0.6__max{judge_max_tokens}"
        / f"{BENCH}__t0.6__max{max_tokens}__n64"
    )


def _write_base(root: Path, task_rows: list[dict], model: str = "qwen-mini", max_tokens: int = 2048) -> Path:
    leaf = _base_leaf(root, model, max_tokens=max_tokens)
    summary = leaf / f"{BENCH}_summary.json"
    _write_json(summary, {"pass_at_k": {"1": 0.5}, "cons_at_k": 0.5})
    _write_per_task_csv(leaf / f"{BENCH}_per_task.csv", task_rows)
    return summary


def _write_judged(
    root: Path,
    task_rows: list[dict],
    majority_rows: list[dict],
    model: str = "qwen-mini",
    judge_state: str = "think",
    max_tokens: int = 2048,
    judge_max_tokens: int = 16384,
) -> Path:
    leaf = _judge_leaf(root, model, judge_state=judge_state, max_tokens=max_tokens, judge_max_tokens=judge_max_tokens)
    summary = leaf / f"{BENCH}_cot_summary.json"
    _write_json(summary, {"pass_at_k": {"1": 0.4}, "cons_at_k": 0.4})
    _write_per_task_csv(leaf / f"{BENCH}_cot_per_task.csv", task_rows)
    _write_jsonl(leaf / f"{BENCH}_cot_majority.jsonl", majority_rows)
    return summary


def test_upsert_replaces_not_duplicates(tmp_path):
    root = tmp_path / "results"
    csv_path = tmp_path / "report.csv"
    tasks_csv = tmp_path / TASK_CSV_NAME

    majority_v1 = [
        {"task_id": f"{BENCH}/1_gen_0", "yes_count": 0, "no_count": 3, "invalid_count": 0, "majority_correct": False},
    ]
    summary = _write_judged(root, [{"task_id": f"{BENCH}/1", "true": 5, "false": 0, "cot_false": 1}], majority_v1)
    upsert_summary(summary, csv_path, root)
    df1 = pd.read_csv(tasks_csv)
    assert len(df1) == 1
    assert df1.iloc[0]["cot_false_complete_verdict"] == 1

    # Re-run with mutated source data for the same run identity.
    majority_v2 = [
        {
            "task_id": f"{BENCH}/1_gen_0",
            "yes_count": 1,
            "no_count": 1,
            "invalid_count": 1,
            "majority_correct": False,
        },  # now a flip instead
    ]
    _write_per_task_csv(
        _judge_leaf(root) / f"{BENCH}_cot_per_task.csv",
        [{"task_id": f"{BENCH}/1", "true": 4, "false": 0, "cot_false": 2}],
    )
    _write_jsonl(_judge_leaf(root) / f"{BENCH}_cot_majority.jsonl", majority_v2)
    upsert_summary(summary, csv_path, root)

    df2 = pd.read_csv(tasks_csv)
    assert len(df2) == 1  # replaced, not duplicated
    assert df2.iloc[0]["true_count"] == 4
    assert df2.iloc[0]["cot_false_count"] == 2
    assert df2.iloc[0]["cot_false_trunc_induced"] == 1
    assert df2.iloc[0]["cot_false_complete_verdict"] == 0


def test_upsert_judge_state_disambiguates(tmp_path):
    root = tmp_path / "results"
    csv_path = tmp_path / "report.csv"
    tasks_csv = tmp_path / TASK_CSV_NAME

    think_summary = _write_judged(
        root,
        [{"task_id": f"{BENCH}/1", "true": 5, "false": 0, "cot_false": 0}],
        [],
        judge_state="think",
    )
    nonthink_summary = _write_judged(
        root,
        [{"task_id": f"{BENCH}/1", "true": 3, "false": 0, "cot_false": 0}],
        [],
        judge_state="non-think",
    )
    upsert_summary(think_summary, csv_path, root)
    upsert_summary(nonthink_summary, csv_path, root)

    df = pd.read_csv(tasks_csv)
    assert len(df) == 2  # both judge_state variants coexist, neither clobbered
    assert set(df["judge_state"]) == {"think", "non-think"}
    by_state = {r["judge_state"]: r["true_count"] for _, r in df.iterrows()}
    assert by_state["think"] == 5
    assert by_state["non-think"] == 3


def test_upsert_max_tokens_disambiguates(tmp_path):
    """Regression test for the collision found in production: two judged runs
    of the same model/state/benchmark/judge at different max_tokens (e.g. a
    16k and a 32k think-mode re-run) must NOT clobber each other in either
    report.csv or report_tasks.csv."""
    root = tmp_path / "results"
    csv_path = tmp_path / "report.csv"
    tasks_csv = tmp_path / TASK_CSV_NAME

    small_summary = _write_judged(
        root,
        [{"task_id": f"{BENCH}/1", "true": 5, "false": 0, "cot_false": 0}],
        [],
        max_tokens=16384,
        judge_max_tokens=16384,
    )
    big_summary = _write_judged(
        root,
        [{"task_id": f"{BENCH}/1", "true": 3, "false": 0, "cot_false": 0}],
        [],
        max_tokens=32768,
        judge_max_tokens=16384,
    )
    upsert_summary(small_summary, csv_path, root)
    upsert_summary(big_summary, csv_path, root)

    report_df = pd.read_csv(csv_path)
    assert len(report_df) == 2  # both max_tokens variants coexist in report.csv too
    assert set(report_df["max_tokens"]) == {16384, 32768}

    tasks_df = pd.read_csv(tasks_csv)
    assert len(tasks_df) == 2  # both variants coexist, neither clobbered
    assert set(tasks_df["max_tokens"]) == {16384, 32768}
    by_max = {r["max_tokens"]: r["true_count"] for _, r in tasks_df.iterrows()}
    assert by_max[16384] == 5
    assert by_max[32768] == 3


def test_upsert_missing_file_is_noop(tmp_path):
    root = tmp_path / "results"
    csv_path = tmp_path / "report.csv"
    tasks_csv = tmp_path / TASK_CSV_NAME

    good = _write_base(root, [{"task_id": f"{BENCH}/1", "true": 1, "false": 63, "cot_false": 0}])
    upsert_summary(good, csv_path, root)
    before = tasks_csv.read_bytes()

    # A second, distinct run whose leaf has no per_task.csv at all.
    other_leaf = _base_leaf(root, model="other-model")
    other_summary = other_leaf / f"{BENCH}_summary.json"
    _write_json(other_summary, {"pass_at_k": {"1": 0.1}, "cons_at_k": 0.1})
    upsert_summary(other_summary, csv_path, root)

    after = tasks_csv.read_bytes()
    assert before == after  # untouched -- missing-source run contributed nothing


def test_aggregate_upsert_parity(tmp_path):
    root = tmp_path / "results"
    base_rows = [
        {"task_id": f"{BENCH}/1", "true": 10, "false": 54, "cot_false": 0},
        {"task_id": f"{BENCH}/2", "true": 0, "false": 64, "cot_false": 0},
    ]
    base_summary = _write_base(root, base_rows)

    judged_rows = [
        {"task_id": f"{BENCH}/1", "true": 8, "false": 0, "cot_false": 2},
        {"task_id": f"{BENCH}/2", "true": 0, "false": 0, "cot_false": 0},
    ]
    majority = [
        {"task_id": f"{BENCH}/1_gen_0", "yes_count": 0, "no_count": 3, "invalid_count": 0, "majority_correct": False},
        {"task_id": f"{BENCH}/1_gen_1", "yes_count": 1, "no_count": 1, "invalid_count": 1, "majority_correct": False},
    ]
    judged_summary = _write_judged(root, judged_rows, majority)

    # Approach A: one full rebuild.
    csv_a = tmp_path / "a" / "report.csv"
    aggregate_results(root, csv_a)
    tasks_a = pd.read_csv(csv_a.parent / TASK_CSV_NAME)

    # Approach B: two sequential incremental upserts.
    csv_b = tmp_path / "b" / "report.csv"
    upsert_summary(base_summary, csv_b, root)
    upsert_summary(judged_summary, csv_b, root)
    tasks_b = pd.read_csv(csv_b.parent / TASK_CSV_NAME)

    sort_cols = ["model", "state", "benchmark", "judge_model", "judge_state", "task_id"]
    tasks_a = tasks_a.sort_values(sort_cols).reset_index(drop=True)
    tasks_b = tasks_b.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(tasks_a[list(TASK_COLUMNS)], tasks_b[list(TASK_COLUMNS)])


def test_aggregate_writes_tasks_csv_next_to_report_csv(tmp_path):
    root = tmp_path / "results"
    _write_base(root, [{"task_id": f"{BENCH}/1", "true": 1, "false": 63, "cot_false": 0}])
    out = tmp_path / "nested" / "report.csv"
    aggregate_results(root, out)
    assert (out.parent / TASK_CSV_NAME).exists()
