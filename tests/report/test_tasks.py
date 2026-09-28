"""Tests for :mod:`evalhub.report.tasks`, the problem-level report_tasks.csv writer.

Each RunRecord explodes into one row per task_id, reusing the per-task source
files (`*_per_task.csv` / `*_cot_per_task.csv`) that evaluation/CoT-finalize
already write to disk, these tests build small, hand-checkable versions of
those files rather than re-deriving correctness from raw generations.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from evalhub.report.scan import record_from_summary  # noqa: E402
from evalhub.report.tasks import (  # noqa: E402
    TASK_COLUMNS,
    _normalize_task_id,  # noqa: E402
    build_task_dataframe,
    cot_false_stats_per_task,
    natural_task_id_key,
    task_rows_from_record,
)

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


def _base_leaf(root: Path, model: str = "qwen-mini") -> Path:
    return root / "base" / model / f"{BENCH}__t0.6__max2048__n64"


def _judge_leaf(root: Path, model: str = "qwen-mini", judge: str = "qwen-judge", judge_state: str = "think") -> Path:
    return (
        root
        / "base"
        / model
        / "judged_by"
        / f"{judge}__state-{judge_state}__t0.6__max16384"
        / f"{BENCH}__t0.6__max2048__n64"
    )


def _no_judge_record(root: Path, task_rows: list[dict], model: str = "qwen-mini"):
    leaf = _base_leaf(root, model)
    _write_json(leaf / f"{BENCH}_summary.json", {"pass_at_k": {"1": 0.5}, "cons_at_k": 0.5})
    _write_per_task_csv(leaf / f"{BENCH}_per_task.csv", task_rows)
    return record_from_summary(leaf / f"{BENCH}_summary.json", root)


def _judged_record(
    root: Path, task_rows: list[dict], majority_rows: list[dict], model: str = "qwen-mini", judge_state: str = "think"
):
    leaf = _judge_leaf(root, model, judge_state=judge_state)
    _write_json(leaf / f"{BENCH}_cot_summary.json", {"pass_at_k": {"1": 0.4}, "cons_at_k": 0.4})
    _write_per_task_csv(leaf / f"{BENCH}_cot_per_task.csv", task_rows)
    _write_jsonl(leaf / f"{BENCH}_cot_majority.jsonl", majority_rows)
    return record_from_summary(leaf / f"{BENCH}_cot_summary.json", root)


def test_no_judge_row_columns(tmp_path):
    rows = [
        {"task_id": f"{BENCH}/1", "true": 11, "false": 53, "cot_false": 0},
        {"task_id": f"{BENCH}/2", "true": 0, "false": 64, "cot_false": 0},
    ]
    record = _no_judge_record(tmp_path, rows)
    out = task_rows_from_record(record)
    assert len(out) == 2
    assert list(out[0].keys()) == list(TASK_COLUMNS)

    r1 = next(r for r in out if r["task_id"] == f"{BENCH}/1")
    assert r1["n_generations"] == 64
    assert r1["true_count"] == 11
    assert r1["false_count"] == 53
    assert r1["judge_model"] is None
    assert r1["judge_state"] is None
    assert r1["cot_false_count"] is None
    assert r1["cot_false_trunc_induced"] is None
    assert r1["cot_false_complete_verdict"] is None
    assert r1["model"] == "qwen-mini"
    assert r1["state"] == "base"
    assert r1["benchmark"] == BENCH
    assert r1["max_tokens"] == 2048


def test_judged_row_mix_complete_verdict_and_trunc_induced(tmp_path):
    # Task 7: two vetoed generations -- one complete-verdict (0 invalid), one
    # trunc-induced (a flip: yes+invalid > no). Task 9: zero vetoed
    # generations -> both breakdown counters are legitimately 0, not blank.
    task_rows = [
        {"task_id": f"{BENCH}/7", "true": 9, "false": 0, "cot_false": 2},
        {"task_id": f"{BENCH}/9", "true": 10, "false": 0, "cot_false": 0},
    ]
    majority_rows = [
        {
            "task_id": f"{BENCH}/7_gen_0",
            "yes_count": 0,
            "no_count": 3,
            "invalid_count": 0,
            "majority_correct": False,
        },  # complete verdict
        {
            "task_id": f"{BENCH}/7_gen_1",
            "yes_count": 1,
            "no_count": 1,
            "invalid_count": 1,
            "majority_correct": False,
        },  # trunc-induced flip (1+1>1)
        {
            "task_id": f"{BENCH}/7_gen_2",
            "yes_count": 3,
            "no_count": 0,
            "invalid_count": 0,
            "majority_correct": True,
        },  # approved, ignored
    ]
    record = _judged_record(tmp_path, task_rows, majority_rows)
    out = {r["task_id"]: r for r in task_rows_from_record(record)}

    r7 = out[f"{BENCH}/7"]
    assert r7["true_count"] == 9
    assert r7["cot_false_count"] == 2
    assert r7["cot_false_complete_verdict"] == 1
    assert r7["cot_false_trunc_induced"] == 1
    assert r7["n_generations"] is None
    assert r7["false_count"] is None
    assert r7["judge_model"] == "qwen-judge"
    assert r7["judge_state"] == "think"
    assert r7["judge_max_tokens"] == 16384
    assert list(r7.keys()) == list(TASK_COLUMNS)

    r9 = out[f"{BENCH}/9"]
    assert r9["cot_false_count"] == 0
    assert r9["cot_false_complete_verdict"] == 0  # legitimate zero, not blank
    assert r9["cot_false_trunc_induced"] == 0


def test_missing_per_task_csv_skips_with_warning(tmp_path):
    from evalhub.utils.logger import logger

    leaf = _base_leaf(tmp_path)
    _write_json(leaf / f"{BENCH}_summary.json", {"pass_at_k": {"1": 0.5}, "cons_at_k": 0.5})
    # deliberately no *_per_task.csv written
    record = record_from_summary(leaf / f"{BENCH}_summary.json", tmp_path)

    # evalhub's logger is loguru, bound to the real stderr fd at import time,
    # so neither pytest's caplog (stdlib logging only) nor capsys/capfd
    # reliably see it; a temporary loguru sink is the robust way to assert on
    # a message.
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        out = task_rows_from_record(record)
    finally:
        logger.remove(sink_id)

    assert out == []
    assert any("per_task.csv" in str(m) for m in messages)


def test_build_task_dataframe_empty_is_correctly_headed(tmp_path):
    leaf = _base_leaf(tmp_path)
    _write_json(leaf / f"{BENCH}_summary.json", {"pass_at_k": {"1": 0.5}, "cons_at_k": 0.5})
    record = record_from_summary(leaf / f"{BENCH}_summary.json", tmp_path)
    df = build_task_dataframe([record])
    assert len(df) == 0
    assert list(df.columns) == list(TASK_COLUMNS)


def test_natural_sort_order(tmp_path):
    rows = [
        {"task_id": f"{BENCH}/10", "true": 1, "false": 0, "cot_false": 0},
        {"task_id": f"{BENCH}/2", "true": 1, "false": 0, "cot_false": 0},
        {"task_id": f"{BENCH}/7", "true": 1, "false": 0, "cot_false": 0},
        {"task_id": "no-trailing-int", "true": 1, "false": 0, "cot_false": 0},
    ]
    record = _no_judge_record(tmp_path, rows)
    out = task_rows_from_record(record)
    ids = [r["task_id"] for r in out]
    assert ids == [f"{BENCH}/2", f"{BENCH}/7", f"{BENCH}/10", "no-trailing-int"]


def test_natural_task_id_key_string_fallback():
    assert natural_task_id_key("abc") < natural_task_id_key("abd")
    assert natural_task_id_key("X/2") < natural_task_id_key("X/10")


def test_cot_false_stats_per_task_empty_for_no_judge_leaf(tmp_path):
    leaf = _base_leaf(tmp_path)
    _write_json(leaf / f"{BENCH}_summary.json", {"pass_at_k": {"1": 0.5}, "cons_at_k": 0.5})
    assert cot_false_stats_per_task(leaf / f"{BENCH}_summary.json") == {}


def test_normalize_task_id_legacy_prefixes():
    # Real-data finding: aime2026_pt/aime2026_tr's task_id prefix changed at
    # some point; historical *_per_task.csv files still carry the old one.
    assert _normalize_task_id("AIME2026-PT/7") == "aime2026_pt/7"
    assert _normalize_task_id("AIME2026-TR/12") == "aime2026_tr/12"
    # Current-format and unrelated prefixes pass through unchanged.
    assert _normalize_task_id("aime2026_pt/7") == "aime2026_pt/7"
    assert _normalize_task_id("AIME2026/7") == "AIME2026/7"
    assert _normalize_task_id("tubitak_math2026/3") == "tubitak_math2026/3"


def test_task_rows_normalizes_legacy_task_id_prefix(tmp_path):
    """A per_task.csv written by an older code version (legacy prefix) must
    still land under the CURRENT prefix in report_tasks.csv, so the same
    problem doesn't fork into two different task_id values depending on
    which run produced it."""
    rows = [
        {"task_id": "AIME2026-PT/7", "true": 2, "false": 62, "cot_false": 0},
        {"task_id": "AIME2026-PT/9", "true": 0, "false": 64, "cot_false": 0},
    ]
    record = _no_judge_record(tmp_path, rows)
    out = task_rows_from_record(record)
    ids = {r["task_id"] for r in out}
    assert ids == {"aime2026_pt/7", "aime2026_pt/9"}
    assert not any(t.startswith("AIME2026-PT") for t in ids)


def test_judged_row_normalizes_legacy_prefix_and_still_matches_breakdown(tmp_path):
    """Regression: normalizing task_id from the per_task.csv must not break
    the lookup into cot_false_stats_per_task's breakdown dict, which is keyed
    off task_id decoded from *_cot_majority.jsonl's own (also legacy-prefixed)
    generation ids -- both sides must normalize the same way or the veto
    breakdown silently zeroes out for every legacy-prefix task."""
    task_rows = [{"task_id": "AIME2026-PT/7", "true": 1, "false": 0, "cot_false": 1}]
    majority_rows = [
        {
            "task_id": "AIME2026-PT/7_gen_0",
            "yes_count": 0,
            "no_count": 3,
            "invalid_count": 0,
            "majority_correct": False,
        },
    ]
    record = _judged_record(tmp_path, task_rows, majority_rows)
    out = task_rows_from_record(record)
    assert len(out) == 1
    assert out[0]["task_id"] == "aime2026_pt/7"
    assert out[0]["cot_false_complete_verdict"] == 1  # not silently 0


def test_zero_fill_when_judge_never_saw_any_base_correct(tmp_path):
    """pipeline_write_empty_cot_summary's case: the judge was never invoked
    because every task's base run had 0 correct generations. Every task_id
    (read from the sibling No-Judge per_task.csv) must get a legitimate
    all-zero judge row, not be skipped."""
    base_rows = [
        {"task_id": f"{BENCH}/1", "true": 0, "false": 64, "cot_false": 0},
        {"task_id": f"{BENCH}/2", "true": 0, "false": 64, "cot_false": 0},
        {"task_id": f"{BENCH}/3", "true": 0, "false": 64, "cot_false": 0},
    ]
    _no_judge_record(tmp_path, base_rows)  # writes the sibling base per_task.csv

    judge_leaf = _judge_leaf(tmp_path)
    _write_json(
        judge_leaf / f"{BENCH}_cot_summary.json",
        {"pass_at_k": {}, "cons_at_k": 0.0, "note": "no base-correct samples"},
    )
    # deliberately no *_cot_per_task.csv / *_cot_majority.jsonl written
    record = record_from_summary(judge_leaf / f"{BENCH}_cot_summary.json", tmp_path)

    out = task_rows_from_record(record)
    assert {r["task_id"] for r in out} == {f"{BENCH}/1", f"{BENCH}/2", f"{BENCH}/3"}
    for r in out:
        assert r["true_count"] == 0
        assert r["cot_false_count"] == 0
        assert r["cot_false_trunc_induced"] == 0
        assert r["cot_false_complete_verdict"] == 0
        assert r["n_generations"] is None
        assert r["false_count"] is None
        assert r["judge_model"] == "qwen-judge"
        assert r["judge_max_tokens"] == 16384


def test_zero_fill_does_not_apply_without_the_exact_note(tmp_path):
    """A missing *_cot_per_task.csv with no (or a different) note must NOT be
    silently zero-filled -- that would fabricate data for a genuinely
    incomplete/broken run. It should fall back to skip+warn."""
    from evalhub.utils.logger import logger

    _no_judge_record(tmp_path, [{"task_id": f"{BENCH}/1", "true": 5, "false": 59, "cot_false": 0}])
    judge_leaf = _judge_leaf(tmp_path)
    _write_json(judge_leaf / f"{BENCH}_cot_summary.json", {"pass_at_k": {"1": 0.1}, "cons_at_k": 0.1})
    record = record_from_summary(judge_leaf / f"{BENCH}_cot_summary.json", tmp_path)

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        out = task_rows_from_record(record)
    finally:
        logger.remove(sink_id)

    assert out == []
    assert any("per_task.csv" in str(m) for m in messages)


def test_zero_fill_falls_back_to_skip_when_base_per_task_also_missing(tmp_path):
    """The note is present but there's no sibling base per_task.csv to source
    task_ids from -- must not crash, must fall back to skip+warn."""
    judge_leaf = _judge_leaf(tmp_path)
    _write_json(
        judge_leaf / f"{BENCH}_cot_summary.json",
        {"pass_at_k": {}, "cons_at_k": 0.0, "note": "no base-correct samples"},
    )
    record = record_from_summary(judge_leaf / f"{BENCH}_cot_summary.json", tmp_path)
    out = task_rows_from_record(record)
    assert out == []
