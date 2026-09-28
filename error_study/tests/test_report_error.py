"""Regression test for the error-study task/question report (report_error.py).

Verifies both grains (per solution-variant and per question), the any/majority/all
judge-approval thresholds, and the answer_still_correct flag.
"""

from __future__ import annotations

import pandas as pd

from error_study.report_error import build


def _gen_rows(error_type, row_id, task_id, verdicts, old=5, new=6, menu="pm12", loc=50.0):
    return [
        {
            "model": "M",
            "benchmark": "aime2026",
            "error_type": error_type,
            "row_id": row_id,
            "task_id": task_id,
            "old_number": old,
            "new_number": new,
            "wrong_number_menu": menu,
            "change_location_pct": loc,
            "gen_index": i,
            "judge_verdict": v,
        }
        for i, v in enumerate(verdicts)
    ]


def test_report_error_thresholds_and_grains(tmp_path):
    # ONE question "Q/1" with two solutions: g0 unanimous 'yes', g1 split (1 yes / 2 no).
    rows = _gen_rows("boxed_only", "M|aime2026|Q/1|g0", "Q/1", ["yes", "yes", "yes"]) + _gen_rows(
        "boxed_only", "M|aime2026|Q/1|g1", "Q/1", ["no", "no", "yes"]
    )
    tasks, q = build(pd.DataFrame(rows), str(tmp_path))

    # per solution-variant (row_id) grain
    assert len(tasks) == 2
    g0 = tasks[tasks.row_id.str.endswith("g0")].iloc[0]
    assert (g0.yes, g0.no, g0.invalid) == (3, 0, 0)
    assert (g0.correct_any, g0.correct_maj, g0.correct_all) == (1, 1, 1)
    assert (g0.veto_any, g0.veto_maj, g0.veto_all) == (0, 0, 0)
    # boxed_only rewrites the final boxed answer -> no longer answer-correct
    assert bool(g0.answer_still_correct) is False
    g1 = tasks[tasks.row_id.str.endswith("g1")].iloc[0]
    assert (g1.yes, g1.no, g1.invalid) == (1, 2, 0)
    assert (g1.correct_any, g1.correct_maj, g1.correct_all) == (1, 0, 0)
    assert (g1.veto_any, g1.veto_maj, g1.veto_all) == (1, 1, 0)

    # per question (task_id) grain
    assert len(q) == 1
    row = q.iloc[0]
    assert row.n_solutions == 2 and row.sum_yes == 4 and row.sum_no == 2
    assert row.correct_any == 2 and row["correct_any_%"] == 100.0
    assert row.correct_maj == 1 and row["correct_maj_%"] == 50.0
    assert row.correct_all == 1 and row["correct_all_%"] == 50.0
    assert row.veto_any == 1 and row["veto_any_%"] == 50.0
    assert row.veto_all == 0 and row["veto_all_%"] == 0.0


def test_report_error_answer_still_correct_by_type(tmp_path):
    # clean + intermediate + truncated keep the correct boxed; boxed_only + consistent don't.
    rows = []
    for et in ("clean", "intermediate_error", "truncated", "boxed_only", "consistent_error"):
        rows += _gen_rows(et, f"M|aime2026|Q/1|{et}", "Q/1", ["yes", "no", "yes"])
    tasks, _ = build(pd.DataFrame(rows), str(tmp_path))
    flag = dict(zip(tasks.error_type, tasks.answer_still_correct, strict=True))
    assert flag["clean"] and flag["intermediate_error"] and flag["truncated"]
    assert not flag["boxed_only"] and not flag["consistent_error"]
