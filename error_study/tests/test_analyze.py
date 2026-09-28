"""Regression test for the analyze aggregation key.

A question can carry up to 6 sampled solution-variants; the summary must aggregate
per solution-variant (row_id), NOT per question (task_id), otherwise multi-solution
questions collapse to one verdict and every headline number is wrong. The demo
masked this because each question had exactly one sampled solution.
"""

from __future__ import annotations

import pandas as pd

from error_study.analyze import summarise


def _gen_rows(rid, task_id, error_type, majority, verdicts):
    return [
        {
            "model": "M",
            "benchmark": "aime2026",
            "row_id": rid,
            "task_id": task_id,
            "error_type": error_type,
            "gen_index": i,
            "judge_verdict": v,
            "majority_verdict": majority,
        }
        for i, v in enumerate(verdicts)
    ]


def test_analyze_aggregates_per_solution_variant_not_question(tmp_path):
    # ONE question "T/1" with TWO sampled solutions: g0 caught (majority no),
    # g1 missed (majority yes). True: 2 solutions, 1 caught -> 50%.
    rows = _gen_rows("M|aime2026|T/1|g0", "T/1", "boxed_only", "no", ["no", "no", "yes"]) + _gen_rows(
        "M|aime2026|T/1|g1", "T/1", "boxed_only", "yes", ["yes", "yes", "no"]
    )
    summarise(pd.DataFrame(rows), str(tmp_path))
    cell = pd.read_csv(tmp_path / "cell_summary.csv").iloc[0]
    assert cell.n_solutions == 2  # both solution-variants counted (not collapsed to 1)
    assert cell.cot_false == 1  # exactly one caught (majority 'no')
    assert cell["cot_false_%"] == 50.0
    # veto = >=1 'no' generation: g0 has 2 'no', g1 has 1 'no' -> both vetoed.
    assert cell.veto_no == 2
    assert cell["veto_%"] == 100.0

    # any/majority/all threshold family (of the 3 judge samples):
    #   g0 y=1,n=2 -> correct any only;  veto any+maj.
    #   g1 y=2,n=1 -> correct any+maj;   veto any only.
    assert cell.correct_any == 2 and cell["correct_any_%"] == 100.0
    assert cell.correct_maj == 1 and cell["correct_maj_%"] == 50.0
    assert cell.correct_all == 0 and cell["correct_all_%"] == 0.0
    assert cell.veto_any == 2 and cell["veto_any_%"] == 100.0
    assert cell.veto_maj == 1 and cell["veto_maj_%"] == 50.0  # == cot_false (majority 'no')
    assert cell.veto_all == 0 and cell["veto_all_%"] == 0.0

    perq = pd.read_csv(tmp_path / "per_question_summary.csv").iloc[0]
    assert perq.n_solutions == 2 and perq.cot_false == 1 and perq.veto_no == 2
    assert perq.correct_maj == 1 and perq.veto_maj == 1
