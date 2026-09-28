"""Task- and question-grained error-study report (the error analog of report_tasks.csv).

Reads the consolidated ``judge_output.parquet`` and emits two CSVs that expose the
any / majority / all judge-approval-threshold family at two grains:

  report_error_tasks.csv: one row per JUDGED SOLUTION-VARIANT (row_id). The raw
                               3-sample judge counts (yes/no/invalid) are KEPT (the
                               downstream process needs the per-judge-generation
                               breakdown) alongside the 6 threshold booleans and the
                               corruption metadata (old->new number, menu, location).
  report_error_questions.csv: one row per (model, benchmark, error_type, question):
                               that question's <=6 solution-variants aggregated into
                               per-threshold counts + rates.

correct = judge boxed "yes"; veto = judge boxed "no". Over a solution-variant's
n_gen judge samples (y=yes, n=no): any = >=1 sample, majority = strict >half
(y>n / n>y), all = every one of the n_gen samples. A solution-variant is ONE
generation judged n_gen times, so there is no pass@k dimension here.

Usage:  PYTHONPATH=/repo python -m error_study.report_error \
          [--input error_study/output/judge_output.parquet]
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from error_study import config

# corruption types that rewrite the final boxed answer (so it is no longer correct)
ANSWER_CHANGING = {"boxed_only", "consistent_error"}
THRESH = ["correct_any", "correct_maj", "correct_all", "veto_any", "veto_maj", "veto_all"]


def _solution_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-generation judge rows into one row per judged solution-variant."""
    g = df.assign(
        is_yes=(df.judge_verdict == "yes").astype(int),
        is_no=(df.judge_verdict == "no").astype(int),
        is_inv=(df.judge_verdict == "invalid").astype(int),
    )
    sv = (
        g.groupby(["model", "benchmark", "error_type", "row_id"])
        .agg(
            task_id=("task_id", "first"),
            old_number=("old_number", "first"),
            new_number=("new_number", "first"),
            wrong_number_menu=("wrong_number_menu", "first"),
            change_location_pct=("change_location_pct", "first"),
            n_gen=("gen_index", "count"),
            yes=("is_yes", "sum"),
            no=("is_no", "sum"),
            invalid=("is_inv", "sum"),
        )
        .reset_index()
    )
    sv["correct_any"] = (sv.yes >= 1).astype(int)
    sv["correct_maj"] = (sv.yes > sv.no).astype(int)
    sv["correct_all"] = (sv.yes == sv.n_gen).astype(int)
    sv["veto_any"] = (sv.no >= 1).astype(int)
    sv["veto_maj"] = (sv.no > sv.yes).astype(int)
    sv["veto_all"] = (sv.no == sv.n_gen).astype(int)
    sv["answer_still_correct"] = ~sv.error_type.isin(ANSWER_CHANGING)
    return sv


def build(df: pd.DataFrame, outdir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sv = _solution_variants(df)

    tasks_cols = [
        "model",
        "benchmark",
        "error_type",
        "row_id",
        "task_id",
        "old_number",
        "new_number",
        "wrong_number_menu",
        "change_location_pct",
        "n_gen",
        "yes",
        "no",
        "invalid",
        *THRESH,
        "answer_still_correct",
    ]
    tasks = sv[tasks_cols].sort_values(["model", "benchmark", "error_type", "task_id", "row_id"]).reset_index(drop=True)
    tasks.to_csv(os.path.join(outdir, "report_error_tasks.csv"), index=False)

    q = (
        sv.groupby(["model", "benchmark", "error_type", "task_id"])
        .agg(
            n_solutions=("row_id", "count"),
            sum_yes=("yes", "sum"),
            sum_no=("no", "sum"),
            sum_invalid=("invalid", "sum"),
            **{t: (t, "sum") for t in THRESH},
        )
        .reset_index()
    )
    for t in THRESH:
        q[f"{t}_%"] = (100 * q[t] / q.n_solutions).round(1)
    q = q.sort_values(["model", "benchmark", "error_type", "task_id"]).reset_index(drop=True)
    q.to_csv(os.path.join(outdir, "report_error_questions.csv"), index=False)

    print(f"report_error: {len(tasks)} solution-variants -> report_error_tasks.csv")
    print(f"report_error: {len(q)} (model,benchmark,error_type,question) -> report_error_questions.csv")
    return tasks, q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(config.OUTPUT_DIR, "judge_output.parquet"))
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    df = pd.read_parquet(a.input)
    build(df, a.outdir or os.path.dirname(os.path.abspath(a.input)))


if __name__ == "__main__":
    main()
