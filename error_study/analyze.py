"""Summarise judge results from the consolidated single file.

Reads judge_output.parquet (produced by consolidate.py, one row per judge
generation, carrying the input metadata + verdict + majority_verdict) and writes:

  cell_summary.csv: per (model, benchmark, error_type): #solution-variants,
                              per-gen yes/no/invalid, majority yes/no (cot_false), no-rate.
  per_question_summary.csv: per (benchmark, question, error_type): cot_false / yes.

Interpretation: for corruptions, majority "no" = the judge CAUGHT the injected
error (a true detection); for clean controls, majority "no" = a false positive.

Usage:  PYTHONPATH=/repo python -m error_study.analyze \
          [--input error_study/output/judge_demo_output.parquet]
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from error_study import config


def summarise(df: pd.DataFrame, outdir: str) -> None:
    if df.empty:
        print("empty judge_output")
        return

    # per-generation verdict counts per solution-variant
    g = df.assign(
        is_yes=(df.judge_verdict == "yes").astype(int),
        is_no=(df.judge_verdict == "no").astype(int),
        is_inv=(df.judge_verdict == "invalid").astype(int),
    )
    # aggregate per SOLUTION-VARIANT (row_id), NOT per question (task_id): one
    # question can carry up to 6 sampled solutions, each its own solution-variant.
    sv = (
        g.groupby(["model", "benchmark", "row_id", "error_type"])
        .agg(
            task_id=("task_id", "first"),
            n_gen=("gen_index", "count"),
            gen_yes=("is_yes", "sum"),
            gen_no=("is_no", "sum"),
            gen_invalid=("is_inv", "sum"),
            majority=("majority_verdict", "first"),
        )
        .reset_index()
    )
    sv["maj_no"] = (sv.majority == "no").astype(int)
    sv["maj_yes"] = (sv.majority == "yes").astype(int)
    sv["maj_invalid"] = (sv.majority == "invalid").astype(int)
    # VETO (legacy alias, kept for continuity): >=1 'no' sample. == veto_any below.
    sv["veto_no"] = (sv.gen_no >= 1).astype(int)

    # any / majority / all threshold family (judge APPROVAL threshold)
    # correct = judge boxed "yes"; veto = judge boxed "no". Over this solution-
    # variant's n_gen judge samples (y=gen_yes, n=gen_no):
    #   any      = >=1 sample said it,
    #   majority = strict > half  (yes>no for correct; no>yes for veto),
    #   all      = every one of the n_gen samples said it.
    # Each solution-variant is ONE generation judged n_gen times (no pass@k here).
    sv["correct_any"] = (sv.gen_yes >= 1).astype(int)
    sv["correct_maj"] = (sv.gen_yes > sv.gen_no).astype(int)
    sv["correct_all"] = (sv.gen_yes == sv.n_gen).astype(int)
    sv["veto_any"] = (sv.gen_no >= 1).astype(int)
    sv["veto_maj"] = (sv.gen_no > sv.gen_yes).astype(int)
    sv["veto_all"] = (sv.gen_no == sv.n_gen).astype(int)
    thresholds = ["correct_any", "correct_maj", "correct_all", "veto_any", "veto_maj", "veto_all"]

    cell = (
        sv.groupby(["model", "benchmark", "error_type"])
        .agg(
            n_solutions=("row_id", "count"),
            gen_yes=("gen_yes", "sum"),
            gen_no=("gen_no", "sum"),
            gen_invalid=("gen_invalid", "sum"),
            maj_yes=("maj_yes", "sum"),
            cot_false=("maj_no", "sum"),
            maj_invalid=("maj_invalid", "sum"),
            veto_no=("veto_no", "sum"),
            **{t: (t, "sum") for t in thresholds},
        )
        .reset_index()
    )
    cell["cot_false_%"] = (100 * cell.cot_false / cell.n_solutions).round(1)
    cell["veto_%"] = (100 * cell.veto_no / cell.n_solutions).round(1)
    for t in thresholds:
        cell[f"{t}_%"] = (100 * cell[t] / cell.n_solutions).round(1)
    cell.to_csv(os.path.join(outdir, "cell_summary.csv"), index=False)

    perq = (
        sv.groupby(["benchmark", "task_id", "error_type"])
        .agg(
            n_solutions=("row_id", "count"),
            cot_false=("maj_no", "sum"),
            maj_yes=("maj_yes", "sum"),
            veto_no=("veto_no", "sum"),
            **{t: (t, "sum") for t in thresholds},
        )
        .reset_index()
        .rename(columns={"task_id": "question"})
    )
    perq.to_csv(os.path.join(outdir, "per_question_summary.csv"), index=False)

    pooled = (
        sv.groupby(["benchmark", "error_type"])
        .agg(
            n=("row_id", "count"),
            cot_false=("maj_no", "sum"),
            veto=("veto_no", "sum"),
            **{t: (t, "sum") for t in thresholds},
        )
        .reset_index()
    )
    pooled["cot_false_%"] = (100 * pooled.cot_false / pooled.n).round(1)
    pooled["veto_%"] = (100 * pooled.veto / pooled.n).round(1)
    for t in thresholds:
        pooled[f"{t}_%"] = (100 * pooled[t] / pooled.n).round(1)
    order = [c for c in config.ERROR_TYPES if c in set(pooled.error_type)]

    def _pivot(col: str) -> str:
        return pooled.pivot_table(index="benchmark", columns="error_type", values=col, fill_value=0)[order].to_string()

    print(f"solution-variants judged: {len(sv)}   generations: {len(df)}")
    print("\n=== cot_false (majority-'no') per benchmark x error_type ===")
    print("(clean: 'no' = false positive; corruptions: 'no' = error CAUGHT)")
    print(_pivot("cot_false"))
    print("\n### CORRECT rate %  (judge approves = boxed 'yes'; threshold = of the 3 judge samples) ###")
    for t, lbl in [
        ("correct_any", "any  (>=1 'yes')"),
        ("correct_maj", "majority (>half 'yes')"),
        ("correct_all", "all  (unanimous 'yes')"),
    ]:
        print(f"\n-- correct / {lbl} --")
        print(_pivot(f"{t}_%"))
    print("\n### VETO rate %  (judge flags = boxed 'no'; corruptions: caught / clean: false-positive) ###")
    for t, lbl in [
        ("veto_any", "any  (>=1 'no')"),
        ("veto_maj", "majority (>half 'no')"),
        ("veto_all", "all  (unanimous 'no')"),
    ]:
        print(f"\n-- veto / {lbl} --")
        print(_pivot(f"{t}_%"))
    print(f"\nwrote cell_summary.csv, per_question_summary.csv -> {outdir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(config.OUTPUT_DIR, "judge_demo_output.parquet"))
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    df = pd.read_parquet(a.input)
    summarise(df, a.outdir or os.path.dirname(os.path.abspath(a.input)))


if __name__ == "__main__":
    main()
