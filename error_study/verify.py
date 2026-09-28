"""Verification: assert every invariant the study promises.

Run after sample.py + corrupt.py + build_judge_inputs.py. Also checks determinism
(rebuild in-memory with the same seed and compare) when called with --determinism.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from error_study import common, config

OUT = config.OUTPUT_DIR
_fails = []


def check(cond: bool, msg: str) -> None:
    print(("  ok  " if cond else "FAIL  ") + msg)
    if not cond:
        _fails.append(msg)


def main() -> None:
    base = pd.read_parquet(os.path.join(OUT, "base_sample.parquet"))
    err = pd.read_parquet(os.path.join(OUT, "error_dataset.parquet"))

    print("== base_sample ==")
    n_groups = len(config.MODELS) * len(config.BENCHMARKS)
    check(len(base) == config.PER_GROUP * n_groups, f"base rows = {len(base)} (== {config.PER_GROUP * n_groups})")
    gsz = base.groupby("group").size()
    check(
        (gsz == config.PER_GROUP).all(), f"every group has {config.PER_GROUP} rows (min={gsz.min()}, max={gsz.max()})"
    )
    check((base["solution_len_tokens"] > config.MIN_TOKENS).all(), f"all solutions > {config.MIN_TOKENS} tokens")
    check(base["state"].eq("think").all(), "state == think for all rows")
    check(base["solution_text"].str.contains(r"\\boxed", regex=True).all(), "every sampled solution has a \\boxed")
    per_q = base.groupby(["group", "task_id"]).size()
    check(per_q.max() <= config.MAX_SOLS_PER_Q, f"<= {config.MAX_SOLS_PER_Q} solutions/question (max={per_q.max()})")
    check(per_q.min() >= config.MIN_SOLS_PER_Q, f">= {config.MIN_SOLS_PER_Q} solution/question (min={per_q.min()})")
    check(base["row_id"].is_unique, "row_id unique")

    print("\n== correctness (re-grade 200 sampled base solutions) ==")
    import evalhub.benchmarks  # noqa: F401
    from evalhub.benchmarks.math.verifier import grade_answer
    from evalhub.benchmarks.registry import DATASET_MAP

    dss = {}
    ok = 0
    sample = base.sample(min(200, len(base)), random_state=0)
    for r in sample.to_dict(orient="records"):
        b = r["benchmark"]
        if b not in dss:
            dss[b] = DATASET_MAP[b](b)
            dss[b].load_tasks()
        ext = dss[b].extract_solution(r["task_id"], r["solution_text"])
        if grade_answer(ext, r["ground_truth"]):
            ok += 1
    check(ok == len(sample), f"re-graded correct: {ok}/{len(sample)}")

    print("\n== error_dataset ==")
    check(len(err) == 5 * len(base), f"error rows = {len(err)} (== 5*{len(base)})")
    per_row = err.groupby("row_id")["error_type"].nunique()
    check((per_row == 5).all(), "every base row has exactly 5 variants (clean+4)")
    check(
        set(err["error_type"].unique()) == set(config.ERROR_TYPES),
        "error_type set == {clean,boxed_only,intermediate_error,consistent_error,truncated}",
    )

    # clean is the unmodified control
    m = err.merge(base[["row_id", "solution_text"]], on="row_id", suffixes=("", "_base"))
    clean = m[m["error_type"] == "clean"]
    check((clean["corrupted_solution"] == clean["solution_text_base"]).all(), "clean == original solution")

    # type1 & type3 share the same wrong number (where both applied)
    piv = err[err["error_type"].isin(["boxed_only", "consistent_error"]) & (err["corruption_applied"])].pivot_table(
        index="row_id", columns="error_type", values="new_number", aggfunc="first"
    )
    both = piv.dropna()
    check(
        (both["boxed_only"] == both["consistent_error"]).all(),
        f"type1/type3 same wrong number ({len(both)} rows, 0 mismatch)",
    )

    print("\n== applied-corruption checks ==")
    ap = err[err["corruption_applied"] == True].copy()  # noqa: E712
    # new_number != ground_truth value; new_number not a question number (int rows)
    bad_eq = bad_q = bad_special = 0
    for r in ap.to_dict(orient="records"):
        qn = common.question_numbers(r["question_text"])
        new = str(r["new_number"])
        if r["error_type"] in ("boxed_only", "consistent_error") and r["answer_type"] == "int":
            if new == str(r["ground_truth"]):
                bad_eq += 1
            if new.lstrip("-").isdigit() and int(new) in qn:
                bad_q += 1
        if r["error_type"] == "intermediate_error" and new.lstrip("-").isdigit():
            if int(new) in (0, 1):
                bad_special += 1
            if int(new) in qn:
                bad_q += 1
    check(bad_eq == 0, f"no applied type1/3 equals the real answer ({bad_eq})")
    check(bad_q == 0, f"no applied wrong-number equals a question number ({bad_q})")
    check(bad_special == 0, f"no type2 intermediate is 0/1 ({bad_special})")

    # type1 changes ONLY boxed; type3 changes body too (int answers, boundary-safe)
    t = err.merge(base[["row_id", "ground_truth"]], on="row_id", suffixes=("", "_b"))
    int_ap = t[(t["answer_type"] == "int") & (t["corruption_applied"])]

    def occ(row):
        return str(row["corrupted_solution"]).count(str(row["ground_truth"]))

    t1 = int_ap[int_ap["error_type"] == "boxed_only"]
    t3 = int_ap[int_ap["error_type"] == "consistent_error"]
    # type3 applied rows skip embedded-answer cases, so EVERY applied int row must
    # have 0 remaining answer-occurrences (strict per-row, not a mean).
    t3occ = t3.apply(occ, axis=1)
    check((t3occ == 0).all(), f"type3: 0 remaining answer-occurrences in every applied int row (max={t3occ.max()})")
    check(t1.apply(occ, axis=1).mean() >= 1.0, f"type1 keeps body answer (mean occ {t1.apply(occ, axis=1).mean():.2f})")

    # type4 ~75% length and keeps a boxed  (err already carries solution_len_chars)
    t4 = err[err["error_type"] == "truncated"]
    ratio = (t4["corrupted_solution"].str.len() / t4["solution_len_chars"]).mean()
    check(0.70 <= ratio <= 0.85, f"type4 mean length ratio {ratio:.2f} (~0.75)")
    check(
        t4["corrupted_solution"].apply(lambda s: common.find_last_boxed_span(s) is not None).all(),
        "type4 keeps a \\boxed in every row",
    )

    print("\n== logs ==")
    log = pd.read_csv(os.path.join(OUT, "corruption_log.csv"))
    check(len(log) == 4 * len(base), f"corruption_log rows = {len(log)} (== 4*{len(base)})")
    skipped = pd.read_csv(os.path.join(OUT, "skipped_solutions.csv"))
    n_skip = int((err["error_type"] != "clean").sum() - (err["corruption_applied"] == True).sum())  # noqa: E712
    check(len(skipped) == n_skip, f"skipped_solutions rows = {len(skipped)} (== {n_skip} skipped corruptions)")

    # deduped realized menu distribution (answer-draw once per row + type2)
    ans = err[(err["error_type"] == "boxed_only") & (err["wrong_number_menu"] != "")]["wrong_number_menu"]
    t2 = err[(err["error_type"] == "intermediate_error") & (err["wrong_number_menu"] != "")]["wrong_number_menu"]
    draws = pd.concat([ans, t2])
    prim = draws.str.replace("(fallback)", "", regex=False)
    dist = (prim.value_counts(normalize=True) * 100).round(1).to_dict()
    fb = 100 * draws.str.contains("fallback").mean()
    print(f"  realized menu (deduped, {len(draws)} draws): {dist}  fallback={fb:.1f}%")

    print("\n== RESULT ==")
    if _fails:
        print(f"{len(_fails)} FAILURES:")
        for f in _fails:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


def check_determinism() -> None:
    """Rebuild base + error in-memory with the same seed and compare to the saved
    files (byte-for-byte on the content-bearing columns)."""
    print("\n== determinism (rebuild in-memory, same seed) ==")
    from error_study import corrupt, sample

    saved_base = pd.read_parquet(os.path.join(OUT, "base_sample.parquet")).sort_values("row_id").reset_index(drop=True)
    reb_base = sample.build().sort_values("row_id").reset_index(drop=True)
    check(saved_base.equals(reb_base), f"base_sample deterministic ({len(saved_base)} rows)")
    saved_err = pd.read_parquet(os.path.join(OUT, "error_dataset.parquet"))
    reb_err = corrupt.build(saved_base)
    key = [
        "row_id",
        "error_type",
        "corrupted_solution",
        "old_number",
        "new_number",
        "wrong_number_menu",
        "change_location_char",
    ]
    a = saved_err.sort_values(["row_id", "error_type"]).reset_index(drop=True)[key]
    b = reb_err.sort_values(["row_id", "error_type"]).reset_index(drop=True)[key]
    check(a.equals(b), f"error_dataset deterministic ({len(a)} rows)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--determinism", action="store_true", help="also rebuild base+error in-memory and compare (slower)")
    args = ap.parse_args()
    if args.determinism:
        check_determinism()
    main()
