"""Stage 2, inject the 4 error types.

Reads base_sample.parquet (1500 rows) and emits 5 rows per base solution
(clean + type1..type4) => 7500 rows. One wrong-number is drawn per base solution
and SHARED by type1 (boxed only) and type3 (all occurrences), the only
difference between them is where the number changed. Skips are recorded, never
dropped (the 7500 count is preserved).

Outputs: error_dataset.parquet, corruption_log.csv, skipped_solutions.csv.
"""

from __future__ import annotations

import hashlib
import os
import random

import pandas as pd

from error_study import common, config


def _row_rng(row_id: str) -> random.Random:
    h = hashlib.md5(f"{config.SEED}:corrupt:{row_id}".encode()).hexdigest()[:8]
    return random.Random(int(h, 16))


def _menu_schedule(n: int) -> list[str]:
    """A shuffled 40/30/30 category schedule of length n (deterministic)."""
    d = config.MENU_DISTRIBUTION
    sched = (
        ["pm12"] * round(d["pm12"] * n)
        + ["digit_swap"] * round(d["digit_swap"] * n)
        + ["single_digit"] * round(d["single_digit"] * n)
    )
    while len(sched) < n:
        sched.append("pm12")
    sched = sched[:n]
    random.Random(config.SEED).shuffle(sched)
    return sched


_EMPTY = {
    "corrupted_solution": None,
    "corruption_applied": False,
    "judge_include": False,
    "skip_reason": "",
    "old_number": "",
    "new_number": "",
    "wrong_number_menu": "",
    "change_location_char": None,
    "change_location_pct": None,
    "intermediate_menu": "",
    "num_occurrences_replaced": None,
}


def _mk(base: dict, error_type: str, **fields) -> dict:
    r = dict(base)
    r["error_type"] = error_type
    out = dict(_EMPTY)
    out.update(fields)
    r.update(out)
    return r


def corrupt_one(base: dict, answer_cat: str, type2_cat: str) -> list[dict]:
    """Return the 5 rows (clean + type1..type4) for one base solution."""
    content = base["solution_text"]
    gt = base["ground_truth"]
    atype = base["answer_type"]
    is_aime = base["benchmark"] in config.AIME_BENCHMARKS
    qnums = common.question_numbers(base["question_text"])
    rng = _row_rng(base["row_id"])

    rows = []
    # clean control (unmodified)
    rows.append(
        _mk(
            base,
            "clean",
            corrupted_solution=content,
            corruption_applied=False,
            judge_include=True,
            skip_reason="control_clean",
        )
    )

    # shared wrong-number for type1 & type3
    w = common.corrupt_answer_value(gt, atype, is_aime, qnums, answer_cat, rng)

    # type1: boxed only
    if w is None:
        rows.append(_mk(base, "boxed_only", skip_reason=f"no_valid_wrong_number({atype})"))
    else:
        r1 = common.apply_type1(content, gt, w)
        if "skip_reason" in r1:
            rows.append(_mk(base, "boxed_only", skip_reason=r1["skip_reason"]))
        else:
            rows.append(
                _mk(
                    base,
                    "boxed_only",
                    corruption_applied=True,
                    judge_include=True,
                    wrong_number_menu=w["category"] + ("(fallback)" if w["fallback"] else ""),
                    **{
                        k: r1[k]
                        for k in (
                            "corrupted_solution",
                            "old_number",
                            "new_number",
                            "change_location_char",
                            "change_location_pct",
                        )
                    },
                )
            )

    # type2: intermediate slip (independent draw)
    r2 = common.apply_type2(content, gt, atype, qnums, type2_cat, rng)
    if "skip_reason" in r2:
        rows.append(_mk(base, "intermediate_error", skip_reason=r2["skip_reason"]))
    else:
        rows.append(
            _mk(
                base,
                "intermediate_error",
                corruption_applied=True,
                judge_include=True,
                **{
                    k: r2[k]
                    for k in (
                        "corrupted_solution",
                        "old_number",
                        "new_number",
                        "change_location_char",
                        "change_location_pct",
                        "intermediate_menu",
                        "wrong_number_menu",
                    )
                },
            )
        )

    # type3: hidden consistent (same wrong-number as type1)
    if w is None:
        rows.append(_mk(base, "consistent_error", skip_reason=f"no_valid_wrong_number({atype})"))
    else:
        r3 = common.apply_type3(content, gt, atype, w, qnums)
        if "skip_reason" in r3:
            rows.append(_mk(base, "consistent_error", skip_reason=r3["skip_reason"]))
        else:
            rows.append(
                _mk(
                    base,
                    "consistent_error",
                    corruption_applied=True,
                    judge_include=True,
                    wrong_number_menu=w["category"] + ("(fallback)" if w["fallback"] else ""),
                    **{
                        k: r3[k]
                        for k in (
                            "corrupted_solution",
                            "old_number",
                            "new_number",
                            "change_location_char",
                            "change_location_pct",
                            "num_occurrences_replaced",
                        )
                    },
                )
            )

    # type4: truncate last 25%, keep boxed
    r4 = common.apply_type4(content)
    if "skip_reason" in r4:
        rows.append(_mk(base, "truncated", skip_reason=r4["skip_reason"]))
    else:
        rows.append(
            _mk(
                base,
                "truncated",
                corruption_applied=True,
                judge_include=True,
                **{k: r4[k] for k in ("corrupted_solution", "change_location_char", "change_location_pct")},
            )
        )
    return rows


def build(base_df: pd.DataFrame) -> pd.DataFrame:
    base_df = base_df.sort_values("row_id").reset_index(drop=True)
    sched = _menu_schedule(2 * len(base_df))
    out_rows = []
    for i, base in enumerate(base_df.to_dict(orient="records")):
        out_rows.extend(corrupt_one(base, sched[2 * i], sched[2 * i + 1]))
    df = pd.DataFrame(out_rows)
    # keep numeric metadata columns numeric (NA where not applicable)
    df["change_location_char"] = pd.array(df["change_location_char"], dtype="Int64")
    df["num_occurrences_replaced"] = pd.array(df["num_occurrences_replaced"], dtype="Int64")
    df["change_location_pct"] = pd.to_numeric(df["change_location_pct"], errors="coerce")
    return df


def main() -> None:
    base_path = os.path.join(config.OUTPUT_DIR, "base_sample.parquet")
    base_df = pd.read_parquet(base_path)
    df = build(base_df)

    err_path = os.path.join(config.OUTPUT_DIR, "error_dataset.parquet")
    df.to_parquet(err_path, index=False)
    err_csv = err_path.replace(".parquet", ".csv")
    df.to_csv(err_csv, index=False)  # QUOTE_MINIMAL: long/multiline LaTeX fields get quoted

    # corruption log (metadata only, no solution text), corruptions = type1..4
    log_cols = [
        "row_id",
        "group",
        "model",
        "benchmark",
        "task_id",
        "error_type",
        "answer_type",
        "solution_len_tokens",
        "solution_len_chars",
        "old_number",
        "new_number",
        "change_location_char",
        "change_location_pct",
        "wrong_number_menu",
        "intermediate_menu",
        "num_occurrences_replaced",
        "corruption_applied",
        "skip_reason",
    ]
    log = df[df["error_type"] != "clean"][log_cols]
    log.to_csv(os.path.join(config.OUTPUT_DIR, "corruption_log.csv"), index=False)

    skipped = log[log["corruption_applied"] == False]  # noqa: E712
    skipped[["row_id", "group", "task_id", "error_type", "answer_type", "skip_reason"]].to_csv(
        os.path.join(config.OUTPUT_DIR, "skipped_solutions.csv"), index=False
    )

    # summary
    print(f"error rows: {len(df)}  (expected {5 * len(base_df)})")
    print("\napplied vs skipped per error_type:")
    for et in ["clean", "boxed_only", "intermediate_error", "consistent_error", "truncated"]:
        sub = df[df["error_type"] == et]
        if et == "clean":
            print(f"  {et:6} n={len(sub):5} (control, unmodified, judged as-is)")
            continue
        applied = int((sub["corruption_applied"] == True).sum())  # noqa: E712
        print(f"  {et:6} n={len(sub):5} applied={applied:5} skipped={len(sub) - applied:5}")
    print("\nskip reasons:")
    print(skipped["skip_reason"].value_counts().to_dict())
    print("\nrealized wrong_number_menu (type1/type3 answer-draws + type2):")
    menu = df[df["wrong_number_menu"] != ""]["wrong_number_menu"]
    print(menu.value_counts().to_dict())
    print(f"\nwrote {err_path}")
    print(f"wrote {err_csv}")
    print(f"wrote corruption_log.csv ({len(log)} rows), skipped_solutions.csv ({len(skipped)} rows)")


if __name__ == "__main__":
    main()
