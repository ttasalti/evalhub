"""Acceptance of the same 720-solution error-injection panel under three judges, per benchmark and condition
(the panel CSV behind Section 5.1 and the McNemar and bootstrap scripts, plus a per-benchmark summary).

The V4-Flash and Qwen3.6 panel verdicts are read from report_error_tasks.csv in analysis/data. The R1-distill
verdicts come from the judge cache judge_r1_8b_consolidated.parquet, which is built from
error_study/output/judge_r1_8b when absent; the Qwen3.6 cache judge_qwen35b_output.parquet is used for a
consistency check only and skipped when absent. Neither the caches nor the error-study output are part of the
repository. The script writes error_injection_panel_3judges.csv to analysis/data and
error_injection_by_language.txt to analysis/tables.
"""

from __future__ import annotations

import sys

import pandas as pd
from common import CACHE_DIR, DATA_DIR, ERROR_STUDY_OUTPUT, REPO_ROOT, TAB_DIR, data, ensure_output_dirs, require

CONDITIONS = ["clean", "intermediate_error", "truncated", "boxed_only", "consistent_error"]
CONDITION_LABEL = {
    "clean": "clean control",
    "intermediate_error": "intermediate numeric error",
    "truncated": "truncation",
    "boxed_only": "final-answer error",
    "consistent_error": "consistent final-answer error",
}
BENCHMARKS = ["aime2026", "aime2026_tr", "aime2026_pt", "tubitak_math2026", "pt_exams_math"]
BENCHMARK_LABEL = {
    "aime2026": "EN",
    "aime2026_tr": "TR",
    "aime2026_pt": "PT",
    "tubitak_math2026": "TUB",
    "pt_exams_math": "PTex",
}
JUDGES = ["Qwen3.6", "V4-Flash", "R1-distill"]
JUDGE_CACHE_COLUMNS = [
    "row_id",
    "error_type",
    "model",
    "benchmark",
    "task_id",
    "gen_index",
    "judge_verdict",
    "old_number",
    "new_number",
    "wrong_number_menu",
    "change_location_pct",
]
R1_JUDGE_META = {"judge_model": "DeepSeek-R1-0528-Qwen3-8B", "judge_state": "think", "judge_max_tokens": 20480}

QWEN_CACHE = CACHE_DIR / "judge_qwen35b_output.parquet"
R1_CACHE_BASE = CACHE_DIR / "judge_r1_8b_consolidated"
R1_CACHE = R1_CACHE_BASE.with_suffix(".parquet")
R1_RAW_DIR = ERROR_STUDY_OUTPUT / "judge_r1_8b"


def solution_variants(judge_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-generation judge verdicts into one row per (solution, condition) with n_gen == 3."""
    from error_study.report_error import _solution_variants

    variants = _solution_variants(judge_rows)
    variants = variants[variants.n_gen == 3]
    return variants.rename(columns={"row_id": "solution_id", "error_type": "condition"})


def check_qwen_cache(panel: pd.DataFrame, panel_ids: set) -> None:
    """Pass the Qwen3.6 cache through the same aggregation and compare with the published verdicts."""
    if not QWEN_CACHE.is_file():
        print("Qwen3.6 cache not found, consistency check skipped:", QWEN_CACHE)
        return
    qwen_variants = solution_variants(pd.read_parquet(QWEN_CACHE, columns=JUDGE_CACHE_COLUMNS))
    qwen_variants = qwen_variants[qwen_variants.solution_id.isin(panel_ids)]
    qwen_csv = panel[panel.judge_short == "Qwen3.6"][["solution_id", "condition", "correct_maj", "n_yes", "n_no"]]
    merged = qwen_variants.merge(qwen_csv, on=["solution_id", "condition"], suffixes=("_pipe", "_csv"))
    maj_agreement = 100 * (merged.correct_maj_pipe == merged.correct_maj_csv).mean()
    yes_agreement = 100 * (merged.yes == merged.n_yes_csv).mean()
    print(
        f"Qwen3.6 check: panel variants {len(merged)}/3600 | correct_maj agreement {maj_agreement:.2f}% "
        f"| n_yes agreement {yes_agreement:.2f}%"
    )


def load_r1_panel(panel_ids: set) -> pd.DataFrame:
    """Consolidate the raw R1-distill judge output when needed and reduce it to the panel."""
    if not R1_CACHE.is_file():
        from error_study.consolidate import consolidate

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        consolidate(str(R1_RAW_DIR), str(R1_CACHE_BASE), judge_meta=R1_JUDGE_META)
    r1_rows = pd.read_parquet(R1_CACHE, columns=JUDGE_CACHE_COLUMNS)
    r1 = solution_variants(r1_rows)
    print(f"R1 variants (n_gen == 3): {len(r1)} / 7115 expected")
    r1 = r1[r1.solution_id.isin(panel_ids)].copy()
    r1["judge_short"] = "R1-distill"
    print(f"R1 panel variants: {len(r1)}/3600  per condition: {r1.condition.value_counts().to_dict()}")
    assert len(r1) == 3600, "R1 panel incomplete"
    return r1


def acceptance_table(panel_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        for judge in JUDGES:
            s = panel_all[(panel_all.condition == condition) & (panel_all.judge_short == judge)]
            per_benchmark = [100 * s[s.benchmark == b].correct_maj.mean() for b in BENCHMARKS]
            rows.append([CONDITION_LABEL[condition], judge] + per_benchmark + [100 * s.correct_maj.mean(), len(s)])
    columns = ["condition", "judge"] + [BENCHMARK_LABEL[b] for b in BENCHMARKS] + ["pooled", "n"]
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    if not R1_CACHE.is_file():
        require(R1_RAW_DIR, what="the R1-distill judge output (error_study/output/judge_r1_8b)")
    sys.path.insert(0, str(REPO_ROOT))
    ensure_output_dirs()

    error_tasks = pd.read_csv(data("report_error_tasks.csv"))
    panel = error_tasks[error_tasks.panel == True]  # noqa: E712
    panel_ids = set(panel.solution_id)

    check_qwen_cache(panel, panel_ids)
    r1 = load_r1_panel(panel_ids)

    columns = ["judge_short", "model", "benchmark", "condition", "solution_id", "correct_maj"]
    published = panel[panel.judge_short.isin(["V4-Flash", "Qwen3.6"])][columns]
    panel_all = pd.concat([published, r1[columns]], ignore_index=True)
    panel_all.to_csv(DATA_DIR / "error_injection_panel_3judges.csv", index=False)

    table = acceptance_table(panel_all)
    pd.set_option("display.width", 200)
    text = table.to_string(index=False, float_format=lambda x: f"{x:.1f}")
    n_per_benchmark = int(
        (
            (panel_all.judge_short == "Qwen3.6")
            & (panel_all.condition == "clean")
            & (panel_all.benchmark == "aime2026")
        ).sum()
    )
    print(f"\nACCEPTANCE RATE (%), majority-correct, same 720 panel; n={n_per_benchmark} per benchmark, 720 pooled\n")
    print(text)
    (TAB_DIR / "error_injection_by_language.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
