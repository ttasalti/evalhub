"""Acceptance rates under the any/majority/all verification rules for the three judges on the 720-solution
error-injection panel (Table tab:error-rules, tab_error_rules.tex).

The V4-Flash and Qwen3.6 panel verdicts come from report_error_tasks.csv in analysis/data. The R1-distill
verdicts come from the judge cache judge_r1_8b_consolidated.parquet, which panel_three_judges.py writes from
the error-study output and which is not part of the repository; the script stops with a message when the
cache is absent. The table is written to analysis/tables/tab_error_rules.tex.
"""

from __future__ import annotations

import sys

import pandas as pd
from common import CACHE_DIR, REPO_ROOT, TAB_DIR, data, ensure_output_dirs, require

JUDGES = ["V4-Flash", "Qwen3.6", "R1-distill"]
CONDITIONS = ["clean", "intermediate_error", "truncated", "boxed_only", "consistent_error"]
CONDITION_LABEL_TEX = {
    "clean": "Clean",
    "intermediate_error": "Intermediate numeric error",
    "truncated": "Truncation error",
    "boxed_only": "Final-answer error",
    "consistent_error": "Consistent final-answer error",
}
RULES = ["correct_any", "correct_maj", "correct_all"]
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
R1_CACHE = CACHE_DIR / "judge_r1_8b_consolidated.parquet"


def load_r1_panel(panel_ids: set) -> pd.DataFrame:
    from error_study.report_error import _solution_variants

    r1_rows = pd.read_parquet(R1_CACHE, columns=JUDGE_CACHE_COLUMNS)
    r1 = _solution_variants(r1_rows)
    r1 = r1[(r1.n_gen == 3) & (r1.row_id.isin(panel_ids))]
    r1 = r1.rename(columns={"row_id": "solution_id", "error_type": "condition"})
    r1["judge_short"] = "R1-distill"
    assert len(r1) == 3600
    return r1


def table_lines(panel_all: pd.DataFrame) -> list[str]:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Variant & any & majority & all \\",
        r"\midrule",
    ]
    for judge_index, judge in enumerate(JUDGES):
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{judge}}}}} \\")
        for condition in CONDITIONS:
            s = panel_all[(panel_all.judge_short == judge) & (panel_all.condition == condition)]
            assert len(s) == 720, (judge, condition, len(s))
            rates = " & ".join(f"{100 * s[rule].mean():.1f}" for rule in RULES)
            lines.append(f"{CONDITION_LABEL_TEX[condition]} & {rates} \\\\")
            if condition == "clean":
                lines.append(r"\cmidrule(lr){1-4}")
        if judge_index < len(JUDGES) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Acceptance rates (\%) under the three verification strategies of the original CoT-Pass@$k$ judge: "
        r"a variant is accepted when at least one of its three judge generations approves it (any-correct), when more "
        r"approve than reject (majority-correct), or when all three do (all-correct). The majority-correct column is "
        r"the one plotted in Figure~\ref{fig:error-injection}. 720 original solutions per error type for "
        r"\crnew{each judge; R1-distill is the judge the metric specifies}.}",
        r"\label{tab:error-rules}",
        r"\end{table}",
    ]
    return lines


def main() -> None:
    require(R1_CACHE, what="the R1-distill judge cache (run panel_three_judges.py first)")
    sys.path.insert(0, str(REPO_ROOT))
    ensure_output_dirs()

    error_tasks = pd.read_csv(data("report_error_tasks.csv"))
    panel = error_tasks[error_tasks.panel == True]  # noqa: E712
    r1 = load_r1_panel(set(panel.solution_id))

    columns = ["judge_short", "condition", "solution_id"] + RULES
    published = panel[panel.judge_short.isin(["V4-Flash", "Qwen3.6"])][columns]
    panel_all = pd.concat([published, r1[columns]], ignore_index=True)

    lines = table_lines(panel_all)
    (TAB_DIR / "tab_error_rules.tex").write_text("\n".join(lines) + "\n")
    for judge in JUDGES:
        s = panel_all[panel_all.judge_short == judge]
        print(judge, " | ".join(f"{c}: {100 * s[s.condition == c].correct_maj.mean():.1f}" for c in CONDITIONS))
    print("\n".join(lines[7:]))


if __name__ == "__main__":
    main()
