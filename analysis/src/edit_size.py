"""Acceptance of injected errors by the kind and the size of the numeric edit (Table tab:edit-size,
tab_edit_size.tex, and the edit-size breakdown in the appendix).

Two files from analysis/data go in: corruption_log.csv (6000 rows = 1500 solutions x 4 conditions; the clean
control is not logged) and error_injection_panel_3judges.csv (720 solutions x 5 conditions x 3 judges, with
correct_maj). Out come edit_size.txt and tab_edit_size.tex in analysis/tables.

The two tables are joined on (solution_id, condition); the 720 panel solutions are a subset of the 1500 logged
ones. Edit kind is the menu label with any "(fallback)" suffix removed (a fallback means the requested menu
could not be applied and the edit fell back to this kind, so the applied kind is the label itself):
pm12 = +-1 or +-2, single_digit = one digit changed, digit_swap = two digits swapped.
Relative size = |new - old| / |old| (absolute difference when old = 0), binned <1%, 1-10%, 10-100%, >100%.
Acceptance = majority-correct rate (%) with a 95% Wilson interval; n is printed for every cell.
"""

from __future__ import annotations

import math

import pandas as pd
from common import TAB_DIR, data, ensure_output_dirs

JUDGES = ["V4-Flash", "Qwen3.6", "R1-distill"]
CONDITIONS = [
    ("intermediate_error", "Intermediate numeric error", "intermediate_menu"),
    ("boxed_only", "Final-answer error", "wrong_number_menu"),
    ("consistent_error", "Consistent final-answer error", "wrong_number_menu"),
]
EDIT_KIND_LABEL = {
    "pm12": "$\\pm1$ or $\\pm2$",
    "single_digit": "one digit changed",
    "digit_swap": "two digits swapped",
}
EDIT_KINDS = ["$\\pm1$ or $\\pm2$", "one digit changed", "two digits swapped"]
SIZE_BINS = ["$<$1\\%", "1--10\\%", "10--100\\%", "$>$100\\%"]
POSITION_BINS = ["40--50\\%", "50--60\\%", "60--70\\%"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def edit_kind(row: pd.Series, column: str) -> str:
    value = str(row[column]).replace("(fallback)", "")
    return EDIT_KIND_LABEL.get(value, value)


def size_bin(row: pd.Series) -> str:
    old, new = row.old_number, row.new_number
    rel = abs(new - old) / abs(old) if old != 0 else abs(new - old)
    if rel < 0.01:
        return "$<$1\\%"
    if rel < 0.1:
        return "1--10\\%"
    if rel < 1:
        return "10--100\\%"
    return "$>$100\\%"


def load_panel_edits() -> pd.DataFrame:
    log = pd.read_csv(data("corruption_log.csv"))
    panel = pd.read_csv(data("error_injection_panel_3judges.csv"))
    assert len(log) == 6000 and len(panel) == 10800
    merged = panel.merge(log, on=["solution_id", "condition"], how="left", suffixes=("", "_log"))
    sub = merged[merged.condition.isin([c for c, _, _ in CONDITIONS])].copy()
    assert sub.corruption_applied.notna().all() and sub.corruption_applied.all(), (
        "panel variants must all be applied edits"
    )
    assert len(sub) == 720 * 3 * 3
    menu_column = sub.condition.map({c: col for c, _, col in CONDITIONS})
    sub["etype"] = [edit_kind(row, col) for (_, row), col in zip(sub.iterrows(), menu_column, strict=False)]
    sub["relbin"] = sub.apply(size_bin, axis=1)
    sub["posbin"] = pd.cut(sub.edit_position_pct, [0, 50, 60, 70.0001], labels=POSITION_BINS)
    return sub


def block(sub: pd.DataFrame, out: list[str], key: str, levels: list[str], title: str) -> list:
    """Append one acceptance block (condition x level x judge) to out and return its cells for the LaTeX table."""
    out.append(f"\n=== {title} ===")
    rows = []
    for condition, condition_name, _ in CONDITIONS:
        for level in levels:
            cells = []
            for judge in JUDGES:
                s = sub[(sub.condition == condition) & (sub[key] == level) & (sub.judge_short == judge)]
                n = len(s)
                k = int(s.correct_maj.sum())
                low, high = wilson(k, n)
                cells.append((100 * k / n if n else float("nan"), low, high, n))
            rows.append((condition_name, level, cells))
            out.append(
                f"{condition_name:32s} {level:22s} "
                + "  ".join(
                    f"{j}: {p:5.1f} [{lo:4.1f},{hi:5.1f}] n={n}"
                    for j, (p, lo, hi, n) in zip(JUDGES, cells, strict=False)
                )
            )
    return rows


def text_report(sub: pd.DataFrame) -> tuple[str, list, list]:
    out: list[str] = []
    rows_kind = block(sub, out, "etype", EDIT_KINDS, "A) acceptance % x edit kind x condition x judge (n = solutions)")
    rows_size = block(sub, out, "relbin", SIZE_BINS, "B) acceptance % x relative size x condition x judge")

    out.append("\n=== C) intermediate error, edit position (% of solution length) ===")
    for level in POSITION_BINS:
        line = f"{level:10s}"
        for judge in JUDGES:
            s = sub[(sub.condition == "intermediate_error") & (sub.posbin == level) & (sub.judge_short == judge)]
            line += f"  {judge}: {100 * s.correct_maj.mean():5.1f} n={len(s)}"
        out.append(line)

    out.append(
        "\n=== D) largest gap within a condition (points), across kinds / across size bins (cells with n>=30) ==="
    )
    for condition, condition_name, _ in CONDITIONS:
        for judge in JUDGES:
            by_condition = sub[(sub.condition == condition) & (sub.judge_short == judge)]
            kind_rates = [
                100 * by_condition[by_condition.etype == level].correct_maj.mean()
                for level in EDIT_KINDS
                if len(by_condition[by_condition.etype == level]) >= 30
            ]
            size_rates = [
                100 * by_condition[by_condition.relbin == level].correct_maj.mean()
                for level in SIZE_BINS
                if len(by_condition[by_condition.relbin == level]) >= 30
            ]
            out.append(
                f"{condition_name:32s} {judge:10s} kind {max(kind_rates) - min(kind_rates):5.1f}"
                f"  size {max(size_rates) - min(size_rates):5.1f}"
            )
    return "\n".join(out), rows_kind, rows_size


def format_cell(rate: float, n: int) -> str:
    return "--" if n == 0 else f"{rate:.0f} ({n})"


def tex_lines(rows_kind: list, rows_size: list) -> list[str]:
    lines = [
        r"\begin{table}[!htb]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\crnew{\begin{tabular}{llrrr}",
        r"\toprule",
        r"Condition & Edit & V4-Flash & Qwen3.6 & R1-distill \\",
        r"\midrule",
    ]
    for _, condition_name, _ in CONDITIONS:
        lines.append(r"\multicolumn{5}{l}{\textit{" + condition_name + r"}} \\")
        for row_condition, level, cells in rows_kind:
            if row_condition == condition_name:
                lines.append(" & " + level + " & " + " & ".join(format_cell(p, n) for p, _, _, n in cells) + r" \\")
        for row_condition, level, cells in rows_size:
            if row_condition == condition_name:
                lines.append(
                    " & change " + level + " & " + " & ".join(format_cell(p, n) for p, _, _, n in cells) + r" \\"
                )
        lines.append(r"\addlinespace[2pt]")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{\crnew{Acceptance (\%, majority-correct) by the kind and the size of the injected edit, with the "
        r"number of solutions in parentheses. Kind is how the wrong number was formed; change is "
        r"$|\text{new}-\text{old}|/|\text{old}|$. Every row of a condition partitions the same 720 solutions.}}",
        r"\label{tab:edit-size}",
        r"\end{table}",
    ]
    return lines


def main() -> None:
    ensure_output_dirs()
    sub = load_panel_edits()
    text, rows_kind, rows_size = text_report(sub)
    print(text)
    (TAB_DIR / "edit_size.txt").write_text(text + "\n")
    (TAB_DIR / "tab_edit_size.tex").write_text("\n".join(tex_lines(rows_kind, rows_size)) + "\n")
    print("\nwrote tab_edit_size.tex")


if __name__ == "__main__":
    main()
