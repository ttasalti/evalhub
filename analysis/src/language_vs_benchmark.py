"""Language versus benchmark effect on cell acceptance rates: paired Wilcoxon tests across the ten solvers
(the tests quoted in Section 4.1).

Reads report_tasks.csv from analysis/data and writes language_vs_benchmark.txt to analysis/tables.

Unit = solver (10), measurement = cell acceptance rate (approvals per correct generation, %) at the 16k solver
budget with the 16k judge budget. Differences are paired within solver: TR-EN and PT-EN compare translations
of the same AIME problems (language effect); TUB-TR compares two benchmarks in the same language.
Test: exact Wilcoxon signed-rank, n = 10, zero differences dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import TAB_DIR, data, ensure_output_dirs
from scipy.stats import wilcoxon

CELL_KEYS = ["cell", "benchmark", "solver_max_tokens"]
ROWS = [
    "Qwen2.5-7B|base",
    "Qwen2.5-32B|base",
    "Qwen2.5-7B-Instruct|non-think",
    "Qwen2.5-32B-Instruct|non-think",
    "Q-4B·Base|base",
    "Q-9B·Base|base",
    "Q-4B|non-think",
    "Q-9B|non-think",
    "G4-E2B|non-think",
    "G4-E4B|non-think",
]
BENCHMARKS = ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026"]
CONTRASTS = [
    ("TR-EN", "aime2026_tr", "aime2026"),
    ("PT-EN", "aime2026_pt", "aime2026"),
    ("TUB-TR", "tubitak_math2026", "aime2026_tr"),
]
BUDGET = 16384


def main() -> None:
    ensure_output_dirs()
    df = pd.read_csv(data("report_tasks.csv"))
    df["cell"] = df.model_short + "|" + df.state
    base = df[df.judge_short == "No-Judge"].groupby(CELL_KEYS).agg(correct=("true_count", "sum")).reset_index()
    judged = df[df.judge_short != "No-Judge"].copy()
    judged["jb"] = judged.judge_max_tokens.astype(int)
    judged_agg = (
        judged.groupby(CELL_KEYS + ["judge_short", "jb"])
        .agg(approved=("true_count", "sum"), rejected=("n_veto", "sum"))
        .reset_index()
    )

    def acceptance(cell: str, benchmark: str, judge: str) -> float:
        b = base[(base.cell == cell) & (base.benchmark == benchmark) & (base.solver_max_tokens == BUDGET)].iloc[0]
        j = judged_agg[
            (judged_agg.cell == cell)
            & (judged_agg.benchmark == benchmark)
            & (judged_agg.solver_max_tokens == BUDGET)
            & (judged_agg.judge_short == judge)
            & (judged_agg.jb == BUDGET)
        ].iloc[0]
        assert j.approved + j.rejected == b.correct
        return 100 * j.approved / b.correct

    lines = []
    for judge in ["Qwen3.6", "Gemma4"]:
        rates = {bm: np.array([acceptance(cell, bm, judge) for cell in ROWS]) for bm in BENCHMARKS}
        lines.append(f"{judge}")
        for label, minuend, subtrahend in CONTRASTS:
            d = rates[minuend] - rates[subtrahend]
            p_value = wilcoxon(d[d != 0], method="exact").pvalue
            lines.append(
                f"  {label:7s} mean {d.mean():+5.1f} median {np.median(d):+5.1f} "
                f"sign {int((d > 0).sum())}+/{int((d < 0).sum())}-  Wilcoxon exact p={p_value:.3f} "
                f"(zero differences dropped: {int((d == 0).sum())})  per-solver {np.round(d, 1)}"
            )
    text = "\n".join(lines)
    print(text)
    (TAB_DIR / "language_vs_benchmark.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
