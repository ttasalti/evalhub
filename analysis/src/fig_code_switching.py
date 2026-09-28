"""Language of the answer-correct reasoning chains per solver, and acceptance by language class
(fig_code_switching.png).

Left column: share of chains in the benchmark language, mixed or English, with
n at the right edge. Right column: acceptance (%) under Qwen3.6 and Gemma4 by
language class (groups with fewer than 20 judged chains are not drawn). Top
row: AIME (PT), AIME (TR) and TUBITAK (TR) pooled; bottom row: the Portuguese
exams. Classes and thresholds are those of code_switching_tables.py. The
per-chain measurements are code_switching_main.csv and code_switching_ptexams.csv
in analysis/data; the figure is written to analysis/figures.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import FIG_DIR, data, ensure_output_dirs  # noqa: E402

COL = {"target": "#009e73", "mixed": "#e69f00", "english": "#0072b2"}
LAB = {"target": "benchmark language", "mixed": "mixed", "english": "English"}
CLS = ["target", "mixed", "english"]
JUD = [("approved_Qwen3.6", "Qwen3.6", "o"), ("approved_Gemma4", "Gemma4", "D")]
SOLVERS_MAIN = [
    "Q2.5-7B",
    "Q2.5-32B",
    "Q2.5-7B-Inst",
    "Q2.5-32B-Inst",
    "Q3.5-4B-Base",
    "Q3.5-9B-Base",
    "Q3.5-4B",
    "Q3.5-9B",
    "G4-E2B",
    "G4-E4B",
]
SOLVERS_PTEXAMS = [
    "Q2.5-7B",
    "Q2.5-32B",
    "Q2.5-7B-Inst",
    "Q2.5-32B-Inst",
    "Q3.5-4B-Base",
    "Q3.5-9B-Base",
    "Q3.5-4B-Think",
    "Q3.5-9B-Think",
    "G4-E2B-Think",
    "G4-E4B-Think",
]


def classify(d):
    d = d[d.n_letters.fillna(0) >= 20].copy()
    d["cls"] = np.where(d.share_target >= 0.8, "target", np.where(d.share_eng >= 0.8, "english", "mixed"))
    return d


def draw_block(a0, a1, df, solvers, title):
    y = np.arange(len(solvers))[::-1]
    for yi, s_name in zip(y, solvers, strict=False):
        s = df[df.solver == s_name]
        sh = s.cls.value_counts(normalize=True) * 100
        left = 0
        for c in CLS:
            v = sh.get(c, 0)
            a0.barh(yi, v, left=left, color=COL[c], height=0.72, edgecolor="white", linewidth=0.6)
            if v >= 12:
                a0.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=6, color="white")
            left += v
        a0.text(101.5, yi, f"{len(s)}", ha="left", va="center", fontsize=5.5, color="#555555")
        for k, c in enumerate(CLS):
            for j, (col, _jn, mk) in enumerate(JUD):
                x = s[(s.cls == c) & s[col].notna()]
                if len(x) < 20:
                    continue
                acc = 100 * x[col].astype(float).mean()
                off = (k - 1) * 0.22 + (j - 0.5) * 0.09
                a1.plot(
                    acc,
                    yi + off,
                    marker=mk,
                    ms=4.2,
                    color=COL[c],
                    mfc=COL[c] if j == 0 else "white",
                    mew=1.1,
                    ls="none",
                )
    for a in (a0, a1):
        a.set_yticks(y)
        a.set_ylim(-0.6, len(solvers) - 0.4)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=6.5)
        a.axhline(y[3] - 0.5, color="#bbbbbb", lw=0.6, ls=":")
    a0.set_yticklabels(solvers)
    a1.set_yticklabels([])
    a0.set_xlim(0, 100)
    a0.set_xticks([0, 25, 50, 75, 100])
    a1.set_xlim(40, 101)
    a1.xaxis.grid(True, color="#e5e5e5", lw=0.5)
    a1.set_axisbelow(True)
    for yi in y:
        a1.axhline(yi - 0.5, color="#f0f0f0", lw=0.4)
    a0.set_title("language of the chains (%)", fontsize=7, loc="left")
    a0.text(-0.36, 1.13, title, transform=a0.transAxes, fontsize=8, fontweight="bold", ha="left")
    a1.set_title("accepted (%), by language of the chain", fontsize=7, loc="left")
    a0.text(101.5, len(solvers) - 0.45, "$n$", fontsize=6, color="#555555", va="bottom")


def main():
    ensure_output_dirs()
    main_grid = classify(pd.read_csv(data("code_switching_main.csv")))
    main_grid = main_grid[main_grid.benchmark != "aime2026"]
    ptexams = classify(pd.read_csv(data("code_switching_ptexams.csv")))
    fig, axes = plt.subplots(
        2, 2, figsize=(7.0, 5.4), gridspec_kw={"width_ratios": [1, 1.25], "hspace": 0.5, "wspace": 0.22}
    )
    blocks = [
        (main_grid, SOLVERS_MAIN, "AIME (PT), AIME (TR) and TÜBİTAK (TR), pooled"),
        (ptexams, SOLVERS_PTEXAMS, "PT exams"),
    ]
    for r, (df, solvers, title) in enumerate(blocks):
        draw_block(axes[r][0], axes[r][1], df, solvers, title)
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in CLS]
    h += [
        plt.Line2D([], [], marker=mk, color="#444444", mfc="#444444" if j == 0 else "white", ls="none", ms=4.5)
        for j, (_, _, mk) in enumerate(JUD)
    ]
    fig.legend(
        h,
        [LAB[c] for c in CLS] + [jn for _, jn, _ in JUD],
        loc="lower center",
        ncol=5,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    out = FIG_DIR / "fig_code_switching.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
