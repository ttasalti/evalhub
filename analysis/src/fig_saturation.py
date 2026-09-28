"""CoT-Pass@64 and the acceptance rate as a function of c, per judge (fig_saturation.png).

Two rows (judges Qwen3.6 and Gemma4) by two columns. c is the number of a
question's 64 generations whose final answer is correct, counted before
judging; the questions of the ten 16k solvers are grouped into four c bins.
Left: share of question and solver pairs with at least one accepted correct
generation (CoT-Pass@64). Right: share of correct generations the judge
accepted. Bars per solver generation with Wilson 95% intervals; the tick
labels carry the pair counts of each bin. Built from report_tasks.csv in
analysis/data; the figure is written to analysis/figures.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import FIG_DIR, data, ensure_output_dirs  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from palette import GEN  # noqa: E402

BENCHMARKS = ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026"]
EARLIER = ["Qwen2.5-7B|base", "Qwen2.5-32B|base", "Qwen2.5-7B-Instruct|non-think", "Qwen2.5-32B-Instruct|non-think"]
CURRENT = [
    "Q-4B·Base|base",
    "Q-9B·Base|base",
    "Q-4B|non-think",
    "Q-9B|non-think",
    "G4-E2B|non-think",
    "G4-E4B|non-think",
]
BINS = [(1, 2), (3, 8), (9, 32), (33, 64)]
BIN_LABELS = ["1–2", "3–8", "9–32", "33–64"]
JUDGES = ["Qwen3.6", "Gemma4"]
BAR_WIDTH = 0.36


def load():
    df = pd.read_csv(data("report_tasks.csv"))
    df["cell"] = df.model_short + "|" + df.state
    df["c"] = (df.true_count.fillna(0) + df.n_veto.fillna(0)).astype(int)
    df["cc"] = df.true_count.fillna(0).astype(int)
    df["veto"] = df.n_veto.fillna(0).astype(int)
    return df


def judge_rows(df, judge):
    s = df[(df.solver_max_tokens == 16384) & df.benchmark.isin(BENCHMARKS) & (df.judge_short == judge)]
    assert s.judge_max_tokens.nunique() == 1, judge
    return s


def cot_pass(b):
    """CoT-Pass@64 (%): share of pairs with at least one accepted correct generation, plus k and n."""
    n = len(b)
    k = int((b.cc > 0).sum())
    return (100 * k / n if n else np.nan), k, n


def acceptance(b):
    """Acceptance (%) of correct generations, plus the accepted and the correct counts."""
    corr = int(b.c.sum())
    a = corr - int(b.veto.sum())
    return (100 * a / corr if corr else np.nan), a, corr


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


PANELS = [
    (cot_pass, "CoT-Pass@64 given c", "CoT-Pass@64 (%)"),
    (acceptance, "acceptance of correct generations given c", "acceptance rate (%)"),
]


def main():
    ensure_output_dirs()
    df = load()
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6})
    pair_counts = {}
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex=False)
    x = np.arange(len(BINS))
    for r, judge in enumerate(JUDGES):
        s = judge_rows(df, judge)
        for col, (fn, title, ylab) in enumerate(PANELS):
            ax = axs[r, col]
            for gi, (gen, solvers) in enumerate([("earlier", EARLIER), ("current", CURRENT)]):
                g = s[s.cell.isin(solvers)]
                off = (gi - 0.5) * BAR_WIDTH
                for bi, (lo, hi) in enumerate(BINS):
                    b = g[g.c.between(lo, hi)]
                    v, k, n = fn(b)
                    lo_, hi_ = wilson(k, n)
                    if col == 0:
                        pair_counts.setdefault(bi, {})[gen] = n
                    ax.bar(x[bi] + off, v, BAR_WIDTH, color=GEN[gen], zorder=2)
                    ax.errorbar(
                        x[bi] + off,
                        v,
                        yerr=[[max(0, v - lo_)], [max(0, hi_ - v)]],
                        fmt="none",
                        ecolor="black",
                        elinewidth=0.7,
                        capsize=2,
                        capthick=0.7,
                        zorder=3,
                    )
                    ax.text(
                        x[bi] + off, hi_ + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.4, color="0.15", zorder=4
                    )
            ax.set_ylim(0, 118)
            ax.set_yticks(range(0, 101, 20))
            ax.set_xlim(-0.6, len(BINS) - 0.4)
            ax.grid(axis="y", alpha=0.3, lw=0.5, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=6.5, length=2)
            ax.set_title(f"{title}, judge {judge}", fontsize=7.2, loc="left", pad=2)
            ax.set_ylabel(ylab, fontsize=6.6)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [
                    f"c = {b}\n{pair_counts[i]['earlier']} | {pair_counts[i]['current']} pairs"
                    for i, b in enumerate(BIN_LABELS)
                ],
                fontsize=6.4,
            )
    fig.supxlabel(
        "c = number of the question's 64 generations whose final answer is correct, counted before judging "
        "(the judge does not change c);  pairs = question and solver pairs in the bin, earlier | current",
        fontsize=6.8,
        y=0.06,
    )
    handles = [
        Patch(color=GEN["earlier"], label="earlier generation (four Qwen2.5 solvers)"),
        Patch(color=GEN["current"], label="current generation (six Qwen3.5 and Gemma-4 solvers)"),
        Line2D([], [], color="black", lw=0.7, label="95% Wilson interval"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.4, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(hspace=0.42, wspace=0.2, left=0.08, right=0.99, top=0.95, bottom=0.17)
    out = FIG_DIR / "fig_saturation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
