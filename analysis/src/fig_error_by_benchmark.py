"""Error-injection acceptance per benchmark, one row per judge (fig_error_by_benchmark.png).

Judges Qwen3.6, V4-Flash and R1-distill, five conditions on the x axis, one
bar per benchmark plus a dark-grey bar for the pooled 720-solution panel; bars
show the majority-vote acceptance (%) with Wilson 95% intervals. The only
input is error_injection_panel_3judges.csv in analysis/data; the figure is
written to analysis/figures.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import FIG_DIR, data, ensure_output_dirs  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from palette import BENCH, BENCH_LABEL, BENCH_ORDER  # noqa: E402

JUDGES = ["Qwen3.6", "V4-Flash", "R1-distill"]
CONDITIONS = ["clean", "intermediate_error", "truncated", "boxed_only", "consistent_error"]
CONDITION_LABELS = ["clean", "intermediate numeric", "truncation", "final-answer", "consistent final-answer"]
BAND_OK = "#dce8f2"
BAND_CLEAN = "#e4e4e4"
BAND_BAD = "#f2dada"
POOL_COLOUR = "#4d4d4d"
BAR_WIDTH = 0.135


def wilson(p, n, z=1.96):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def draw_bar(ax, x, m, n, colour, fontsize, bold=False):
    lo, hi = wilson(m, n)
    ax.bar(
        x,
        100 * m,
        BAR_WIDTH,
        yerr=[[max(0, 100 * (m - lo))], [max(0, 100 * (hi - m))]],
        capsize=1,
        error_kw={"lw": 0.5},
        color=colour,
        edgecolor="black",
        lw=0.3,
        zorder=2,
    )
    ax.annotate(
        f"{100 * m:.0f}",
        xy=(x, 100 * hi),
        xytext=(0, 1.5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        rotation=90,
        fontweight="bold" if bold else "normal",
        color="0.1" if bold else "0.2",
    )


def main():
    ensure_output_dirs()
    d = pd.read_csv(data("error_injection_panel_3judges.csv"))
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.5})
    fig, axs = plt.subplots(3, 1, figsize=(6.6, 3.0), sharex=True)
    x = np.arange(5)
    for ax, j in zip(axs, JUDGES, strict=False):
        ax.axvspan(-0.5, 0.5, color=BAND_CLEAN, zorder=0)
        ax.axvspan(0.5, 2.5, color=BAND_OK, zorder=0)
        ax.axvspan(2.5, 4.5, color=BAND_BAD, zorder=0)
        ax.axvline(0.5, color="0.35", lw=0.7, zorder=1)
        ax.axvline(2.5, color="0.35", lw=0.7, zorder=1)
        for bi, bm in enumerate(BENCH_ORDER):
            off = (bi - 2.5) * BAR_WIDTH
            for ci, c in enumerate(CONDITIONS):
                s = d[(d.judge_short == j) & (d.condition == c) & (d.benchmark == bm)].correct_maj
                draw_bar(ax, x[ci] + off, s.mean(), len(s), BENCH[bm], 5.6)
        for ci, c in enumerate(CONDITIONS):
            s = d[(d.judge_short == j) & (d.condition == c)].correct_maj
            draw_bar(ax, x[ci] + 2.5 * BAR_WIDTH, s.mean(), len(s), POOL_COLOUR, 6.2, bold=True)
        ax.set_ylim(0, 150)
        ax.set_yticks([0, 50, 100])
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylabel(j, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for xpos, label in [(1.5, "final answer correct"), (3.5, "final answer wrong")]:
        axs[0].text(
            xpos,
            1.03,
            label,
            transform=axs[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            style="italic",
            color="0.25",
        )
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(CONDITION_LABELS, fontsize=8)
    fig.supylabel("CoT accepted (%)", fontsize=8, x=0.005)
    handles = [Patch(facecolor=BENCH[b], edgecolor="black", lw=0.3, label=BENCH_LABEL[b]) for b in BENCH_ORDER]
    handles.append(Patch(facecolor=POOL_COLOUR, edgecolor="black", lw=0.3, label="pooled (720)"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=1.4,
        columnspacing=1.2,
    )
    fig.subplots_adjust(hspace=0.1, bottom=0.17, top=0.93, left=0.1, right=0.99)
    out = FIG_DIR / "fig_error_by_benchmark.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
