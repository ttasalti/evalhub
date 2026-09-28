"""Acceptance rate of correct generations per solver and benchmark (fig_acceptance.png).

Two rows, one per judge (Qwen3.6, Gemma4). Each solver has four benchmark bars
and a fifth dark-grey bar pooling the four benchmarks; the number of correct
generations is printed at the foot of each bar. The y axis starts at 30 (%).
The counts come from report_tasks.csv in analysis/data and the figure is
written to analysis/figures.
"""

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import FIG_DIR, data, ensure_output_dirs  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from palette import BENCH, BENCH_LABEL, GEN  # noqa: E402

SOLVER_KEY = ["cell", "benchmark", "solver_max_tokens"]
JUDGE_BUDGET = {"Qwen3.6": 16384, "Gemma4": 16384, "V4-Flash": 20480, "R1-distill": 16384}
BENCHMARKS = [(bm, BENCH_LABEL[bm], BENCH[bm]) for bm in ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026"]]
ROWS = [
    ("Q2.5-7B", "Qwen2.5-7B|base"),
    ("Q2.5-32B", "Qwen2.5-32B|base"),
    ("Q2.5-7B-Inst", "Qwen2.5-7B-Instruct|non-think"),
    ("Q2.5-32B-Inst", "Qwen2.5-32B-Instruct|non-think"),
    ("Q3.5-4B-Base", "Q-4B·Base|base"),
    ("Q3.5-9B-Base", "Q-9B·Base|base"),
    ("Q3.5-4B", "Q-4B|non-think"),
    ("Q3.5-9B", "Q-9B|non-think"),
    ("G4-E2B", "G4-E2B|non-think"),
    ("G4-E4B", "G4-E4B|non-think"),
]
JUDGE_ROWS = ["Qwen3.6", "Gemma4"]
PREV = GEN["earlier"]
CUR = GEN["current"]
POOL_COLOUR = "#4d4d4d"
YMIN = 30
XPOS = list(range(10))
DIV = 3.5
XMAX = 9.6
BAR_WIDTH = 0.16


def load_tables():
    df = pd.read_csv(data("report_tasks.csv"))
    df["cell"] = df.model_short + "|" + df.state
    nj = df[df.judge_short == "No-Judge"]
    base = nj.groupby(SOLVER_KEY).agg(Q=("task_id", "nunique"), correct=("true_count", "sum")).reset_index()
    jd = df[df.judge_short != "No-Judge"].copy()
    jd["jb"] = jd.judge_max_tokens.astype(int)
    jag = (
        jd.groupby(SOLVER_KEY + ["judge_short", "jb"])
        .agg(approved=("true_count", "sum"), rejected=("n_veto", "sum"), nt=("task_id", "nunique"))
        .reset_index()
    )
    return base, jag


def cell(base, jag, c, bm, js):
    """Rejection rate (%), rejected count and correct count for one solver, benchmark and judge."""
    b = base[(base.cell == c) & (base.benchmark == bm) & (base.solver_max_tokens == 16384)].iloc[0]
    j = jag[
        (jag.cell == c)
        & (jag.benchmark == bm)
        & (jag.solver_max_tokens == 16384)
        & (jag.judge_short == js)
        & (jag.jb == JUDGE_BUDGET[js])
    ]
    if not len(j):
        return None
    j = j.iloc[0]
    assert j.nt == b.Q and j.approved + j.rejected == b.correct
    return 100 * j.rejected / b.correct, int(j.rejected), int(b.correct)


def bar_with_labels(ax, x, value, n, colour, edge=False):
    kwargs = {"edgecolor": "black", "lw": 0.3} if edge else {}
    ax.bar(x, value - YMIN, BAR_WIDTH, bottom=YMIN, color=colour, zorder=2, **kwargs)
    if value - YMIN >= 16:
        ax.text(x, value + 0.8, f"{value:.0f}%", ha="center", va="bottom", fontsize=5.0, rotation=90, color="0.25")
        ax.text(x, YMIN + 1.0, f"{n}", ha="center", va="bottom", fontsize=4.6, rotation=90, color="white", zorder=5)
    else:
        ax.text(
            x, value + 0.8, f"{value:.0f}% ({n})", ha="center", va="bottom", fontsize=5.0, rotation=90, color="0.25"
        )


def draw_judge_row(ax, base, jag, js):
    for i, (_lab, c) in enumerate(ROWS):
        n_accepted = n_total = 0
        for k, (bm, _t, col) in enumerate(BENCHMARKS):
            res = cell(base, jag, c, bm, js)
            if res is None:
                continue
            v, rj, n = res
            v = 100 - v
            n_accepted += n - rj
            n_total += n
            bar_with_labels(ax, XPOS[i] + (k - 2) * BAR_WIDTH, v, n, col)
        bar_with_labels(ax, XPOS[i] + 2 * BAR_WIDTH, 100 * n_accepted / n_total, n_total, POOL_COLOUR, edge=True)
    pooled = {}
    for cells, col in [(ROWS[:4], PREV), (ROWS[4:], CUR)]:
        n_accepted = n_total = 0
        for _, c in cells:
            for bm, _, _ in BENCHMARKS:
                _, rj, n = cell(base, jag, c, bm, js)
                n_accepted += n - rj
                n_total += n
        pooled[col] = 100 * n_accepted / n_total
    ax.set_ylim(YMIN, 112)
    ax.set_xlim(-0.6, XMAX)
    ax.set_yticks(range(30, 101, 10))
    ax.set_yticklabels([str(t) for t in range(30, 101, 10)])
    # Axis-break marks at the foot of the y axis.
    for dy in (0.0, 0.012):
        ax.plot(
            [-0.008, 0.008], [0.01 + dy, 0.03 + dy], transform=ax.transAxes, color="k", lw=0.8, clip_on=False, zorder=5
        )
    ax.tick_params(axis="y", labelsize=6.5)
    ax.axvspan(-0.6, DIV, color="#e8edf5", zorder=0)
    ax.axvspan(DIV, XMAX, color="#faf5ee", zorder=0)
    ax.axvline(DIV, color="0.35", lw=0.7, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, lw=0.5, zorder=0.5)
    ax.set_ylabel(f"acceptance rate (%)\njudge: {js}", fontsize=7)
    ax.text(
        0.20,
        1.02,
        f"earlier generation (pooled {pooled[PREV]:.1f}%)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color=PREV,
    )
    ax.text(
        0.70,
        1.02,
        f"current generation (pooled {pooled[CUR]:.1f}%)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color=CUR,
    )
    ax.set_xticks(XPOS)
    ax.set_xticklabels([lab for lab, c in ROWS], rotation=30, ha="right", fontsize=6.8)
    ax.tick_params(axis="x", length=2)


def main():
    ensure_output_dirs()
    base, jag = load_tables()
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
    fig, axs = plt.subplots(
        len(JUDGE_ROWS), 1, figsize=(7.2, 2.6 * len(JUDGE_ROWS) + 0.9), sharex=False, gridspec_kw={"hspace": 0.4}
    )
    for ax, js in zip(axs, JUDGE_ROWS, strict=False):
        draw_judge_row(ax, base, jag, js)
    handles = [Patch(color=col, label=t) for _, t, col in BENCHMARKS]
    handles.append(Patch(facecolor=POOL_COLOUR, edgecolor="black", lw=0.3, label="pooled (four benchmarks)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    out = FIG_DIR / "fig_acceptance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
