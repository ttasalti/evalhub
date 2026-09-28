"""Pass@k - CoT-Pass@k per solver generation as k grows, drawn as two lines (fig_k_ladder.png).

One curve per solver generation over k = 1, 2, ..., 64 under the anchor judge
Qwen3.6 at the 16k budget. The weighting matches bootstrap_questions.py: each
solver is a question-weighted mean over its four 64-sample benchmarks, each
generation the plain mean over its solvers. The two endpoint values are
printed on the figure; the per-solver curves are computed for the console
summary only. Input is gap_curve.csv in analysis/data; the figure is saved
as analysis/figures/fig_k_ladder.png.
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from common import FIG_DIR, data, ensure_output_dirs

matplotlib.use("Agg")

CSV = data("gap_curve.csv")
OUT = FIG_DIR / "fig_k_ladder.png"

KS = [1, 2, 4, 8, 16, 32, 64]
BM = ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026"]
ROWS = [
    ("Qwen2.5-7B", "base", "earlier"),
    ("Qwen2.5-32B", "base", "earlier"),
    ("Qwen2.5-7B-Instruct", "non-think", "earlier"),
    ("Qwen2.5-32B-Instruct", "non-think", "earlier"),
    ("Qwen3.5-4B-Base", "base", "current"),
    ("Qwen3.5-9B-Base", "base", "current"),
    ("Qwen3.5-4B", "non-think", "current"),
    ("Qwen3.5-9B", "non-think", "current"),
    ("gemma-4-E2B-it", "non-think", "current"),
    ("gemma-4-E4B-it", "non-think", "current"),
]
STYLE = {"earlier": ("#08519c", "Earlier generation"), "current": ("#a63603", "Current generation")}


def generation_curves(g):
    """Mean difference per generation and k: question-weighted per solver, plain mean over solvers."""
    curves = {}
    for gen in ["earlier", "current"]:
        per_solver = []
        for m, st, gg in ROWS:
            if gg != gen:
                continue
            ys = []
            for k in KS:
                d = g[(g.solver_model == m) & (g.state == st) & (g.k == k)]
                if len(d) != 4:
                    raise SystemExit(f"missing cells: {m} {st} k={k}: {len(d)}")
                w = d.n_tasks.to_numpy()
                ys.append(100 * float((d.cot_gap.to_numpy() * w).sum() / w.sum()))
            per_solver.append(ys)
        curves[gen] = [sum(c[i] for c in per_solver) / len(per_solver) for i in range(len(KS))]
    return curves


def main():
    ensure_output_dirs()
    g = pd.read_csv(CSV)
    g = g[
        (g.judge_short == "Qwen3.6")
        & (g.judge_state == "think")
        & (g.solver_max_tokens == 16384)
        & (g.judge_max_tokens == 16384)
        & (g.benchmark.isin(BM))
    ]
    curves = generation_curves(g)

    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    for gen, ys in curves.items():
        color, label = STYLE[gen]
        ax.plot(range(len(KS)), ys, "-o", color=color, lw=2.4, ms=4.5, mec="white", mew=0.8, zorder=4, label=label)
        ax.annotate(
            f"{ys[0]:.1f}",
            (0, ys[0]),
            textcoords="offset points",
            xytext=(-2, 7 if gen == "earlier" else -12),
            ha="center",
            fontsize=7.5,
            color=color,
        )
        ax.annotate(
            f"{ys[-1]:.1f}",
            (len(KS) - 1, ys[-1]),
            textcoords="offset points",
            xytext=(0, 7 if gen == "earlier" else -12),
            ha="right",
            fontsize=8.0,
            color=color,
            fontweight="bold",
        )

    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels(KS, fontsize=8)
    ax.set_xlabel("$k$ (sampled generations)", fontsize=8.5)
    ax.set_ylabel("Pass@$k$ $-$ CoT-Pass@$k$", fontsize=8.5)
    ax.set_ylim(-1.5, 22.5)
    ax.set_xlim(-0.35, len(KS) - 0.35)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="0.88", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left", handlelength=1.4, borderpad=0.1, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, dpi=400)
    print(f"wrote {OUT}")
    for gen, ys in curves.items():
        print(f"  {gen:8s} " + " ".join(f"{v:5.1f}" for v in ys))


if __name__ == "__main__":
    main()
