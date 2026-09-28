"""Dumbbell plot of CoT-Pass@64 versus Pass@64 per solver and benchmark (fig_gap_endpoints.png).

A 2x2 grid at single-column width, one panel per 64-sample benchmark. One row
per solver, grouped into the earlier and the current generation with shaded
bands; per row a thick segment runs from CoT-Pass@64 (open marker) to Pass@64
(filled marker) and the difference is printed at the row's right edge.
Coincident markers render as a bullseye, so a zero difference stays visible.
Anchor judge Qwen3.6 at the 16k budget; every number is one solver x
benchmark cell, nothing is averaged. The values are taken from gap_curve.csv
in analysis/data and the figure is written to analysis/figures.
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from common import FIG_DIR, data, ensure_output_dirs

matplotlib.use("Agg")

CSV = data("gap_curve.csv")
OUT = FIG_DIR / "fig_gap_endpoints.png"

K = 64
PANELS = [
    ("aime2026", "AIME (EN)"),
    ("aime2026_pt", "AIME (PT)"),
    ("aime2026_tr", "AIME (TR)"),
    ("tubitak_math2026", "TÜBİTAK (TR)"),
]
# (short label, solver, state, colour); None separates the two generations
ROWS = [
    ("Q2.5-7B", "Qwen2.5-7B", "base", "#08519c"),
    ("Q2.5-32B", "Qwen2.5-32B", "base", "#08519c"),
    ("Q2.5-7B-Inst", "Qwen2.5-7B-Instruct", "non-think", "#6baed6"),
    ("Q2.5-32B-Inst", "Qwen2.5-32B-Instruct", "non-think", "#6baed6"),
    None,
    ("Q3.5-4B-Base", "Qwen3.5-4B-Base", "base", "#a63603"),
    ("Q3.5-9B-Base", "Qwen3.5-9B-Base", "base", "#a63603"),
    ("Q3.5-4B", "Qwen3.5-4B", "non-think", "#fd8d3c"),
    ("Q3.5-9B", "Qwen3.5-9B", "non-think", "#fd8d3c"),
    ("G4-E2B", "gemma-4-E2B-it", "non-think", "#2ca02c"),
    ("G4-E4B", "gemma-4-E4B-it", "non-think", "#2ca02c"),
]


def load():
    g = pd.read_csv(CSV)
    return g[
        (g.judge_short == "Qwen3.6")
        & (g.judge_state == "think")
        & (g.solver_max_tokens == 16384)
        & (g.judge_max_tokens == 16384)
        & (g.k == K)
    ]


def main():
    ensure_output_dirs()
    g = load()

    fig, axgrid = plt.subplots(
        2, 2, figsize=(3.8, 6.6), sharex=True, sharey=True, gridspec_kw={"hspace": 0.22, "wspace": 0.06}
    )
    axes = axgrid.ravel()
    ys = []
    y = 0
    for row in ROWS:
        if row is None:
            y -= 1.5  # gap: the current-generation label gets its own strip
            ys.append(None)
        else:
            ys.append(y)
            y -= 1

    for ax, (bench, title) in zip(axes, PANELS, strict=False):
        for row, yy in zip(ROWS, ys, strict=False):
            if row is None:
                continue
            label, model, state, color = row
            d = g[(g.solver_model == model) & (g.state == state) & (g.benchmark == bench)]
            if not len(d):
                print(f"[WARN] missing cell: {model} x {bench}")
                continue
            p = 100 * d.pass_at_k.iloc[0]
            c = 100 * d.cot_pass_at_k.iloc[0]
            diff = p - c
            # thick rounded segment so its length (the difference) carries the visual weight
            ax.plot([c, p], [yy, yy], "-", color=color, lw=5.2, alpha=0.95, solid_capstyle="round", zorder=2)
            # open CoT-Pass marker first, filled Pass marker on top: coincident markers form a bullseye
            ax.plot([c], [yy], "o", mfc="white", mec=color, mew=1.5, ms=6.8, zorder=3)
            ax.plot([p], [yy], "o", mfc=color, mec="white", mew=0.7, ms=4.2, zorder=4)
            ax.text(
                125,
                yy,
                f"{diff:.1f}" if diff else "0",
                ha="right",
                va="center",
                fontsize=8.0,
                color=color,
                fontweight="bold" if diff >= 10 else "normal",
            )
        # left-aligned so the wide TÜBİTAK title clears the "diff." header at the right
        ax.set_title(title, fontsize=10.5, pad=3, fontweight="bold", loc="left")
        ax.set_xlim(0, 127)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="x", labelsize=7.5)
        ax.tick_params(axis="y", labelsize=8.5)
        # white gridlines read cleanly on top of the pastel generation bands
        ax.grid(axis="x", color="white", lw=1.0, alpha=0.9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        # the "diff." header sits above the axis so it cannot collide with the in-band generation labels
        ax.text(
            1.0,
            1.012,
            "diff.",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.0,
            color="0.35",
            style="italic",
        )
        sep_y = (ys[3] + ys[5]) / 2
        ax.axhspan(sep_y, ys[0] + 1.45, color="#e8edf5", zorder=0)
        ax.axhspan(ys[-1] - 0.7, sep_y, color="#faf5ee", zorder=0)
        ax.axhline(sep_y, color="0.7", lw=0.6)
        ax.text(3, ys[0] + 0.55, "Earlier generation", fontsize=8.5, fontweight="bold", color="0.2", va="bottom")
        ax.text(3, ys[5] + 0.42, "Current generation", fontsize=8.5, fontweight="bold", color="0.2", va="bottom")
        ax.set_ylim(ys[-1] - 0.7, ys[0] + 1.45)
    for ax in axgrid[:, 0]:
        ax.set_yticks([yy for yy in ys if yy is not None])
        ax.set_yticklabels([r[0] for r in ROWS if r is not None], fontsize=8.5)
    # x tick numbers under the top row too, not only under the bottom one
    for ax in axgrid[0]:
        ax.tick_params(axis="x", labelbottom=True)

    # the marker legend lives in the shared x label; CoT-Pass is usually the left endpoint
    fig.supxlabel("CoT-Pass@64 (○)  /  Pass@64 (●)", fontsize=9.5, y=0.012)
    fig.tight_layout(rect=(0, 0.022, 1, 1), pad=0.4)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
