"""Forest plot of the question-level bootstrap intervals for Pass@64 - CoT-Pass@64 (fig_bootstrap.png).

The intervals are read from diff_bootstrap_ci.csv in analysis/data, which
bootstrap_questions.py produces; the figure goes to analysis/figures.

One row per solver, grouped into the earlier and the current generation, plus
a bold "mean" row per generation. The observed difference and its 95% interval
are printed at the right of each row. The per-cell rows of the CSV are not
drawn: a 30-question cell whose difference is zero everywhere gets a degenerate
[0, 0] interval, which would read as "no uncertainty".
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from common import FIG_DIR, data, ensure_output_dirs

matplotlib.use("Agg")

SRC = data("diff_bootstrap_ci.csv")
OUT = FIG_DIR / "fig_bootstrap.png"

BLUE = "#08519c"
ORANGE = "#a63603"
EARLIER = [
    ("Qwen2.5-7B", "Q2.5-7B"),
    ("Qwen2.5-32B", "Q2.5-32B"),
    ("Qwen2.5-7B-Instruct", "Q2.5-7B-Inst"),
    ("Qwen2.5-32B-Instruct", "Q2.5-32B-Inst"),
]
CURRENT = [
    ("Qwen3.5-4B-Base", "Q3.5-4B-Base"),
    ("Qwen3.5-9B-Base", "Q3.5-9B-Base"),
    ("Qwen3.5-4B", "Q3.5-4B"),
    ("Qwen3.5-9B", "Q3.5-9B"),
    ("gemma-4-E2B-it", "G4-E2B"),
    ("gemma-4-E4B-it", "G4-E4B"),
]
# x extent of the axis line, x anchor of the printed estimate, x anchor of the printed interval
XMAX = 30
XNUM = 35.5
XCI = 37.0


def build_rows(d):
    """Rows as (label, obs, lo, hi, colour, bold); None marks the gap between generations."""
    mod = d[d.level == "model_avg"].set_index("model")
    gen = d[d.level == "generation"].set_index("model")
    rows = []
    for key, short in EARLIER:
        r = mod.loc[key]
        rows.append((short, r.obs, r.lo, r.hi, BLUE, False))
    r = gen.loc["earlier"]
    rows.append(("mean", r.obs, r.lo, r.hi, BLUE, True))
    rows.append(None)
    for key, short in CURRENT:
        r = mod.loc[key]
        rows.append((short, r.obs, r.lo, r.hi, ORANGE, False))
    r = gen.loc["current"]
    rows.append(("mean", r.obs, r.lo, r.hi, ORANGE, True))
    return rows


def main():
    ensure_output_dirs()
    rows = build_rows(pd.read_csv(SRC))

    ys = []
    y = 0.0
    for r in rows:
        ys.append(None if r is None else y)
        y -= 1.2 if r is None else 1.0

    fig, ax = plt.subplots(figsize=(3.5, 3.05))
    sep1 = (ys[4] + ys[6]) / 2  # between the earlier mean and the first current row
    sep2 = ys[-1] - 0.6  # bottom edge of the current band
    ax.axhspan(sep1, ys[0] + 1.15, color="#e8edf5", zorder=0)
    ax.axhspan(sep2, sep1, color="#faf5ee", zorder=0)
    ax.axhline(sep1, color="0.7", lw=0.6, zorder=1)
    ax.axhline(sep2, color="0.7", lw=0.6, zorder=1)
    ax.axvline(0, color="0.72", lw=0.8, zorder=1)

    for r, yy in zip(rows, ys, strict=False):
        if r is None:
            continue
        label, obs, lo, hi, color, bold = r
        ax.plot(
            [lo, hi], [yy, yy], "-", color=color, lw=2.6 if bold else 2.0, alpha=0.95, solid_capstyle="round", zorder=3
        )
        ax.plot([obs], [yy], "o", mfc=color, mec="white", mew=0.9, ms=6.5 if bold else 5.2, zorder=4)
        ax.text(
            XNUM,
            yy,
            f"{obs:.1f}",
            ha="right",
            va="center",
            fontsize=7.5,
            color=color,
            fontweight="bold" if bold else "normal",
        )
        ax.text(
            XCI,
            yy,
            f"[{lo:.1f}, {hi:.1f}]",
            ha="left",
            va="center",
            fontsize=7.0,
            color=color if bold else "0.45",
            fontweight="bold" if bold else "normal",
        )

    ax.text(0.7, ys[0] + 0.5, "Earlier generation", fontsize=8.5, fontweight="bold", color="0.2", va="bottom", zorder=5)
    ax.text(
        0.7, ys[6] + 0.42, "Current generation", fontsize=8.5, fontweight="bold", color="0.2", va="bottom", zorder=5
    )

    drawn = [r for r in rows if r is not None]
    ax.set_yticks([v for v in ys if v is not None])
    ax.set_yticklabels([r[0] for r in drawn], fontsize=8.5)
    for tick, r in zip(ax.get_yticklabels(), drawn, strict=False):
        if r[5]:
            tick.set_fontweight("bold")
    ax.set_ylim(sep2 - 0.25, ys[0] + 1.15)
    ax.set_xlim(-2, 55)
    ax.set_xticks([0, 10, 20, 30])
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Pass@64 $-$ CoT-Pass@64", fontsize=8.5)
    # white gridlines read cleanly over the pastel bands
    ax.grid(axis="x", color="white", lw=1.0, alpha=0.9, zorder=2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_bounds(-2, XMAX)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, dpi=400)
    print(f"wrote {OUT}: {len(drawn)} rows")


if __name__ == "__main__":
    main()
