"""Error-injection acceptance by the kind and the size of the injected edit (fig_edit_size.png).

Two rows (kind of edit; relative change of the edited number) by three columns
(the three numeric error conditions), one line per judge with Wilson 95%
intervals; the tick labels carry n per category. The edits are described in
corruption_log.csv and the verdicts in error_injection_panel_3judges.csv, both
in analysis/data; the figure is saved under analysis/figures.
"""

import math

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import FIG_DIR, data, ensure_output_dirs  # noqa: E402
from palette import JUDGE  # noqa: E402

JUDGES = ["V4-Flash", "Qwen3.6", "R1-distill"]
# (condition, panel title, column of corruption_log.csv that names the edit type)
CONDITIONS = [
    ("intermediate_error", "intermediate numeric error", "intermediate_menu"),
    ("boxed_only", "final-answer error", "wrong_number_menu"),
    ("consistent_error", "consistent final-answer error", "wrong_number_menu"),
]
EDIT_TYPES = ["±1 or ±2", "one digit\nchanged", "two digits\nswapped"]
REL_BINS = ["<1%", "1–10%", "10–100%", ">100%"]
EDIT_TYPE_LABEL = {"pm12": "±1 or ±2", "single_digit": "one digit\nchanged", "digit_swap": "two digits\nswapped"}


def edit_type(row, col):
    v = str(row[col]).replace("(fallback)", "")
    return EDIT_TYPE_LABEL.get(v, v)


def rel_bin(r):
    o = r.old_number
    n = r.new_number
    rel = abs(n - o) / abs(o) if o != 0 else abs(n - o)
    if rel < 0.01:
        return "<1%"
    if rel < 0.1:
        return "1–10%"
    if rel < 1:
        return "10–100%"
    return ">100%"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def load():
    log = pd.read_csv(data("corruption_log.csv"))
    pan = pd.read_csv(data("error_injection_panel_3judges.csv"))
    m = pan.merge(log, on=["solution_id", "condition"], how="left")
    sub = m[m.condition.isin([c for c, _, _ in CONDITIONS])].copy()
    assert len(sub) == 720 * 3 * 3 and sub.corruption_applied.all()
    menu_column = sub.condition.map({c: col for c, _, col in CONDITIONS})
    sub["etype"] = [edit_type(r, col) for (_, r), col in zip(sub.iterrows(), menu_column, strict=False)]
    sub["relbin"] = sub.apply(rel_bin, axis=1)
    return sub


def main():
    ensure_output_dirs()
    sub = load()
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.5})
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.6), sharey=True)
    rows = [("etype", EDIT_TYPES, "kind of edit"), ("relbin", REL_BINS, "relative change of the number")]
    for r, (key, levels, rowlab) in enumerate(rows):
        for k, (c, cname, _) in enumerate(CONDITIONS):
            ax = axes[r, k]
            x = np.arange(len(levels))
            for ji, j in enumerate(JUDGES):
                ys = []
                lo = []
                hi = []
                for lv in levels:
                    s = sub[(sub.condition == c) & (sub[key] == lv) & (sub.judge_short == j)]
                    n = len(s)
                    kk = int(s.correct_maj.sum())
                    p = 100 * kk / n
                    a, b = wilson(kk, n)
                    ys.append(p)
                    lo.append(p - a)
                    hi.append(b - p)
                ax.errorbar(
                    x + (ji - 1) * 0.12,
                    ys,
                    yerr=[lo, hi],
                    fmt="o-",
                    ms=3,
                    lw=1,
                    capsize=1.5,
                    elinewidth=0.6,
                    color=JUDGE.get(j, "k"),
                    label=j,
                )
            ns = [len(sub[(sub.condition == c) & (sub[key] == lv) & (sub.judge_short == JUDGES[0])]) for lv in levels]
            ax.set_xticks(x)
            ax.set_xticklabels([f"{lv}\n(n={n})" for lv, n in zip(levels, ns, strict=False)], fontsize=6)
            ax.set_ylim(0, 105)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(-0.5, len(levels) - 0.5)
            if r == 0:
                ax.set_title(cname, fontsize=7.5, fontweight="bold", pad=4)
            if k == 0:
                ax.set_ylabel(f"{rowlab}\naccepted (%)", fontsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = FIG_DIR / "fig_edit_size.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
