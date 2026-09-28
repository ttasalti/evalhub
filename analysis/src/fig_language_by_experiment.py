"""Language of the reasoning chains and of the judges' output, one figure per experiment
(fig_language_exp1.png, fig_language_exp2.png, fig_language_exp3.png).

Each chain (or judgment) is classified from its letter shares as benchmark
language (share >= 0.8), English (share >= 0.8) or mixed; chains with fewer than
20 letters are dropped and groups with fewer than 20 chains are not drawn.

fig_language_exp1.png  Experiment 1 (error injection): left, language of the 720 original chains per
                       benchmark and solver (Qwen3.5-4B / 9B thinking); right, language of the judges'
                       output on the same solutions (Qwen3.6, V4-Flash, R1-distill; clean condition).
fig_language_exp2.png  Experiment 2 (main grid): top, language of the answer-correct chains of the ten
                       solvers per benchmark; bottom, language of the judgments per judge.
fig_language_exp3.png  Experiment 3 (token budget and generation mode): language of the answer-correct
                       chains of the four current-generation solvers, non-thinking against thinking at
                       16k (top), and Qwen3.5-4B/9B thinking at the low and the raised budget (bottom).

The chain measurements are code_switching_panel.csv, code_switching_main.csv and code_switching_budget.csv,
the judge measurements judge_language.csv and judge_language_v4flash.csv, all in analysis/data. The three
figures are written to analysis/figures.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import FIG_DIR, data, ensure_output_dirs  # noqa: E402

COL = {"target": "#009e73", "mixed": "#e69f00", "english": "#0072b2"}
LAB = {"target": "benchmark language", "mixed": "mixed", "english": "English"}
CLASSES = ["target", "mixed", "english"]
SHORT = {
    "aime2026": "AIME (EN)",
    "aime2026_pt": "AIME (PT)",
    "aime2026_tr": "AIME (TR)",
    "tubitak_math2026": "TÜBİTAK (TR)",
    "pt_exams_math": "PT exams",
}
# English AIME is left out of the non-English panels: every chain and judgment there is English.
NON_ENGLISH = ["aime2026_pt", "aime2026_tr", "tubitak_math2026", "pt_exams_math"]
MAIN_BENCHMARKS = ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026"]
SOLVERS = [
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
CURRENT_SOLVERS = [
    ("Q3.5-4B", "Q3.5-4B-Think"),
    ("Q3.5-9B", "Q3.5-9B-Think"),
    ("G4-E2B", "G4-E2B-Think"),
    ("G4-E4B", "G4-E4B-Think"),
]


def classify(d):
    d = d[d.n_letters.fillna(0) >= 20].copy()
    d["cls"] = np.where(d.share_target >= 0.8, "target", np.where(d.share_eng >= 0.8, "english", "mixed"))
    return d


def stacked(ax, groups, title, ylabel=None, wrap=False):
    """Stacked bars in labelled groups; groups is a list of (group label, [(bar label, dataframe), ...])."""
    x = 0
    ticks = []
    labels = []
    for gl, items in groups:
        start = x
        for name, s in items:
            if len(s) < 20:
                continue
            sh = s.cls.value_counts(normalize=True) * 100
            bottom = 0
            for c in CLASSES:
                v = sh.get(c, 0)
                ax.bar(x, v, bottom=bottom, color=COL[c], width=0.8, edgecolor="white", linewidth=0.4)
                if v >= 12:
                    ax.text(x, bottom + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=6, color="white")
                bottom += v
            ticks.append(x)
            labels.append(name)
            x += 1
        group_label = gl.replace(" (", "\n(") if wrap else gl
        ax.text(
            (start + x - 1) / 2,
            103,
            group_label,
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
            linespacing=1.0,
        )
        x += 0.9
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylim(0, 118 if wrap else 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=8, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)


def stacked_one(ax, items, title, ylabel=None, rot=90, fs=6.5):
    """One stacked bar per (label, dataframe) item; items with fewer than 20 rows leave a gap."""
    x = 0
    ticks = []
    labels = []
    for name, s in items:
        if len(s) < 20:
            x += 1
            continue
        sh = s.cls.value_counts(normalize=True) * 100
        bottom = 0
        for c in CLASSES:
            v = sh.get(c, 0)
            ax.bar(x, v, bottom=bottom, color=COL[c], width=0.78, edgecolor="white", linewidth=0.4)
            if v >= 10:
                ax.text(x, bottom + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color="white")
            bottom += v
        ticks.append(x)
        labels.append(name)
        x += 1
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=rot, fontsize=fs)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)


def legend(fig):
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in CLASSES]
    fig.legend(
        h, [LAB[c] for c in CLASSES], loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.04)
    )


def save(fig, name):
    out = FIG_DIR / name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print("wrote", out)


def experiment_1(panel, judge):
    jp = judge[(judge.source == "panel") & (judge.condition == "clean")]
    v4 = classify(pd.read_csv(data("judge_language_v4flash.csv")))
    v4 = v4[v4.condition == "clean"]
    v4["source"] = "panel"
    jp = pd.concat([jp, v4])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [4, 6]})
    solver_groups = [
        (
            SHORT[b],
            [
                ("4B", panel[(panel.benchmark == b) & (panel.solver == "Qwen3.5-4B")]),
                ("9B", panel[(panel.benchmark == b) & (panel.solver == "Qwen3.5-9B")]),
            ],
        )
        for b in NON_ENGLISH
    ]
    stacked(axes[0], solver_groups, "solution chains (Qwen3.5 thinking)", "share of chains / judgments (%)", wrap=True)
    judge_groups = [
        (SHORT[b], [(jn, jp[(jp.benchmark == b) & (jp.judge == jn)]) for jn in ["Qwen3.6", "V4-Flash", "R1-distill"]])
        for b in NON_ENGLISH
    ]
    stacked(axes[1], judge_groups, "judges' output on the clean controls")
    legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_language_exp1.png")


def experiment_2(main, judge):
    jm = judge[judge.source == "main"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2), gridspec_kw={"height_ratios": [5, 3.2]})
    for k, b in enumerate(MAIN_BENCHMARKS[1:]):
        items = [(sv, main[(main.benchmark == b) & (main.solver == sv)]) for sv in SOLVERS]
        stacked_one(axes[0, k], items, SHORT[b], "answer-correct chains per solver (%)" if k == 0 else None)
        axes[0, k].axvline(3.5, color="0.5", linewidth=0.6, linestyle=":")
        axes[0, k].text(1.5, 102, "earlier", ha="center", va="bottom", fontsize=6.5, color="0.3", clip_on=False)
        axes[0, k].text(6.5, 102, "current", ha="center", va="bottom", fontsize=6.5, color="0.3", clip_on=False)
        axes[0, k].set_title(axes[0, k].get_title(), fontsize=8, fontweight="bold", pad=14)
        judge_items = [
            (jn, jm[(jm.benchmark == b) & (jm.judge == jn)]) for jn in ["Qwen3.6", "Gemma4", "V4-Flash", "R1-distill"]
        ]
        stacked_one(axes[1, k], judge_items, SHORT[b], "judgments per judge (%)" if k == 0 else None, rot=0, fs=6.5)
    legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.5)
    save(fig, "fig_language_exp2.png")


def experiment_3(main, budget):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.4), gridspec_kw={"height_ratios": [5.5, 3.6]})
    for k, b in enumerate(MAIN_BENCHMARKS[1:]):
        items = []
        for nt, th in CURRENT_SOLVERS:
            items.append((nt + " non-think", main[(main.benchmark == b) & (main.solver == nt)]))
            items.append(
                (nt + " think", budget[(budget.benchmark == b) & (budget.solver == th) & (budget.budget == 16384)])
            )
        stacked_one(
            axes[0, k],
            items,
            SHORT[b],
            "generation mode at 16k, share of chains (%)" if k == 0 else None,
            rot=90,
            fs=6.2,
        )
        hi = 65536 if b == "aime2026" else 32768
        items = []
        for th in ["Q3.5-4B-Think", "Q3.5-9B-Think"]:
            name = th.replace("-Think", "")
            rows = budget[(budget.benchmark == b) & (budget.solver == th)]
            items.append((f"{name}\n16k", rows[rows.budget == 16384]))
            items.append((f"{name}\n{hi // 1024}k", rows[rows.budget == hi]))
        stacked_one(
            axes[1, k],
            items,
            SHORT[b],
            "budget ladder, thinking, share of chains (%)" if k == 0 else None,
            rot=0,
            fs=6.5,
        )
    legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.5)
    save(fig, "fig_language_exp3.png")


def main():
    ensure_output_dirs()
    panel = classify(pd.read_csv(data("code_switching_panel.csv")))
    judge = classify(pd.read_csv(data("judge_language.csv")))
    main_grid = classify(pd.read_csv(data("code_switching_main.csv")))
    budget = classify(pd.read_csv(data("code_switching_budget.csv")))
    experiment_1(panel, judge)
    experiment_2(main_grid, judge)
    experiment_3(main_grid, budget)


if __name__ == "__main__":
    main()
