"""Decomposition of Pass@64 - CoT-Pass@64 into judge verdicts and judge max-token stops (fig_judge_capping.png).

Every judgment in verdicts.csv carries n_missing_capped, the number of the
three judge generations that produced no verdict because they hit the judge's
output cap. A capped generation counts as a non-approval, so it can veto a
solution the judge never rejected. CoT-Pass@k is therefore recomputed twice per
cell: observed (capped generations count as non-approval, which is what the
metric does) and counterfactual (capped generations count as approval). The
part of the difference that survives the counterfactual comes from verdicts;
the rest is produced by the stops. Pass@k and the observed CoT-Pass@k are
rebuilt from verdicts.csv with the unbiased estimator and checked against
gap_curve.csv. All three inputs (report.csv, verdicts.csv, gap_curve.csv) live
in analysis/data; the figure is written to analysis/figures.

Scope: the 16,384-token solver budget, the two judges Gemma4 and Qwen3.6 (both
in thinking mode, both at a 16,384-token judge budget), the four 64-sample
benchmarks, and the fully crossed design of the four instruct solvers in both
solver modes. A judge must have scored every correct solution of a cell, and
both judges must have at least MIN_JUDGED judged solutions in it; any benchmark
that is not complete for every solver in every mode is dropped.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import FIG_DIR, data, ensure_output_dirs
from matplotlib.patches import Patch
from scipy.special import comb

matplotlib.use("Agg")

OUT = FIG_DIR / "fig_judge_capping.png"

JUDGES = ["Gemma4", "Qwen3.6"]
JUDGE_LABEL = {"Gemma4": "Gemma4-26B", "Qwen3.6": "Qwen3.6-35B"}
COLOR = {"Gemma4": "#8172b3", "Qwen3.6": "#ee854a"}
# The state axis is the solver's mode; both judges always ran in thinking mode.
STATE_LABEL = {"base": "base checkpoints", "non-think": "solver: non-thinking", "think": "solver: thinking"}
KS = [1, 2, 4, 8, 16, 32, 64]
BUDGET = 16384
MIN_JUDGED = 30
BENCHMARKS = ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026"]

# Fully crossed design: every solver is measured on every benchmark in both
# modes and every cell is judged by both judges, so the mode, family and judge
# contrasts all run over the same set of solver outputs.
MAIN_SOLVERS = ["gemma-4-E2B-it", "gemma-4-E4B-it", "Qwen3.5-4B", "Qwen3.5-9B"]
MAIN_STATES = ["non-think", "think"]

CELL = ["solver_model", "state", "benchmark", "judge_short"]
RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
}


def pass_at_k(n, c, k):
    """Unbiased Pass@k over a task with n samples of which c count as correct."""
    n = np.asarray(n, float)
    c = np.asarray(c, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n - c < k, 1.0, 1.0 - comb(n - c, k) / comb(n, k))
    return np.where(k > n, np.nan, out)


def load_cells():
    """Per task_id counts for the three curves, restricted to the fully crossed design."""
    r = pd.read_csv(data("report.csv"))
    v = pd.read_csv(data("verdicts.csv"))

    v = v[
        (v.judge_short.isin(JUDGES)) & (v.solver_max_tokens == BUDGET) & (~v.solver_model.str.startswith("Qwen2.5"))
    ].copy()
    assert set(v.judge_state.unique()) == {"think"}, v.judge_state.unique()
    # Both judges must share one output budget: the judge's own budget drives its
    # veto rate, so two budgets would confound the judge contrast.
    jmax = set(v.judge_max_tokens.dropna().unique())
    assert jmax == {BUDGET}, f"judge budgets are not uniform: {sorted(jmax)}"
    # counterfactual: the capped generations would have approved
    v["approved_cf"] = (v.n_yes + v.n_missing_capped.fillna(0)) >= 2

    meta = (
        r[r.solver_max_tokens == BUDGET]
        .drop_duplicates(["model", "state", "benchmark"])[["model", "state", "benchmark", "n_samples", "n_tasks"]]
        .rename(columns={"model": "solver_model"})
    )
    per = (
        v.groupby(CELL + ["task_id"])
        .agg(c=("answer_correct", "size"), c_obs=("approved_maj", "sum"), c_cf=("approved_cf", "sum"))
        .reset_index()
        .merge(meta, on=["solver_model", "state", "benchmark"], how="left")
    )

    per = per[per.benchmark.isin(BENCHMARKS)]
    per = per[per.n_samples >= max(KS)]  # the k axis reaches 64
    n_judged = per.groupby(CELL).c.sum().rename("n_judged").reset_index()
    per = per.merge(n_judged, on=CELL)

    # Completeness guard: a judge must have scored every correct solution in the
    # cell, otherwise CoT-Pass@k is understated by exactly the solutions the judge
    # never saw. The count of correct solutions comes from the unjudged row of
    # report.csv, which exists for every solver run.
    truth = (
        r[(r.solver_max_tokens == BUDGET) & (~r.judged.astype(bool))]
        .drop_duplicates(["model", "state", "benchmark"])
        .set_index(["model", "state", "benchmark"])
        .true_count
    )
    key = pd.MultiIndex.from_frame(per[["solver_model", "state", "benchmark"]])
    per["n_correct"] = truth.reindex(key).to_numpy()
    incomplete = per[per.n_judged < per.n_correct].drop_duplicates(CELL)
    for _, x in incomplete.iterrows():
        print(
            f"[scope] dropped {x.solver_model} {x.state} {x.benchmark} ({x.judge_short}): "
            f"{int(x.n_judged)}/{int(x.n_correct)} correct solutions judged"
        )
    per = per[per.n_judged >= per.n_correct.fillna(0)]

    per = per[per.groupby(CELL[:3]).judge_short.transform("nunique") == 2]
    per = per[per.groupby(CELL[:3]).n_judged.transform("min") >= MIN_JUDGED]

    main_per = per[per.solver_model.isin(MAIN_SOLVERS) & per.state.isin(MAIN_STATES)]
    return crossed(main_per, "main", MAIN_SOLVERS, MAIN_STATES)


def crossed(p, name, solvers, states):
    """Keep the largest fully crossed design: drop any benchmark that is not present
    for every solver in every state, and say which."""
    have = p.drop_duplicates(["solver_model", "benchmark", "state"])
    need = len(solvers) * len(states)
    ok = [b for b in BENCHMARKS if (have.benchmark == b).sum() == need]
    for b in BENCHMARKS:
        if b not in ok:
            print(
                f"[scope] {name}: dropping {b}: {(have.benchmark == b).sum()}/{need} "
                f"solver x state cells survive the completeness guard"
            )
    assert ok, f"{name}: no benchmark is complete"
    p = p[p.benchmark.isin(ok)]
    got = p.drop_duplicates(["solver_model", "benchmark", "state"])
    assert len(got) == len(ok) * need, f"{name} design incomplete"
    return p


def curves(per):
    """Pass@k, observed CoT-Pass@k and counterfactual CoT-Pass@k per cell and k."""
    out = []
    for k in KS:
        t = per.copy()
        for src, dst in [("c", "pass"), ("c_obs", "obs"), ("c_cf", "cf")]:
            t[dst] = pass_at_k(t.n_samples, t[src], k)
        a = (
            t.groupby(CELL)
            .agg(n_tasks=("n_tasks", "first"), **{c: (c, "sum") for c in ["pass", "obs", "cf"]})
            .reset_index()
        )
        for c in ["pass", "obs", "cf"]:
            a[c] = a[c] / a.n_tasks  # tasks with no correct solution contribute 0
        a["k"] = k
        out.append(a)
    return pd.concat(out, ignore_index=True)


def check_reconstruction(cur):
    """Pass@k and the observed CoT-Pass@k must reproduce gap_curve.csv."""
    g = pd.read_csv(data("gap_curve.csv"))
    g = g[g.solver_max_tokens == BUDGET]
    m = cur.merge(g, on=CELL + ["k"])
    dp = (m["pass"] - m.pass_at_k).abs().max()
    dc = (m.obs - m.cot_pass_at_k).abs().max()
    print(f"[check] reconstruction vs gap_curve.csv on {len(m)} rows: max|Pass@k|={dp:.2e}  max|CoT-Pass@k|={dc:.2e}")
    assert dp < 1e-9 and dc < 1e-9, "reconstruction does not match gap_curve.csv"


def build_decomposition(cur, states, out, figsize):
    """One bar per judge x solver-mode cell mean at k=64: the solid part is what
    survives when stopped judgments are credited as approvals, the hatched part
    is what those stops produce."""
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=figsize)

    order = [s for s in ["think", "non-think", "base"] if s in states]
    rows = []
    ys = []
    heads = []
    y = 0.0
    for st in order:
        heads.append((st, y + 0.85))
        for j in JUDGES:
            s = cur[(cur.judge_short == j) & (cur.state == st) & (cur.k == 64)]
            delta = 100 * (s["pass"] - s.obs).mean()
            verdicts = 100 * (s["pass"] - s.cf).mean()
            rows.append((st, j, delta, verdicts))
            ys.append(y)
            y -= 1.0
        y -= 0.85

    for (_st, j, delta, verdicts), yy in zip(rows, ys, strict=False):
        c = COLOR[j]
        ax.barh(yy, verdicts, height=0.6, color=c, zorder=3)
        ax.barh(
            yy, delta - verdicts, left=verdicts, height=0.6, color="white", edgecolor=c, hatch="////", lw=0.55, zorder=3
        )
        share = 100 * (delta - verdicts) / delta if delta > 1e-9 else 0.0
        lbl = f"{delta:.1f}" + (f"   {share:.0f}% from stops" if share >= 20 else "")
        ax.text(delta + 0.2, yy, lbl, fontsize=6.6, va="center", ha="left", color="0.25")

    ax.set_yticks(ys)
    ax.set_yticklabels([JUDGE_LABEL[j] for _, j, _, _ in rows], fontsize=7)
    for st, yy in heads:
        ax.text(0.0, yy, STATE_LABEL[st], fontsize=6.8, style="italic", color="0.35", va="center", ha="left")
    ax.set_xlabel("Pass@$64$ $-$ CoT-Pass@$64$ (points)", fontsize=7.6)
    ax.set_xlim(0, 9.8)
    ax.set_ylim(min(ys) - 0.6, max(heads, key=lambda h: h[1])[1] + 0.55)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.4, color="0.92", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)

    handles = [
        Patch(facecolor="0.45", lw=0, label="the judge's verdicts"),
        Patch(
            facecolor="white", edgecolor="0.45", hatch="////", lw=0.55, label="judgments stopped at the max-token limit"
        ),
    ]
    fig.legend(
        handles=handles,
        fontsize=6.4,
        frameon=False,
        ncol=1,
        loc="lower left",
        bbox_to_anchor=(0.10, 0.005),
        handlelength=1.3,
        handletextpad=0.45,
        labelspacing=0.2,
        borderpad=0.05,
    )
    fig.tight_layout(pad=0.35, rect=(0, 0.115, 1, 1))
    fig.savefig(out, dpi=300)
    plt.close(fig)


def report(per, cur, states):
    """Console summary: the difference at k = 1, 16, 64 and the share of it produced by judge stops."""
    cells = per.drop_duplicates(CELL)
    print(
        f"\nscope: {BUDGET}-token solver budget, judges {' + '.join(JUDGES)} (both in thinking mode), "
        f"paired, n_judged>={MIN_JUDGED}, k reaches 64"
    )
    print(f"design: {sorted(cells.solver_model.unique())}")
    print(f"      x {sorted(cells.benchmark.unique())} x {states}")
    print(f"      = {cells.groupby('state').size().div(len(JUDGES)).astype(int).to_dict()} cells per panel, complete")
    print("\nPass@k - CoT-Pass@k and the share of it produced by judge capping (state = solver mode)")
    for j in JUDGES:
        for st in states:
            s = cur[(cur.judge_short == j) & (cur.state == st)]
            row = []
            for k in [1, 16, 64]:
                m = s[s.k == k]
                delta = (m["pass"] - m.obs).mean()
                cap = (m["cf"] - m.obs).mean()
                share = 100 * cap / delta if delta > 1e-9 else float("nan")
                row.append(f"k={k:<2d} delta={100 * delta:5.2f}pt cap={share:5.1f}%")
            print(f"  {j:8s} {st:10s} " + "   ".join(row))


def main():
    ensure_output_dirs()
    per = load_cells()
    cur = curves(per)
    check_reconstruction(cur)
    build_decomposition(cur, MAIN_STATES, OUT, figsize=(3.4, 2.35))
    report(per, cur, MAIN_STATES)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
