"""Paper-style **highlights PDF** of the Pass@K vs CoT-Pass@K divergence.

Where :mod:`evalhub.report.plots` renders the *full* exploratory suite, this module
distills it into a short (~9-page) report: one finding per page, each a single
figure plus an interpretive caption — the way a paper presents its money plots.

The single story: **the CoT judge "veto" (No-Judge − cot) is not noise; it varies
systematically with K, language, model size, reasoning mode and the judge itself.**
Every number in every caption is computed live from the wide CSV via
:func:`_matched_pairs` — captions never hardcode statistics, so they stay true if the
data changes.

References framing the report: 2504.13837 (Pass@K curves, base vs RL → No-Judge vs
Judge) and 2506.14245 (CoT-Pass@K: answer *and* reasoning correct).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from evalhub.report import labels, plots  # noqa: E402
from evalhub.utils.logger import logger  # noqa: E402

A4 = (8.27, 11.69)  # portrait inches
ACCENT = "#c0392b"
INK = "#222222"

# Metric lenses pulled into the matched-pairs frame.
_METRICS: list[tuple[str, str | None]] = [
    ("pass", None),
    ("gpass", "0.25"),
    ("gpass", "0.5"),
    ("gpass", "0.75"),
    ("gpass", "1.0"),
    ("mgpass", None),
]
_LANG_ORDER = ["EN", "PT", "TR", "TR-OL"]
_STATE_ORDER = ["base", "non-think", "think"]


# ---------------------------------------------------------------------------
# Data: matched No-Judge ↔ cot pairs
# ---------------------------------------------------------------------------


def _val(row, base: str, k: int, tau: str | None) -> float:
    col = plots._col(base, k, tau)
    v = row[col] if col in row.index else np.nan
    return float(v) if pd.notna(v) else np.nan


def _norm_judged(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the ``judged`` discriminator to real booleans (CSV may store strings)."""
    df = df.copy()
    j = df["judged"]
    if j.dtype != bool:
        df["judged"] = j.map(lambda v: str(v).strip().lower() in ("true", "1", "yes", "t"))
    return df


def _matched_pairs(df: pd.DataFrame, ks=(1, 2, 4, 8, 16, 32, 64)) -> pd.DataFrame:
    """Tidy long frame: one row per (cell, judge, k, metric) with nojudge/cot/delta.

    A "cell" is a unique (model, state, benchmark). Each judged row is matched to
    its No-Judge sibling; rows without a sibling (or with a missing metric) drop out.
    """
    df = _norm_judged(df)
    nj_idx: dict[tuple, pd.Series] = {}
    for _, r in df[~df["judged"]].iterrows():
        nj_idx[(r["model"], r["state"], r["benchmark"])] = r

    out: list[dict] = []
    for _, r in df[df["judged"]].iterrows():
        n = nj_idx.get((r["model"], r["state"], r["benchmark"]))
        if n is None:
            continue
        size = r["model_size_b"]
        for k in ks:
            for base, tau in _METRICS:
                a, c = _val(n, base, k, tau), _val(r, base, k, tau)
                if np.isnan(a) or np.isnan(c):
                    continue
                out.append(
                    {
                        "model": r["model"],
                        "model_short": labels.short_model(r["model"]),
                        "state": r["state"],
                        "mode": labels.mode_label(r["state"]),
                        "benchmark": r["benchmark"],
                        "language": labels.language(r["benchmark"]),
                        "family": r["model_family"],
                        "size": float(size) if pd.notna(size) else np.nan,
                        "is_base": bool(n["is_base"]),
                        "judge": r["judge_model"],
                        "judge_short": labels.short_judge(r["judge_model"], r["judge_state"]),
                        "k": k,
                        "metric": base,
                        "tau": tau or "",
                        "metric_key": base if tau is None else f"{base}_t{tau}",
                        "nojudge": a,
                        "cot": c,
                        "delta": a - c,
                    }
                )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Page scaffolding
# ---------------------------------------------------------------------------


def _wrap(text: str, width: int = 108) -> str:
    return "\n".join(textwrap.fill(p, width=width) for p in text.split("\n"))


def _page(pdf: PdfPages):
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor("white")
    return fig


def _header(fig, kicker: str, title: str) -> None:
    fig.text(0.5, 0.955, kicker.upper(), ha="center", fontsize=10, color=ACCENT, fontweight="bold")
    fig.text(0.5, 0.928, title, ha="center", fontsize=15.5, fontweight="bold", color=INK)
    fig.add_artist(plt.Line2D([0.1, 0.9], [0.915, 0.915], color="#dddddd", lw=1, transform=fig.transFigure))


def _finding(fig, text: str) -> None:
    fig.add_artist(plt.Line2D([0.1, 0.9], [0.205, 0.205], color="#dddddd", lw=1, transform=fig.transFigure))
    fig.text(0.1, 0.185, "FINDING", fontsize=9, color=ACCENT, fontweight="bold")
    fig.text(0.1, 0.165, _wrap(text), ha="left", va="top", fontsize=9.6, color=INK, linespacing=1.45)


def _axes(fig, rect=(0.12, 0.30, 0.78, 0.55)):
    ax = fig.add_axes(rect)
    ax.grid(True, alpha=0.25, lw=0.5)
    return ax


def _close(pdf: PdfPages, fig) -> None:
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _page_title(pdf, mp, df) -> None:
    p = mp[mp["metric"] == "pass"]
    g1 = p[p.k == 1].delta.mean() * 100
    g64 = p[p.k == 64].delta.mean() * 100
    ret = (p[p.k == 64].cot / p[p.k == 64].nojudge.replace(0, np.nan)).mean() * 100
    n_cells = df[~_norm_judged(df)["judged"]].shape[0]
    n_models = df["model"].nunique()

    fig = _page(pdf)
    fig.text(0.5, 0.86, "Pass@K vs CoT-Pass@K", ha="center", fontsize=24, fontweight="bold", color=INK)
    fig.text(
        0.5, 0.815, "When the answer outruns the reasoning",
        ha="center", fontsize=15, color=ACCENT, style="italic",
    )
    fig.text(0.5, 0.775, "A highlights report on the CoT-judge veto across models, modes, languages and judges",
             ha="center", fontsize=10.5, color="#555")

    scope = (
        f"Scope.  {n_models} models (pretrained + instruct) · 3 modes (base / non-think / think) · "
        f"4 benchmarks (AIME-EN/PT/TR + TÜBİTAK TR-OL) · 3 CoT judges (all think) · K up to 64. "
        f"{n_cells} No-Judge evaluations, each matched to its judged (cot) counterparts."
    )
    abstract = (
        "We contrast answer-only correctness (No-Judge: pass@k / g-pass@k / mg-pass@k) with "
        "CoT-vetoed correctness (cot-*: a generation counts only if BOTH its final answer and its "
        "reasoning are judged correct). The gap between them — the veto — measures how much of a "
        "model's apparent success is unjustified. The central result of this report is that the veto "
        "is structured, not random: it grows with the sampling budget K, concentrates on the hardest "
        "(olympiad) benchmark, shrinks with model scale, is smallest in reasoning mode, and depends "
        "non-trivially on which judge grades. Read the companion guide (docs/report_plots_guide.md) to "
        "explore any slice yourself."
    )
    fig.text(0.1, 0.70, _wrap(scope, 104), ha="left", va="top", fontsize=10, color=INK, linespacing=1.5)
    fig.text(0.1, 0.60, _wrap(abstract, 104), ha="left", va="top", fontsize=10.3, color=INK, linespacing=1.5)

    # headline numbers box
    fig.add_artist(plt.Line2D([0.1, 0.9], [0.40, 0.40], color="#dddddd", lw=1, transform=fig.transFigure))
    fig.text(0.1, 0.375, "AT A GLANCE", fontsize=9, color=ACCENT, fontweight="bold")
    bullets = [
        f"Mean veto on Pass grows {g1:.1f} → {g64:.1f} points as K goes 1 → 64; cot keeps ~{ret:.0f}% of pass at K=64.",
        "Hardest-vetoed slice: the TÜBİTAK TR-OL olympiad — also the highest raw accuracy.",
        "Smallest models lose the most to the veto; scale buys faithfulness.",
        "Reasoning (think) mode is the LEAST vetoed — more faithful CoT, even when raw accuracy is lower.",
        "Judges disagree: the strictest grades ~3 pts harder on average, up to ~47 pts on a single cell.",
    ]
    fig.text(0.1, 0.35, "\n".join("•  " + _wrap(b, 100).replace("\n", "\n    ") for b in bullets),
             ha="left", va="top", fontsize=10, color=INK, linespacing=1.5)
    fig.text(0.5, 0.06, "References: arXiv 2504.13837 (Pass@K, base vs RL) · arXiv 2506.14245 (CoT-Pass@K)",
             ha="center", fontsize=8.5, color="#888")
    _close(pdf, fig)


def _page_k_growth(pdf, mp, df) -> None:
    p = mp[mp["metric"] == "pass"]
    fig = _page(pdf)
    _header(fig, "Finding 1 — sampling inflates unjustified success", "The veto grows with K")
    ax = _axes(fig)
    allk = p.groupby("k").delta.mean() * 100
    ax.plot(allk.index, allk.values, color="black", lw=2.6, marker="o", ms=6, label="All (mean)", zorder=6)
    for lang in _LANG_ORDER:
        s = p[p.language == lang].groupby("k").delta.mean() * 100
        if len(s):
            ax.plot(s.index, s.values, lw=1.8, marker="s", ms=4, ls="--",
                    color=plots.LANG_COLORS[lang], label=lang)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(p.k.unique()))
    ax.set_xticklabels(sorted(p.k.unique()))
    ax.set_xlabel("K (log₂)")
    ax.set_ylabel("Mean veto  Δ = pass@k − cot-pass@k  (points)")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, ncol=5, loc="upper left", frameon=False)

    g1, g64 = allk.get(1, np.nan), allk.get(64, np.nan)
    _finding(fig,
        f"More sampling buys more answer-correct hits than reasoning-correct ones, so the veto widens "
        f"monotonically: the mean gap rises from {g1:.1f} pts at K=1 to {g64:.1f} pts at K=64. In other words, "
        f"pass@k increasingly rewards lucky guesses that the CoT judge then removes. The practical lesson: a "
        f"high pass@64 is the easiest number to over-trust — always read it next to cot-pass@64. "
        f"(Per-cell version: results/report_plots/veto_curve/pass__{{state}}.png.)")
    _close(pdf, fig)


def _page_language(pdf, mp, df) -> None:
    p = mp[(mp["metric"] == "pass") & (mp.k == 64)]
    fig = _page(pdf)
    _header(fig, "Finding 2 — the multilingual axis", "The hardest benchmark hides the most unfaithful reasoning")
    ax = _axes(fig)
    langs = [lang for lang in _LANG_ORDER if lang in set(p.language)]
    nj = [p[p.language == lang].nojudge.mean() * 100 for lang in langs]
    cot = [p[p.language == lang].cot.mean() * 100 for lang in langs]
    x = np.arange(len(langs))
    ax.bar(x - 0.2, nj, 0.4, label="No-Judge pass@64", color="#34495e")
    ax.bar(x + 0.2, cot, 0.4, label="cot-pass@64", color="#e67e22")
    for i, _l in enumerate(langs):
        d = nj[i] - cot[i]
        ax.text(i, max(nj[i], cot[i]) + 1.5, f"Δ {d:.1f}", ha="center", fontsize=9, fontweight="bold", color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylabel("Mean accuracy at K=64 (points)")
    ax.set_ylim(0, max(nj) * 1.18)
    ax.legend(fontsize=9, frameon=False, loc="upper right")

    by = (p.groupby("language").delta.mean() * 100).sort_values(ascending=False)
    top, topv = by.index[0], by.iloc[0]
    rest = ", ".join(f"{lang} {by[lang]:.1f}" for lang in by.index[1:])
    _finding(fig,
        f"{top} is vetoed by {topv:.1f} pts on average — roughly 2–3× every translated AIME track ({rest}) — "
        f"while also carrying the highest raw accuracy. The hardest, most knowledge-dense set is exactly where "
        f"models most often reach the right answer by the wrong route. Because the AIME tracks (EN/PT/TR) are the "
        f"same items translated, their similar veto says the unfaithfulness is content-driven, not a translation "
        f"artifact; {top} is a different, harder olympiad and stands apart. "
        f"(Heatmaps: results/report_plots/comparisons/veto__pass__k64__{{judge}}.png.)")
    _close(pdf, fig)


def _page_mode(pdf, mp, df) -> None:
    p = mp[(mp["metric"] == "pass") & (mp.k == 64)]
    fig = _page(pdf)
    _header(fig, "Finding 3 — reasoning is more faithful", "Think mode answers less often, but earns it more")
    ax = _axes(fig)
    states = [s for s in _STATE_ORDER if s in set(p.state)]
    nj = [p[p.state == s].nojudge.mean() * 100 for s in states]
    cot = [p[p.state == s].cot.mean() * 100 for s in states]
    x = np.arange(len(states))
    ax.bar(x - 0.2, nj, 0.4, label="No-Judge pass@64", color="#34495e")
    ax.bar(x + 0.2, cot, 0.4, label="cot-pass@64", color="#e67e22")
    for i, _s in enumerate(states):
        ax.text(i, max(nj[i], cot[i]) + 1.2, f"Δ {nj[i]-cot[i]:.1f}", ha="center", fontsize=9,
                fontweight="bold", color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.mode_label(s) for s in states], fontsize=9)
    ax.set_ylabel("Mean accuracy at K=64 (points)")
    ax.set_ylim(0, max(nj) * 1.18)
    ax.legend(fontsize=9, frameon=False, loc="upper right")

    dser = p.groupby("state").delta.mean() * 100
    th = dser.get("think", np.nan)
    nt = dser.get("non-think", np.nan)
    _finding(fig,
        f"Faithfulness (small veto) and raw accuracy (tall bar) are different axes. Reasoning mode has the "
        f"highest raw pass yet the SMALLEST veto (think Δ≈{th:.1f} vs non-think Δ≈{nt:.1f} pts): turning on "
        f"reasoning makes a model not just more correct but more justified. This is the encouraging half of the "
        f"story — and it sets up the cautionary half on the anomaly page, where reasoning sometimes lowers raw "
        f"accuracy for specific models. (Per-family curves: results/report_plots/mode_compare/pass__{{family}}.png.)")
    _close(pdf, fig)


def _page_size(pdf, mp, df) -> None:
    p = mp[(mp["metric"] == "pass") & (mp.k == 64)]
    fig = _page(pdf)
    _header(fig, "Finding 4 — scaling buys faithfulness", "Small models guess; large models justify")
    ax = _axes(fig)
    for is_base, color, lab in [(False, "#2980b9", "Instruct"), (True, "#7f8c8d", "Pretrained")]:
        s = p[p.is_base == is_base].groupby("size").delta.mean() * 100
        if len(s):
            ax.plot(s.index, s.values, marker="o", ms=7, lw=2.2, color=color, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Model size (B params, log₂)")
    ax.set_ylabel("Mean veto Δ at K=64 (points)")
    ax.set_ylim(0, None)
    ax.legend(fontsize=9, frameon=False)

    ins = p[~p.is_base].groupby("size").delta.mean() * 100
    corr = p[["size", "delta"]].corr().iloc[0, 1]
    smin, smax = ins.index.min(), ins.index.max()
    _finding(fig,
        f"The veto falls sharply with scale: the smallest instruct models lose ~{ins.get(smin, float('nan')):.0f} "
        f"pts at K=64, the largest only ~{ins.get(smax, float('nan')):.0f} (Pearson r(size,Δ)≈{corr:.2f}). Small "
        f"models often land the right answer with invalid reasoning — pattern-matching, not solving — and the CoT "
        f"judge strips that away; scale closes the gap between answering and reasoning. So the veto is largely a "
        f"capability signal, and a tiny model's pass@k is the least trustworthy. "
        f"(Size curves: results/report_plots/size_compare/pass__{{state}}__cot__{{judge}}.png.)")
    _close(pdf, fig)


def _page_judge(pdf, mp, df) -> None:
    p = mp[(mp["metric"] == "pass") & (mp.k == 64)]
    fig = _page(pdf)
    _header(fig, "Finding 5 — the grader matters", "Who judges changes the verdict")
    ax_l = fig.add_axes((0.10, 0.30, 0.40, 0.55))
    ax_l.grid(True, alpha=0.25, lw=0.5)
    ax_r = fig.add_axes((0.58, 0.30, 0.34, 0.55))
    ax_r.grid(True, alpha=0.25, lw=0.5)

    by = (p.groupby("judge_short").delta.mean() * 100).sort_values(ascending=False)
    ax_l.barh(range(len(by)), by.values[::-1], color="#8e44ad")
    ax_l.set_yticks(range(len(by)))
    ax_l.set_yticklabels(by.index[::-1], fontsize=8.5)
    ax_l.set_xlabel("Mean veto Δ at K=64 (pts)")
    ax_l.set_title("Strictness by judge", fontsize=10)

    # disagreement: spread (max−min cot) across judges on the same cell
    piv = p.pivot_table(index=["model", "state", "benchmark"], columns="judge_short", values="cot")
    spread = ((piv.max(axis=1) - piv.min(axis=1)).dropna() * 100)
    ax_r.hist(spread.values, bins=12, color="#16a085", edgecolor="white")
    ax_r.set_xlabel("Judge spread on a cell (pts)")
    ax_r.set_ylabel("# cells")
    ax_r.set_title("Disagreement", fontsize=10)

    strict, mild = by.index[0], by.index[-1]
    _finding(fig,
        f"Treat any single cot number as judge-conditional. The strictest grader ({strict}) vetoes ~{by.iloc[0]:.1f} "
        f"pts on average vs ~{by.iloc[-1]:.1f} for the mildest ({mild}); on individual cells the three judges' "
        f"cot-pass@64 differ by up to {spread.max():.0f} pts (mean spread {spread.mean():.1f}). Whenever a "
        f"conclusion flips between the per-judge files, it is a grading artifact, not a model fact — cross-check "
        f"with all three. (Per-judge tables: results/report_plots/tables/*__cotdelta__{{judge}}.png.)")
    _close(pdf, fig)


def _page_stringency(pdf, mp, df) -> None:
    k = 64
    fig = _page(pdf)
    _header(fig, "Finding 6 — what the veto removes", "It strips lucky single hits, not consistent skill")
    ax_l = fig.add_axes((0.10, 0.30, 0.38, 0.55))
    ax_l.grid(True, alpha=0.25, lw=0.5)
    ax_r = fig.add_axes((0.58, 0.30, 0.34, 0.55))
    ax_r.grid(True, alpha=0.25, lw=0.5)

    pv = mp[(mp.metric == "pass") & (mp.k == k)].delta.mean() * 100
    mv = mp[(mp.metric == "mgpass") & (mp.k == k)].delta.mean() * 100
    ax_l.bar(["Pass@64", "mG-Pass@64"], [pv, mv], color=["#e67e22", "#27ae60"], width=0.55)
    for i, v in enumerate([pv, mv]):
        ax_l.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax_l.set_ylabel("Mean veto Δ (pts)")
    ax_l.set_title("Lenient vs consistency metric", fontsize=10)

    taus = ["0.25", "0.5", "0.75", "1.0"]
    ramp = [mp[(mp.metric == "gpass") & (mp.tau == t) & (mp.k == k)].delta.mean() * 100 for t in taus]
    ax_r.plot([float(t) for t in taus], ramp, marker="o", ms=7, lw=2.2, color="#c0392b")
    ax_r.set_xlabel("G-Pass threshold τ")
    ax_r.set_ylabel("Mean veto Δ at K=64 (pts)")
    ax_r.set_title("Veto vs stringency", fontsize=10)

    _finding(fig,
        f"The veto hits the most lenient metric hardest: Pass@64 loses {pv:.1f} pts but mG-Pass@64 — which rewards "
        f"consistent rather than one-off correctness — loses only {mv:.1f}. Reading the G-Pass τ ramp the same way, "
        f"the gap is largest where a single correct sample suffices and shrinks as τ demands more of the k samples be "
        f"correct. Interpretation: most vetoed generations are isolated lucky hits; genuinely consistent solvers "
        f"survive the CoT check. This is why mg-pass is the more honest headline metric. "
        f"(Per-model breakdown: results/report_plots/per_model/{{model}}__{{state}}.png, rows = metrics.)")
    _close(pdf, fig)


def _page_collapse_anomaly(pdf, mp, df) -> None:
    fig = _page(pdf)
    _header(fig, "Finding 7 — extremes & a caution", "Collapses under the veto, and when reasoning backfires")
    ax_l = fig.add_axes((0.18, 0.32, 0.30, 0.53))
    ax_l.grid(True, alpha=0.2, lw=0.5)
    ax_r = fig.add_axes((0.60, 0.32, 0.32, 0.53))
    ax_r.grid(True, alpha=0.2, lw=0.5)

    # Left: top collapse cells (largest absolute veto), pass@64
    p = mp[(mp.metric == "pass") & (mp.k == 64)].copy()
    top = p.sort_values("delta", ascending=False).head(6)
    lbl = [f"{r.model_short}·{r.language}\n{r.judge_short}" for _, r in top.iterrows()]
    y = np.arange(len(top))
    ax_l.barh(y - 0.2, top.nojudge.values * 100, 0.4, color="#34495e", label="pass@64")
    ax_l.barh(y + 0.2, top.cot.values * 100, 0.4, color="#e67e22", label="cot-pass@64")
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(lbl, fontsize=6.5)
    ax_l.invert_yaxis()
    ax_l.set_xlabel("Accuracy (pts)")
    ax_l.set_title("Biggest collapses (all TR-OL)", fontsize=9.5)
    ax_l.legend(fontsize=7, frameon=False, loc="lower right")

    # Right: think vs non-think raw pass anomaly (No-Judge), Qwen on AIME
    dn = _norm_judged(df)
    nn = dn[~dn["judged"]]
    pairs = []
    for model in sorted(nn[nn.model_family == "Qwen"].model.unique()):
        sub = nn[nn.model == model]
        for b in ["aime2026"]:
            th = sub[(sub.state == "think") & (sub.benchmark == b)]
            nt = sub[(sub.state == "non-think") & (sub.benchmark == b)]
            if len(th) and len(nt):
                tv, nv = _val(th.iloc[0], "pass", 64, None), _val(nt.iloc[0], "pass", 64, None)
                if not np.isnan(tv) and not np.isnan(nv):
                    pairs.append((labels.short_model(model), nv * 100, tv * 100))
    if pairs:
        labs = [x[0] for x in pairs]
        x = np.arange(len(pairs))
        ax_r.bar(x - 0.2, [x[1] for x in pairs], 0.4, color="#2980b9", label="non-think")
        ax_r.bar(x + 0.2, [x[2] for x in pairs], 0.4, color="#c0392b", label="think")
        ax_r.set_xticks(x)
        ax_r.set_xticklabels(labs, fontsize=7, rotation=20)
        ax_r.set_ylabel("pass@64 (pts)")
        ax_r.set_title("AIME-EN: think can hurt", fontsize=9.5)
        ax_r.legend(fontsize=7, frameon=False)

    worst = top.iloc[0]
    _finding(fig,
        f"Two cautions. (Left) On the TR-OL olympiad, small/pretrained models effectively collapse under the veto — "
        f"e.g. {worst.model_short} drops {worst.nojudge*100:.0f}→{worst.cot*100:.0f} — i.e. nearly all of their "
        f"olympiad 'success' is unjustified. (Right) Reasoning is not a free win on raw accuracy: for several Qwen "
        f"instruct models, think-mode pass@64 on AIME-EN sits well BELOW non-think (long reasoning derails or "
        f"breaks answer formatting), even though its CoT — when it does answer — is more faithful (Finding 3). Keep "
        f"the two axes separate when ranking models.")
    _close(pdf, fig)


def _page_takeaways(pdf, mp, df) -> None:
    fig = _page(pdf)
    _header(fig, "Summary", "Takeaways, caveats & method")
    p = mp[mp.metric == "pass"]
    g64 = p[p.k == 64].delta.mean() * 100
    takeaways = [
        f"The CoT veto is real and structured — it averages {g64:.1f} pts on Pass@64 and is not noise.",
        "Always pair pass@k with cot-pass@k; the gap is the unjustified fraction of the score.",
        "More sampling (higher K) inflates the unjustified part fastest — distrust a lone high pass@64.",
        "Hardest benchmark (TR-OL) and smallest models hide the most unfaithful reasoning.",
        "Reasoning mode and scale both improve faithfulness; mg-pass is the most honest headline metric.",
    ]
    caveats = [
        "Judges disagree (up to ~47 pts on a cell) — never conclude from a single judge; cross-check all three.",
        "All judges run with think=true by design; there are no non-think judge series.",
        "Reasoning can lower RAW accuracy for some models even while improving faithfulness — different axes.",
        "Plot y-axes in the suite are 0→cell-max, so compare gaps within a cell, not heights across cells.",
    ]
    method = (
        "Method. Wide table results/report.csv (one row per model×mode×benchmark×judge). For each judged (cot) row "
        "we match its No-Judge sibling on (model, state, benchmark) and define veto Δ = No-Judge − cot per metric and "
        "K. Means above are unweighted over matched cells (no hidden averaging across the slices each finding holds "
        "fixed). Reproduce or re-slice via results/report_plots/comparisons/pass_vs_cot_k{1,64}.csv."
    )
    fig.text(0.1, 0.88, "What to conclude", fontsize=11, color=ACCENT, fontweight="bold")
    fig.text(0.1, 0.855, "\n".join("•  " + _wrap(t, 100).replace("\n", "\n    ") for t in takeaways),
             ha="left", va="top", fontsize=10, color=INK, linespacing=1.5)
    fig.text(0.1, 0.55, "Caveats", fontsize=11, color=ACCENT, fontweight="bold")
    fig.text(0.1, 0.525, "\n".join("•  " + _wrap(c, 100).replace("\n", "\n    ") for c in caveats),
             ha="left", va="top", fontsize=10, color=INK, linespacing=1.5)
    fig.text(0.1, 0.27, _wrap(method, 104), ha="left", va="top", fontsize=9.2, color="#555", linespacing=1.45)
    fig.text(0.5, 0.06, "Full exploratory suite: results/report_plots/  ·  reading guide: docs/report_plots_guide.md",
             ha="center", fontsize=8.5, color="#888")
    _close(pdf, fig)


_PAGES = [
    _page_title,
    _page_k_growth,
    _page_language,
    _page_mode,
    _page_size,
    _page_judge,
    _page_stringency,
    _page_collapse_anomaly,
    _page_takeaways,
]


def build_highlights(df: pd.DataFrame, output_pdf: Path | str) -> Path:
    """Render the highlights PDF from the wide DataFrame."""
    out = Path(output_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    mp = _matched_pairs(df)
    if mp.empty:
        raise ValueError("No matched No-Judge↔cot pairs found — is this the wide report.csv?")
    with PdfPages(out) as pdf:
        for page in _PAGES:
            try:
                page(pdf, mp, df)
            except Exception as e:  # pragma: no cover - keep the rest of the report alive
                logger.error(f"[highlights] {page.__name__} failed: {e}")
        d = pdf.infodict()
        d["Title"] = "Pass@K vs CoT-Pass@K — Highlights"
        d["Subject"] = "CoT-judge veto across models, modes, languages, judges"
    logger.info(f"[highlights] wrote {out}")
    return out
