"""Pass@K vs CoT-Pass@K visualisation suite, rendered from the wide master CSV.

Two reference papers frame everything here:

* **2504.13837** — Pass@K curves over K (log-x), base vs RL. Here the contrast is
  **No-Judge** (``judge_model`` empty → ``pass@k`` / ``g-pass@k`` / ``mg-pass@k``)
  vs **Judge** (``judge_model`` set → the *cot* family).
* **2506.14245** — introduces **CoT-Pass@K**: a generation only counts when both the
  final answer *and* its reasoning are correct. Our judged rows are exactly that.

So the through-line of every figure is: **how much does the CoT veto move the
metric, per model, per benchmark, per language** — never averaged away.

Design rules carried from the spec:

* The judge is **always** ``think`` (any non-think judge label is a migration
  mislabel and never appears in the data; see ``project_judge_always_think``).
* **Y axis runs 0 → the cell's own max** (not a fixed 0–100%), so small No-Judge↔cot
  gaps stay visible. The 0 baseline is kept so absolute rates remain readable.
* X axis is K on a log₂ scale.

The whole suite is driven by :func:`render_all`, which fans out over the six
metric specs (Pass, four G-Pass τ, mG-Pass) × the three modes (base / non-think /
think) and writes PNGs into ``results/report_plots/<family>/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never needs a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from evalhub.report import labels  # noqa: E402
from evalhub.utils.logger import logger  # noqa: E402

sns.set_theme(style="whitegrid", context="paper")

# ---------------------------------------------------------------------------
# Static ordering / styling
# ---------------------------------------------------------------------------

# The six metric "lenses". key -> filename stem; base/tau pick the wide columns.
METRIC_SPECS: list[dict] = [
    {"key": "pass", "base": "pass", "tau": None, "nj": "Pass@K", "cot": "CoT-Pass@K"},
    {"key": "gpass_t0.25", "base": "gpass", "tau": "0.25", "nj": "G-Pass@K τ=0.25", "cot": "CoT-G-Pass@K τ=0.25"},
    {"key": "gpass_t0.5", "base": "gpass", "tau": "0.5", "nj": "G-Pass@K τ=0.5", "cot": "CoT-G-Pass@K τ=0.5"},
    {"key": "gpass_t0.75", "base": "gpass", "tau": "0.75", "nj": "G-Pass@K τ=0.75", "cot": "CoT-G-Pass@K τ=0.75"},
    {"key": "gpass_t1.0", "base": "gpass", "tau": "1.0", "nj": "G-Pass@K τ=1.0", "cot": "CoT-G-Pass@K τ=1.0"},
    {"key": "mgpass", "base": "mgpass", "tau": None, "nj": "mG-Pass@K", "cot": "CoT-mG-Pass@K"},
]

STATE_ORDER = ["base", "non-think", "think"]
BENCH_ORDER = ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026"]
LANG_COLORS = {"EN": "#1f77b4", "PT": "#2ca02c", "TR": "#d62728", "TR-OL": "#9467bd"}
MODE_COLORS = {"base": "#7f7f7f", "non-think": "#1f77b4", "think": "#d62728"}
NJ_COLOR = "black"

# Matches intermediate RL checkpoint model names produced by scan.py.
_STEP_MODEL_RE = re.compile(r"^(.+?)@step(\d+)$")


def _bench_rank(b: str) -> int:
    return BENCH_ORDER.index(b) if b in BENCH_ORDER else 99


def _state_rank(s: str) -> int:
    return STATE_ORDER.index(s) if s in STATE_ORDER else 99


# ---------------------------------------------------------------------------
# Wide-column access
# ---------------------------------------------------------------------------


def _col(base: str, k: int, tau: str | None) -> str:
    if base == "pass":
        return f"pass@{k}"
    if base == "mgpass":
        return f"mgpass@{k}"
    return f"gpass@{k}_t{tau}"


def series(row, base: str, tau: str | None, ks: list[int]) -> list[tuple[int, float]]:
    """Extract the sorted ``(k, value)`` series for one metric from a wide row."""
    out: list[tuple[int, float]] = []
    for k in ks:
        col = _col(base, k, tau)
        if col in row.index:
            v = row[col]
            if pd.notna(v):
                out.append((k, float(v)))
    return out


def _value(row, base: str, k: int, tau: str | None) -> float | None:
    col = _col(base, k, tau)
    if col not in row.index:
        return None
    v = row[col]
    return float(v) if pd.notna(v) else None


def _k_axis(df: pd.DataFrame) -> list[int]:
    ks = sorted(int(c.split("@")[1]) for c in df.columns if c.startswith("pass@"))
    return ks or [1, 2, 4, 8, 16, 32, 64, 128]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _judge_style(df: pd.DataFrame) -> dict[str, tuple]:
    judges = sorted(df.loc[df["judged"], "judge_model"].dropna().unique())
    palette = sns.color_palette("Set1", n_colors=max(3, len(judges)))
    return {jm: palette[i] for i, jm in enumerate(judges)}


def _judge_state(df: pd.DataFrame, jm: str) -> str:
    s = df.loc[df["judge_model"] == jm, "judge_state"].dropna()
    return s.iloc[0] if len(s) else "think"


def _judge_handles(df: pd.DataFrame, style: dict[str, tuple]) -> list[Line2D]:
    handles = [Line2D([0], [0], color=NJ_COLOR, lw=2.2, marker="o", ms=5, label="No-Judge · pass@k")]
    for jm, c in style.items():
        handles.append(
            Line2D([0], [0], color=c, lw=1.7, ls="--", marker="s", ms=4,
                   label=f"cot · {labels.short_judge(jm, _judge_state(df, jm))}")
        )
    return handles


def _lang_handles() -> list[Line2D]:
    return [Line2D([0], [0], color=c, lw=2, marker="o", label=lang) for lang, c in LANG_COLORS.items()]


def _mode_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=MODE_COLORS[s], lw=2, marker="o", label=labels.mode_label(s))
        for s in STATE_ORDER
    ]


def _setup_ax(ax, ymax: float, ks: list[int]) -> None:
    """Log₂ K on x; **0 → cell-max** on y (per spec, so precision stays visible)."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=6.5)
    ax.set_xlim(min(ks) * 0.85, max(ks) * 1.15)
    ax.set_ylim(0, (ymax * 1.08) if ymax > 0 else 1.0)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=6.5)


def _ordered_models(df: pd.DataFrame) -> list[str]:
    u = df[["model", "model_family", "model_size_b", "is_base"]].drop_duplicates().copy()
    u["sz"] = u["model_size_b"].fillna(0.0)
    u = u.sort_values(["model_family", "sz", "is_base", "model"])
    return list(u["model"])


def _ordered_model_modes(df: pd.DataFrame) -> list[tuple[str, str]]:
    u = df[["model", "state", "model_family", "model_size_b", "is_base"]].drop_duplicates().copy()
    u["sz"] = u["model_size_b"].fillna(0.0)
    u["st"] = u["state"].map(_state_rank)
    u = u.sort_values(["model_family", "sz", "is_base", "model", "st"])
    return list(zip(u["model"], u["state"], strict=False))


def mm_label(model: str, state: str) -> str:
    sm = labels.short_model(model)
    if state == "base":
        return sm
    return f"{sm}·{ {'non-think': 'NT', 'think': 'TH'}.get(state, state) }"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _finish(fig, suptitle: str, handles: list[Line2D] | None, path: Path) -> Path:
    if handles:
        fig.legend(
            handles=handles, loc="lower center", ncol=min(len(handles), 6),
            fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01),
        )
    fig.suptitle(suptitle, y=1.0, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04 if handles else 0.0, 1, 0.97])
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Cell painters (return cell ymax, or None when nothing was drawn)
# ---------------------------------------------------------------------------


def _cell_judge_effect(ax, sub: pd.DataFrame, spec: dict, style: dict, ks: list[int]) -> float | None:
    if sub.empty:
        return None
    ymax, drew = 0.0, False
    for _, r in sub[~sub["judged"]].iterrows():
        pts = series(r, spec["base"], spec["tau"], ks)
        if pts:
            xs, ys = zip(*pts, strict=False)
            ax.plot(xs, ys, color=NJ_COLOR, lw=2.2, marker="o", ms=4, zorder=6)
            ymax, drew = max(ymax, max(ys)), True
    for _, r in sub[sub["judged"]].sort_values("judge_model").iterrows():
        pts = series(r, spec["base"], spec["tau"], ks)
        if pts:
            xs, ys = zip(*pts, strict=False)
            ax.plot(xs, ys, color=style.get(r["judge_model"], "gray"), lw=1.7, ls="--",
                    marker="s", ms=3, alpha=0.95, zorder=4)
            ymax, drew = max(ymax, max(ys)), True
    if not drew:
        return None
    _setup_ax(ax, ymax, ks)
    return ymax


def _cell_langs(ax, sub: pd.DataFrame, spec: dict, ks: list[int]) -> float | None:
    if sub.empty:
        return None
    ymax, drew = 0.0, False
    for _, r in sub.assign(_r=sub["benchmark"].map(_bench_rank)).sort_values("_r").iterrows():
        pts = series(r, spec["base"], spec["tau"], ks)
        if pts:
            xs, ys = zip(*pts, strict=False)
            lang = labels.language(r["benchmark"])
            ax.plot(xs, ys, color=LANG_COLORS.get(lang, "gray"), lw=1.8, marker="o", ms=3)
            ymax, drew = max(ymax, max(ys)), True
    if not drew:
        return None
    _setup_ax(ax, ymax, ks)
    return ymax


def _cell_sizes(ax, sub: pd.DataFrame, spec: dict, ks: list[int], judge: str | None) -> float | None:
    """One curve per model, coloured by size; ``judge`` overlays cot (dashed)."""
    nj = sub[~sub["judged"]]
    if nj.empty:
        return None
    models = _ordered_models(nj)
    sizes = [labels.model_size_b(m) or 0.0 for m in models]
    lo, hi = (min(sizes), max(sizes)) if sizes else (0.0, 1.0)
    cmap = plt.cm.viridis
    ymax, drew = 0.0, False
    for m in models:
        t = 0.5 if hi == lo else (((labels.model_size_b(m) or 0.0) - lo) / (hi - lo))
        color = cmap(0.12 + 0.76 * t)
        rn = nj[nj["model"] == m]
        if not rn.empty:
            pts = series(rn.iloc[0], spec["base"], spec["tau"], ks)
            if pts:
                xs, ys = zip(*pts, strict=False)
                ax.plot(xs, ys, color=color, lw=1.8, marker="o", ms=3, label=labels.short_model(m))
                ymax, drew = max(ymax, max(ys)), True
        if judge is not None:
            rc = sub[(sub["model"] == m) & (sub["judge_model"] == judge)]
            if not rc.empty:
                pts = series(rc.iloc[0], spec["base"], spec["tau"], ks)
                if pts:
                    xs, ys = zip(*pts, strict=False)
                    ax.plot(xs, ys, color=color, lw=1.4, ls="--", marker="s", ms=2.5, alpha=0.9)
                    ymax = max(ymax, max(ys))
    if not drew:
        return None
    _setup_ax(ax, ymax, ks)
    ax.legend(fontsize=5, loc="upper left", framealpha=0.55, handlelength=1.2)
    return ymax


def _cell_modes(ax, sub: pd.DataFrame, spec: dict, ks: list[int], cot_judge: str | None) -> float | None:
    """Overlay pretrained / non-think / think (No-Judge solid; optional cot dashed)."""
    if sub.empty:
        return None
    ymax, drew = 0.0, False
    for st in STATE_ORDER:
        rn = sub[(sub["state"] == st) & (~sub["judged"])]
        if not rn.empty:
            pts = series(rn.iloc[0], spec["base"], spec["tau"], ks)
            if pts:
                xs, ys = zip(*pts, strict=False)
                ax.plot(xs, ys, color=MODE_COLORS[st], lw=1.9, marker="o", ms=3)
                ymax, drew = max(ymax, max(ys)), True
        if cot_judge is not None:
            rc = sub[(sub["state"] == st) & (sub["judge_model"] == cot_judge)]
            if not rc.empty:
                pts = series(rc.iloc[0], spec["base"], spec["tau"], ks)
                if pts:
                    xs, ys = zip(*pts, strict=False)
                    ax.plot(xs, ys, color=MODE_COLORS[st], lw=1.4, ls="--", marker="s", ms=2.5, alpha=0.85)
                    ymax = max(ymax, max(ys))
    if not drew:
        return None
    _setup_ax(ax, ymax, ks)
    return ymax


def _cell_veto(ax, sub: pd.DataFrame, spec: dict, style: dict, ks: list[int]) -> float | None:
    """Δ(k) = No-Judge − cot per judge: how much the veto bites as K grows."""
    nj = sub[~sub["judged"]]
    if nj.empty:
        return None
    base_pts = dict(series(nj.iloc[0], spec["base"], spec["tau"], ks))
    if not base_pts:
        return None
    ymax, drew = 0.0, False
    for _, r in sub[sub["judged"]].sort_values("judge_model").iterrows():
        cot = dict(series(r, spec["base"], spec["tau"], ks))
        xs = [k for k in ks if k in base_pts and k in cot]
        ys = [base_pts[k] - cot[k] for k in xs]
        if xs:
            ax.plot(xs, ys, color=style.get(r["judge_model"], "gray"), lw=1.7, ls="--",
                    marker="s", ms=3, alpha=0.95)
            ymax, drew = max(ymax, max(ys)), True
    if not drew:
        return None
    ax.axhline(0, color="#999", lw=0.6)
    _setup_ax(ax, ymax, ks)
    return ymax


# ---------------------------------------------------------------------------
# Generic curve-matrix builder (rows × cols of cells)
# ---------------------------------------------------------------------------


def _grid(
    df: pd.DataFrame, *, rows: list, cols: list, row_field, col_field,
    painter, row_label, col_label, suptitle: str, handles, path: Path,
    cell_w: float = 2.9, cell_h: float = 2.3,
) -> Path | None:
    nr, nc = len(rows), len(cols)
    if nr == 0 or nc == 0:
        return None
    fig, axes = plt.subplots(nr, nc, figsize=(cell_w * nc + 0.6, cell_h * nr + 0.8), squeeze=False)
    drew_any = False
    for i, rv in enumerate(rows):
        for j, cv in enumerate(cols):
            ax = axes[i][j]
            sub = df[(df[row_field] == rv) & (df[col_field] == cv)]
            got = painter(ax, sub)
            if got is None:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor("#fafafa")
            else:
                drew_any = True
            if i == 0:
                ax.set_title(col_label(cv), fontsize=8.5)
            if j == 0:
                ax.set_ylabel(row_label(rv), fontsize=8.5)
    if not drew_any:
        plt.close(fig)
        return None
    return _finish(fig, suptitle, handles, path)


# ---------------------------------------------------------------------------
# Family A — judge_effect (the core Pass@K vs CoT-Pass@K matrices)
# ---------------------------------------------------------------------------


def render_judge_effect(df: pd.DataFrame, out: Path, ks: list[int], style: dict) -> list[Path]:
    written: list[Path] = []
    handles = _judge_handles(df, style)
    for state in STATE_ORDER:
        d = df[df["state"] == state]
        if d.empty:
            continue
        # Exclude intermediate checkpoints — they have no CoT data and would
        # appear as empty rows in the judge_effect grid.
        models = [m for m in _ordered_models(d) if not _STEP_MODEL_RE.match(m)]
        benches = sorted(d["benchmark"].dropna().unique(), key=_bench_rank)
        for spec in METRIC_SPECS:
            path = out / "judge_effect" / f"{spec['key']}__{state}.png"
            p = _grid(
                d, rows=models, cols=benches, row_field="model", col_field="benchmark",
                painter=lambda ax, sub, _spec=spec: _cell_judge_effect(ax, sub, _spec, style, ks),
                row_label=lambda m: labels.short_model(m),
                col_label=lambda b: f"{labels.language(b)}",
                suptitle=f"{spec['nj']}  vs  {spec['cot']}   ·   {labels.mode_label(state)}",
                handles=handles, path=path,
            )
            if p:
                written.append(p)
    return written


# ---------------------------------------------------------------------------
# Family B — bench_compare (language transfer; nojudge + cot variants)
# ---------------------------------------------------------------------------


def render_bench_compare(df: pd.DataFrame, out: Path, ks: list[int], style: dict) -> list[Path]:
    written: list[Path] = []
    lang_handles = _lang_handles()
    for state in STATE_ORDER:
        d = df[df["state"] == state]
        if d.empty:
            continue
        models = _ordered_models(d)
        # --- nojudge: a grid of models, each cell = 4 language curves (No-Judge) ---
        dn = d[~d["judged"]]
        for spec in METRIC_SPECS:
            ncol = min(4, len(models)) or 1
            nrow = int(np.ceil(len(models) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol + 0.6, 2.3 * nrow + 0.8), squeeze=False)
            drew = False
            for idx in range(nrow * ncol):
                ax = axes[idx // ncol][idx % ncol]
                if idx >= len(models):
                    ax.axis("off")
                    continue
                m = models[idx]
                got = _cell_langs(ax, dn[dn["model"] == m], spec, ks)
                if got is None:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_facecolor("#fafafa")
                else:
                    drew = True
                ax.set_title(labels.short_model(m), fontsize=8.5)
            path = out / "bench_compare" / f"{spec['key']}__{state}__nojudge.png"
            if drew:
                written.append(_finish(fig, f"{spec['nj']} · language transfer (No-Judge) · {labels.mode_label(state)}",
                                       lang_handles, path))
            else:
                plt.close(fig)
        # --- cot: rows=model, cols=[No-Judge | each judge], cell = 4 language curves ---
        series_cols = ["No-Judge"] + sorted(d.loc[d["judged"], "series"].dropna().unique())
        for spec in METRIC_SPECS:
            p = _grid(
                d, rows=models, cols=series_cols, row_field="model", col_field="series",
                painter=lambda ax, sub, _spec=spec: _cell_langs(ax, sub, _spec, ks),
                row_label=lambda m: labels.short_model(m),
                col_label=lambda s: s.replace("cot:", "cot·"),
                suptitle=f"{spec['nj']} → {spec['cot']} · language × judge · {labels.mode_label(state)}",
                handles=lang_handles,
                path=out / "bench_compare" / f"{spec['key']}__{state}__cot.png",
                cell_w=2.7, cell_h=2.2,
            )
            if p:
                written.append(p)
    return written


# ---------------------------------------------------------------------------
# Family C — size_compare (scaling; nojudge + per-judge cot overlay)
# ---------------------------------------------------------------------------


def render_size_compare(df: pd.DataFrame, out: Path, ks: list[int], style: dict) -> list[Path]:
    written: list[Path] = []
    for state in STATE_ORDER:
        d = df[df["state"] == state]
        if d.empty:
            continue
        families = sorted(d["model_family"].dropna().unique())
        benches = sorted(d["benchmark"].dropna().unique(), key=_bench_rank)
        for spec in METRIC_SPECS:
            # nojudge
            p = _grid(
                d, rows=families, cols=benches, row_field="model_family", col_field="benchmark",
                painter=lambda ax, sub, _spec=spec: _cell_sizes(ax, sub, _spec, ks, None),
                row_label=lambda f: f, col_label=lambda b: labels.language(b),
                suptitle=f"{spec['nj']} · size scaling (No-Judge) · {labels.mode_label(state)}",
                handles=None,
                path=out / "size_compare" / f"{spec['key']}__{state}__nojudge.png",
            )
            if p:
                written.append(p)
            # cot, one PNG per judge (size colour, No-Judge solid + that judge dashed)
            for jm in sorted(d.loc[d["judged"], "judge_model"].dropna().unique()):
                jtag = labels.short_judge(jm, _judge_state(df, jm))
                p = _grid(
                    d, rows=families, cols=benches, row_field="model_family", col_field="benchmark",
                    painter=lambda ax, sub, _spec=spec, _jm=jm: _cell_sizes(ax, sub, _spec, ks, _jm),
                    row_label=lambda f: f, col_label=lambda b: labels.language(b),
                    suptitle=f"{spec['nj']} → {spec['cot']} · size scaling · cot={jtag} · {labels.mode_label(state)}",
                    handles=None,
                    path=out / "size_compare" / f"{spec['key']}__{state}__cot__{labels.short_model(jm)}.png",
                )
                if p:
                    written.append(p)
    return written


# ---------------------------------------------------------------------------
# Family F — veto_curve (Δ(k) = No-Judge − cot)
# ---------------------------------------------------------------------------


def render_veto_curve(df: pd.DataFrame, out: Path, ks: list[int], style: dict) -> list[Path]:
    written: list[Path] = []
    handles = [
        Line2D([0], [0], color=c, lw=1.7, ls="--", marker="s",
               label=f"cot · {labels.short_judge(jm, _judge_state(df, jm))}")
        for jm, c in style.items()
    ]
    for state in STATE_ORDER:
        d = df[df["state"] == state]
        if d.empty:
            continue
        models = _ordered_models(d)
        benches = sorted(d["benchmark"].dropna().unique(), key=_bench_rank)
        for spec in METRIC_SPECS:
            p = _grid(
                d, rows=models, cols=benches, row_field="model", col_field="benchmark",
                painter=lambda ax, sub, _spec=spec: _cell_veto(ax, sub, _spec, style, ks),
                row_label=lambda m: labels.short_model(m),
                col_label=lambda b: labels.language(b),
                suptitle=f"Veto effect Δ(k) = {spec['nj']} − {spec['cot']} · {labels.mode_label(state)}",
                handles=handles,
                path=out / "veto_curve" / f"{spec['key']}__{state}.png",
            )
            if p:
                written.append(p)
    return written


# ---------------------------------------------------------------------------
# Family H — mode_compare (pretrained vs non-think vs think)
# ---------------------------------------------------------------------------


def render_mode_compare(df: pd.DataFrame, out: Path, ks: list[int], style: dict) -> list[Path]:
    written: list[Path] = []
    families = sorted(df["model_family"].dropna().unique())
    for fam in families:
        d = df[df["model_family"] == fam]
        sizes = sorted(d["model_size_b"].dropna().unique())
        benches = sorted(d["benchmark"].dropna().unique(), key=_bench_rank)
        if not sizes or not benches:
            continue
        # pick one representative judge for the cot overlay (most-covered) and
        # name it explicitly so the dashed lines aren't ambiguous about the judge.
        jcounts = d.loc[d["judged"], "judge_model"].value_counts()
        cot_judge = jcounts.index[0] if len(jcounts) else None
        jtag = labels.short_judge(cot_judge, _judge_state(df, cot_judge)) if cot_judge is not None else "cot"
        handles = _mode_handles() + [
            Line2D([0], [0], color="#555", lw=1.4, ls="--", marker="s",
                   label=f"cot · {jtag} (dashed)")
        ]
        for spec in METRIC_SPECS:
            def paint(ax, sub, _spec=spec, _j=cot_judge):
                return _cell_modes(ax, sub, _spec, ks, _j)

            # rows=size, cols=benchmark; sub keyed by (size, benchmark) within family
            rows = sizes
            cols = benches
            nr, nc = len(rows), len(cols)
            fig, axes = plt.subplots(nr, nc, figsize=(2.9 * nc + 0.6, 2.3 * nr + 0.8), squeeze=False)
            drew = False
            for i, sz in enumerate(rows):
                for j, b in enumerate(cols):
                    ax = axes[i][j]
                    sub = d[(d["model_size_b"] == sz) & (d["benchmark"] == b)]
                    got = paint(ax, sub)
                    if got is None:
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_facecolor("#fafafa")
                    else:
                        drew = True
                    if i == 0:
                        ax.set_title(labels.language(b), fontsize=8.5)
                    if j == 0:
                        ax.set_ylabel(f"{sz:g}B", fontsize=8.5)
            path = out / "mode_compare" / f"{spec['key']}__{fam}.png"
            if drew:
                written.append(_finish(
                    fig, f"{spec['nj']} · pretrained vs instruct modes · {fam}  ·  cot judge: {jtag}",
                    handles, path))
            else:
                plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Family G — per_model fingerprint (rows=metric × cols=benchmark)
# ---------------------------------------------------------------------------


def render_per_model(df: pd.DataFrame, out: Path, ks: list[int], style: dict) -> list[Path]:
    written: list[Path] = []
    handles = _judge_handles(df, style)
    for model, state in _ordered_model_modes(df):
        d = df[(df["model"] == model) & (df["state"] == state)]
        if d.empty:
            continue
        benches = sorted(d["benchmark"].dropna().unique(), key=_bench_rank)
        if not benches:
            continue
        nr, nc = len(METRIC_SPECS), len(benches)
        fig, axes = plt.subplots(nr, nc, figsize=(2.7 * nc + 0.6, 2.1 * nr + 0.9), squeeze=False)
        drew = False
        for i, spec in enumerate(METRIC_SPECS):
            for j, b in enumerate(benches):
                ax = axes[i][j]
                got = _cell_judge_effect(ax, d[d["benchmark"] == b], spec, style, ks)
                if got is None:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_facecolor("#fafafa")
                else:
                    drew = True
                if i == 0:
                    ax.set_title(labels.language(b), fontsize=8.5)
                if j == 0:
                    ax.set_ylabel(spec["nj"], fontsize=8)
        path = out / "per_model" / f"{labels.short_model(model)}__{state}.png".replace("·", "_")
        if drew:
            written.append(_finish(fig, f"Fingerprint · {labels.short_model(model)} · {labels.mode_label(state)}",
                                   handles, path))
        else:
            plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Tables (matplotlib) — families D & I
# ---------------------------------------------------------------------------


def _render_table(row_labels, col_labels, text, colors, title, path, footnote=None) -> Path:
    nr, nc = len(row_labels), len(col_labels)
    fig, ax = plt.subplots(figsize=(1.15 * nc + 2.4, 0.34 * nr + 1.3))
    ax.axis("off")
    tbl = ax.table(cellText=text, rowLabels=row_labels, colLabels=col_labels,
                   cellColours=colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.25)
    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#dddddd")
    ax.set_title(title, fontweight="bold", fontsize=11, pad=14)
    if footnote:
        fig.text(0.5, 0.005, footnote, ha="center", fontsize=7, color="#666")
    return _save(fig, path)


def _seq_color(v: float, lo: float, hi: float):
    if hi <= lo or not np.isfinite(v):
        return (1, 1, 1, 1)
    t = (v - lo) / (hi - lo)
    return plt.cm.Blues(0.12 + 0.6 * t)


def _div_color(v: float, vmax: float):
    if vmax <= 0 or not np.isfinite(v):
        return (1, 1, 1, 1)
    t = 0.5 + 0.5 * max(-1.0, min(1.0, v / vmax))
    return plt.cm.RdBu_r(t)


def render_tables(df: pd.DataFrame, out: Path, ks: list[int], style: dict) -> list[Path]:
    written: list[Path] = []
    table_ks = [k for k in (1, 64) if k in ks] or [ks[-1]]
    benches = sorted(df["benchmark"].dropna().unique(), key=_bench_rank)
    metric_heads = [s["nj"].replace("@K", "").replace("G-Pass ", "G ").replace("mG-Pass", "mG").strip()
                    for s in METRIC_SPECS]
    judges = sorted(df.loc[df["judged"], "judge_model"].dropna().unique())

    for b in benches:
        db = df[df["benchmark"] == b]
        mm = [(m, s) for (m, s) in _ordered_model_modes(db)]
        for k in table_ks:
            # --- nojudge absolute (×100) ---
            text, colors, rlabels = [], [], []
            for m, st in mm:
                rn = db[(db["model"] == m) & (db["state"] == st) & (~db["judged"])]
                if rn.empty:
                    continue
                r = rn.iloc[0]
                vals = [(_value(r, s["base"], k, s["tau"])) for s in METRIC_SPECS]
                row_t, row_c = [], []
                for v in vals:
                    row_t.append("" if v is None else f"{v * 100:.1f}")
                    row_c.append(_seq_color(v if v is not None else np.nan, 0.0, 1.0))
                text.append(row_t)
                colors.append(row_c)
                rlabels.append(mm_label(m, st))
            if text:
                written.append(_render_table(
                    rlabels, metric_heads, text, colors,
                    f"{labels.language(b)} · No-Judge · k={k} (×100)",
                    out / "tables" / f"{b}__k{k}__nojudge.png",
                    footnote="value = metric ×100; darker = higher",
                ))
            # --- cot Δ (No-Judge − cot) per judge ---
            for jm in judges:
                text, colors, rlabels = [], [], []
                for m, st in mm:
                    rn = db[(db["model"] == m) & (db["state"] == st) & (~db["judged"])]
                    rc = db[(db["model"] == m) & (db["state"] == st) & (db["judge_model"] == jm)]
                    if rn.empty or rc.empty:
                        continue
                    r0, r1 = rn.iloc[0], rc.iloc[0]
                    row_t, row_c = [], []
                    for s in METRIC_SPECS:
                        a = _value(r0, s["base"], k, s["tau"])
                        c = _value(r1, s["base"], k, s["tau"])
                        if a is None or c is None:
                            row_t.append("")
                            row_c.append((1, 1, 1, 1))
                        else:
                            d = (a - c) * 100
                            row_t.append(f"{d:.1f}")
                            row_c.append(_div_color(d, 30.0))
                    if any(t for t in row_t):
                        text.append(row_t)
                        colors.append(row_c)
                        rlabels.append(mm_label(m, st))
                if text:
                    jtag = labels.short_judge(jm, _judge_state(df, jm))
                    written.append(_render_table(
                        rlabels, metric_heads, text, colors,
                        f"{labels.language(b)} · veto Δ (No-Judge − cot) · {jtag} · k={k}",
                        out / "tables" / f"{b}__k{k}__cotdelta__{labels.short_model(jm)}.png",
                        footnote="Δ = (No-Judge − cot) ×100; red = bigger veto",
                    ))
    return written


def render_headline(df: pd.DataFrame, out: Path, ks: list[int], style: dict) -> list[Path]:
    """Per-judge headline: cell = 'pass@k / cot@k', coloured by the veto Δ."""
    written: list[Path] = []
    table_ks = [k for k in (1, 64) if k in ks] or [ks[-1]]
    benches = sorted(df["benchmark"].dropna().unique(), key=_bench_rank)
    judges = sorted(df.loc[df["judged"], "judge_model"].dropna().unique())
    mm = _ordered_model_modes(df)
    for jm in judges:
        jtag = labels.short_judge(jm, _judge_state(df, jm))
        for k in table_ks:
            text, colors, rlabels = [], [], []
            for m, st in mm:
                rn = df[(df["model"] == m) & (df["state"] == st) & (~df["judged"])]
                row_t, row_c, has = [], [], False
                for b in benches:
                    rnb = rn[rn["benchmark"] == b]
                    rcb = df[(df["model"] == m) & (df["state"] == st)
                             & (df["benchmark"] == b) & (df["judge_model"] == jm)]
                    a = _value(rnb.iloc[0], "pass", k, None) if not rnb.empty else None
                    c = _value(rcb.iloc[0], "pass", k, None) if not rcb.empty else None
                    if a is None:
                        row_t.append("")
                        row_c.append((1, 1, 1, 1))
                    elif c is None:
                        row_t.append(f"{a * 100:.0f}/–")
                        row_c.append((1, 1, 1, 1))
                        has = True
                    else:
                        row_t.append(f"{a * 100:.0f}/{c * 100:.0f}")
                        row_c.append(_div_color((a - c) * 100, 30.0))
                        has = True
                if has:
                    text.append(row_t)
                    colors.append(row_c)
                    rlabels.append(mm_label(m, st))
            if text:
                written.append(_render_table(
                    rlabels, [labels.language(b) for b in benches], text, colors,
                    f"Headline pass@{k} / cot-pass@{k} · cot={jtag}",
                    out / "tables" / f"headline__{labels.short_model(jm)}__k{k}.png",
                    footnote="cell = pass / cot-pass (×100); red = larger veto",
                ))
    return written


# ---------------------------------------------------------------------------
# Family E — multilingual veto Δ heatmaps + companion CSV
# ---------------------------------------------------------------------------


def render_comparisons(df: pd.DataFrame, out: Path, ks: list[int], style: dict) -> list[Path]:
    written: list[Path] = []
    table_ks = [k for k in (1, 64) if k in ks] or [ks[-1]]
    benches = sorted(df["benchmark"].dropna().unique(), key=_bench_rank)
    langs = [labels.language(b) for b in benches]
    judges = sorted(df.loc[df["judged"], "judge_model"].dropna().unique())
    mm = _ordered_model_modes(df)

    # ---- heatmaps: rows=(model·mode), cols=language, value = No-Judge − cot ----
    for spec in METRIC_SPECS:
        for k in table_ks:
            for jm in judges:
                mat, rlabels = [], []
                for m, st in mm:
                    rn = df[(df["model"] == m) & (df["state"] == st) & (~df["judged"])]
                    row = []
                    any_v = False
                    for b in benches:
                        rnb = rn[rn["benchmark"] == b]
                        rcb = df[(df["model"] == m) & (df["state"] == st)
                                 & (df["benchmark"] == b) & (df["judge_model"] == jm)]
                        a = _value(rnb.iloc[0], spec["base"], k, spec["tau"]) if not rnb.empty else None
                        c = _value(rcb.iloc[0], spec["base"], k, spec["tau"]) if not rcb.empty else None
                        if a is None or c is None:
                            row.append(np.nan)
                        else:
                            row.append((a - c) * 100)
                            any_v = True
                    if any_v:
                        mat.append(row)
                        rlabels.append(mm_label(m, st))
                if not mat:
                    continue
                arr = np.array(mat, dtype=float)
                jtag = labels.short_judge(jm, _judge_state(df, jm))
                fig, ax = plt.subplots(figsize=(1.0 * len(langs) + 2.0, 0.32 * len(rlabels) + 1.4))
                vmax = np.nanmax(np.abs(arr)) or 1.0
                im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
                ax.set_xticks(range(len(langs)))
                ax.set_xticklabels(langs, fontsize=8)
                ax.set_yticks(range(len(rlabels)))
                ax.set_yticklabels(rlabels, fontsize=6.5)
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        if np.isfinite(arr[i, j]):
                            ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", fontsize=6,
                                    color="white" if abs(arr[i, j]) > 0.6 * vmax else "black")
                ax.set_title(f"Veto Δ · {spec['nj']} · k={k} · cot={jtag}", fontsize=10, fontweight="bold")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="No-Judge − cot (×100)")
                written.append(
                    _save(fig, out / "comparisons" / f"veto__{spec['key']}__k{k}__{labels.short_model(jm)}.png")
                )

    # ---- companion long CSV: every (model, mode, lang, judge, metric, k) base/cot/Δ ----
    for k in table_ks:
        rows = []
        for m, st in mm:
            for b in benches:
                rn = df[(df["model"] == m) & (df["state"] == st)
                        & (df["benchmark"] == b) & (~df["judged"])]
                if rn.empty:
                    continue
                r0 = rn.iloc[0]
                for jm in judges:
                    rc = df[(df["model"] == m) & (df["state"] == st)
                            & (df["benchmark"] == b) & (df["judge_model"] == jm)]
                    if rc.empty:
                        continue
                    r1 = rc.iloc[0]
                    for spec in METRIC_SPECS:
                        a = _value(r0, spec["base"], k, spec["tau"])
                        c = _value(r1, spec["base"], k, spec["tau"])
                        if a is None or c is None:
                            continue
                        rows.append({
                            "model": m, "model_short": labels.short_model(m),
                            "state": st, "mode": labels.mode_label(st),
                            "benchmark": b, "language": labels.language(b),
                            "judge_model": jm, "judge_short": labels.short_judge(jm, _judge_state(df, jm)),
                            "metric": spec["key"], "k": k,
                            "nojudge": a, "cot": c, "delta": a - c,
                        })
        if rows:
            csv_path = out / "comparisons" / f"pass_vs_cot_k{k}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            written.append(csv_path)
    return written


# ---------------------------------------------------------------------------
# Family RL — rl_progress (pass@K across RL training steps)
# ---------------------------------------------------------------------------


def render_rl_progress(df: pd.DataFrame, out: Path, ks: list[int], style: dict) -> list[Path]:
    """RL training progression: pass@K vs discrete training steps.

    Generates one PNG per RL model series detected in the data.  A series is
    identified when one or more ``@stepN`` checkpoint variants (produced by
    scan.py's step_NNN recognition) appear alongside the final RL model and,
    optionally, a pretrained baseline of the same family and size.

    Layout: rows = selected K values (k=1 and k=64); cols = benchmarks.
    Each cell: x-axis = categorical step positions ("Base", "s120", …, "Final");
    solid line = No-Judge, dashed lines = each CoT judge.
    """
    # Detect all @stepN intermediate checkpoint models.
    all_models = df["model"].dropna().unique()
    step_variants = [m for m in all_models if _STEP_MODEL_RE.match(m)]
    if not step_variants:
        return []

    written: list[Path] = []

    # Group @stepN variants by their base RL model (the name without @stepN).
    series_map: dict[str, list[str]] = {}
    for m in step_variants:
        base_rl = _STEP_MODEL_RE.match(m).group(1)
        series_map.setdefault(base_rl, []).append(m)

    for base_rl_model, ckpt_models in series_map.items():
        ckpt_sorted = sorted(ckpt_models, key=lambda m: int(_STEP_MODEL_RE.match(m).group(2)))

        # Identify pretrained baseline (step 0): same family+size, is_base=True,
        # no @step suffix, different name from the RL model.
        rl_rows = df[(df["model"] == base_rl_model) & (~df["judged"])].head(1)
        if rl_rows.empty:
            rl_rows = df[df["model"] == ckpt_sorted[0]].head(1)
        if rl_rows.empty:
            continue
        rl_family = rl_rows["model_family"].iloc[0]
        rl_size = rl_rows["model_size_b"].iloc[0]

        pretrained: str | None = None
        for m in all_models:
            if m == base_rl_model or _STEP_MODEL_RE.match(m):
                continue
            r = df[(df["model"] == m) & (~df["judged"])].head(1)
            if r.empty:
                continue
            if r["model_family"].iloc[0] == rl_family and r["model_size_b"].iloc[0] == rl_size:
                pretrained = m
                break

        # Build ordered progression: [(x_position, x_label, model_name)]
        progression: list[tuple[int, str, str]] = []
        x = 0
        if pretrained:
            progression.append((x, "Base", pretrained))
            x += 1
        for m in ckpt_sorted:
            step = int(_STEP_MODEL_RE.match(m).group(2))
            progression.append((x, f"s{step}", m))
            x += 1
        if not df[df["model"] == base_rl_model].empty:
            progression.append((x, "Final", base_rl_model))

        if len(progression) < 2:
            continue

        xs = [p[0] for p in progression]
        xlabels = [p[1] for p in progression]
        all_series_models = {m for _, _, m in progression}

        # Filter to base state only (we're tracking pretrained checkpoint evaluations).
        sub = df[df["model"].isin(all_series_models) & (df["state"] == "base")]

        benches = sorted(sub["benchmark"].dropna().unique(), key=_bench_rank)
        plot_ks = [k for k in (1, 64) if k in ks] or ks[:2]
        judges = sorted(sub.loc[sub["judged"], "judge_model"].dropna().unique())

        if not benches or not plot_ks:
            continue

        nr, nc = len(plot_ks), len(benches)
        fig, axes = plt.subplots(nr, nc, figsize=(2.8 * nc + 0.5, 2.3 * nr + 0.9), squeeze=False)
        drew = False

        # Collect global ymax for a consistent y-axis across all cells.
        all_vals: list[float] = []
        for k in plot_ks:
            for b in benches:
                for _, _, model in progression:
                    row = sub[(sub["model"] == model) & (sub["benchmark"] == b) & (~sub["judged"])].head(1)
                    if not row.empty:
                        v = _value(row.iloc[0], "pass", k, None)
                        if v is not None:
                            all_vals.append(v * 100)
        global_ymax = max(all_vals) if all_vals else 10.0

        for ki, k in enumerate(plot_ks):
            for bi, b in enumerate(benches):
                ax = axes[ki][bi]

                # No-Judge line
                nj_pts: list[tuple[int, float]] = []
                for xp, _xl, model in progression:
                    row = sub[(sub["model"] == model) & (sub["benchmark"] == b) & (~sub["judged"])].head(1)
                    if not row.empty:
                        v = _value(row.iloc[0], "pass", k, None)
                        if v is not None:
                            nj_pts.append((xp, v * 100))
                if nj_pts:
                    pxs, pys = zip(*nj_pts, strict=False)
                    ax.plot(pxs, pys, color=NJ_COLOR, lw=2.2, marker="o", ms=5, zorder=6)
                    drew = True

                # CoT judge lines
                for jm in judges:
                    jpts: list[tuple[int, float]] = []
                    for xp, _xl, model in progression:
                        row = sub[(sub["model"] == model) & (sub["benchmark"] == b)
                                  & (sub["judge_model"] == jm)].head(1)
                        if not row.empty:
                            v = _value(row.iloc[0], "pass", k, None)
                            if v is not None:
                                jpts.append((xp, v * 100))
                    if jpts:
                        pxs, pys = zip(*jpts, strict=False)
                        ax.plot(pxs, pys, color=style.get(jm, "gray"), lw=1.7, ls="--",
                                marker="s", ms=4, alpha=0.9, zorder=4)
                        drew = True

                ax.set_xticks(xs)
                ax.set_xticklabels(xlabels, fontsize=7)
                ax.set_xlim(-0.4, max(xs) + 0.4)
                ax.set_ylim(0, global_ymax * 1.12)
                ax.grid(True, axis="y", alpha=0.25, lw=0.5)
                ax.tick_params(axis="y", labelsize=6.5)
                if ki == 0:
                    ax.set_title(labels.language(b), fontsize=8.5)
                if bi == 0:
                    ax.set_ylabel(f"pass@{k} ×100", fontsize=8)

        if not drew:
            plt.close(fig)
            continue

        handles = _judge_handles(sub, style)
        pre_short = labels.short_model(pretrained) if pretrained else "Pretrained"
        rl_short = labels.short_model(base_rl_model)
        suptitle = f"RL Training Progress · {pre_short} → {rl_short}"
        safe = re.sub(r"[·\s/]+", "_", rl_short)
        written.append(_finish(fig, suptitle, handles, out / "rl_progress" / f"{safe}.png"))

    return written


# ---------------------------------------------------------------------------
# Family I — rl_table (checkpoints as columns)
# ---------------------------------------------------------------------------


def render_rl_table(df: pd.DataFrame, out: Path, ks: list[int], style: dict) -> list[Path]:
    """RL progression table: columns = [Base, s120, s240, Final], rows = metric × series.

    One table per (benchmark, k).  Cells show value × 100; '–' where data is
    absent (intermediate checkpoints have no CoT data).
    """
    all_models = df["model"].dropna().unique()
    step_variants = [m for m in all_models if _STEP_MODEL_RE.match(m)]
    if not step_variants:
        return []

    written: list[Path] = []

    series_map: dict[str, list[str]] = {}
    for m in step_variants:
        base_rl = _STEP_MODEL_RE.match(m).group(1)
        series_map.setdefault(base_rl, []).append(m)

    table_ks = [k for k in (1, 64) if k in ks] or [ks[-1]]

    for base_rl_model, ckpt_models in series_map.items():
        ckpt_sorted = sorted(ckpt_models, key=lambda m: int(_STEP_MODEL_RE.match(m).group(2)))

        rl_row = df[(df["model"] == base_rl_model) & (~df["judged"])].head(1)
        if rl_row.empty:
            rl_row = df[df["model"] == ckpt_sorted[0]].head(1)
        if rl_row.empty:
            continue
        rl_family = rl_row["model_family"].iloc[0]
        rl_size = rl_row["model_size_b"].iloc[0]

        pretrained: str | None = None
        for m in all_models:
            if m == base_rl_model or _STEP_MODEL_RE.match(m):
                continue
            r = df[(df["model"] == m) & (~df["judged"])].head(1)
            if r.empty:
                continue
            if r["model_family"].iloc[0] == rl_family and r["model_size_b"].iloc[0] == rl_size:
                pretrained = m
                break

        # Ordered checkpoints: Base → s120 → s240 → Final
        progression: list[tuple[str, str]] = []  # (model_name, col_label)
        if pretrained:
            progression.append((pretrained, labels.short_model(pretrained)))
        for m in ckpt_sorted:
            step = int(_STEP_MODEL_RE.match(m).group(2))
            progression.append((m, f"s{step}"))
        if not df[df["model"] == base_rl_model].empty:
            progression.append((base_rl_model, labels.short_model(base_rl_model)))

        if len(progression) < 2:
            continue

        col_labels = [label for _, label in progression]
        judges = sorted(df.loc[df["judged"], "judge_model"].dropna().unique())
        benches = sorted(df["benchmark"].dropna().unique(), key=_bench_rank)
        state = "base"

        for k in table_ks:
            for b in benches:
                row_labels: list[str] = []
                text: list[list[str]] = []
                tcolors: list[list] = []

                def _fmt(v):
                    return f"{v * 100:.1f}" if v is not None else "–"

                def _col(v, hi=1.0):
                    if v is None:
                        return (0.97, 0.97, 0.97, 1.0)
                    return _seq_color(v, 0.0, hi)

                for spec in METRIC_SPECS:
                    metric_key = (spec["nj"].replace("@K", "")
                                  .replace("G-Pass ", "G").replace("mG-Pass", "mG").strip())

                    # No-Judge row
                    nj_vals = []
                    for m, _ in progression:
                        r = df[(df["model"] == m) & (df["state"] == state)
                               & (df["benchmark"] == b) & (~df["judged"])].head(1)
                        v = _value(r.iloc[0], spec["base"], k, spec["tau"]) if not r.empty else None
                        nj_vals.append(v)
                    row_labels.append(f"{metric_key} NJ")
                    text.append([_fmt(v) for v in nj_vals])
                    hi = max((v for v in nj_vals if v is not None), default=1.0) or 1.0
                    tcolors.append([_col(v, hi) for v in nj_vals])

                    # CoT rows (one per judge)
                    for jm in judges:
                        jtag = labels.short_judge(jm, _judge_state(df, jm))
                        cot_vals = []
                        for m, _ in progression:
                            r = df[(df["model"] == m) & (df["state"] == state)
                                   & (df["benchmark"] == b) & (df["judge_model"] == jm)].head(1)
                            v = _value(r.iloc[0], spec["base"], k, spec["tau"]) if not r.empty else None
                            cot_vals.append(v)
                        if any(v is not None for v in cot_vals):
                            row_labels.append(f"{metric_key} {jtag}")
                            text.append([_fmt(v) for v in cot_vals])
                            hi_c = max((v for v in cot_vals if v is not None), default=1.0) or 1.0
                            tcolors.append([_col(v, hi_c) for v in cot_vals])

                if not text:
                    continue

                title = f"{labels.language(b)} · RL progression · k={k} (×100)"
                safe_rl = re.sub(r"[·\s/]+", "_", labels.short_model(base_rl_model))
                path = out / "rl_table" / f"{b}__k{k}__{safe_rl}.png"
                written.append(_render_table(row_labels, col_labels, text, tcolors, title, path,
                                             footnote="NJ = No-Judge; darker = higher; – = no data"))

    return written


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_FAMILIES = {
    "judge_effect": render_judge_effect,
    "bench_compare": render_bench_compare,
    "size_compare": render_size_compare,
    "veto_curve": render_veto_curve,
    "mode_compare": render_mode_compare,
    "per_model": render_per_model,
    "tables": render_tables,
    "headline": render_headline,
    "comparisons": render_comparisons,
    "rl_progress": render_rl_progress,
    "rl_table": render_rl_table,
}


def render_all(df: pd.DataFrame, output_dir: Path | str) -> dict[str, list[Path]]:
    """Render the whole suite, one family at a time, into ``output_dir``.

    Each family is isolated: a failure in one logs and is skipped rather than
    aborting the rest. Returns ``{family: [written paths]}``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    # Normalise the discriminator to real booleans regardless of CSV dtype.
    if df["judged"].dtype == object:
        df["judged"] = df["judged"].map(
            {"True": True, "False": False, True: True, False: False}
        ).fillna(False)
    df["judged"] = df["judged"].astype(bool)

    ks = _k_axis(df)
    style = _judge_style(df)

    written: dict[str, list[Path]] = {}
    for name, fn in _FAMILIES.items():
        try:
            paths = fn(df, out, ks, style)
            written[name] = paths
            logger.info(f"[plots] {name}: {len(paths)} file(s)")
        except Exception as e:  # pragma: no cover - defensive, keep the suite alive
            logger.error(f"[plots] {name} failed: {e}")
            written[name] = []
    total = sum(len(v) for v in written.values())
    logger.info(f"[plots] suite complete: {total} file(s) under {out}")
    return written
