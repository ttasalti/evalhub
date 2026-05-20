"""Publication-ready static plots rendered from the long-form CSV.

Each public function takes the DataFrame produced by
:func:`evalhub.report.aggregate.build_dataframe` and writes one or more image
files into ``output_dir``. Filenames are sanitised so the output directory is
safe to commit or upload to a paper repository as-is.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from evalhub.utils.logger import logger

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


_SUPPORTED_FORMATS: tuple[str, ...] = ("png", "pdf")


def _require_plotting():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "evalhub.report.plots requires matplotlib + seaborn. Install with "
            "`pip install evalhub[report]`."
        ) from e
    sns.set_theme(context="paper", style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
        }
    )
    return plt, sns


def _normalise_formats(formats: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for fmt in formats:
        fmt = fmt.lower().strip()
        if fmt == "both":
            out.extend(_SUPPORTED_FORMATS)
            continue
        if fmt not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format {fmt!r}. Allowed: {_SUPPORTED_FORMATS} or 'both'.")
        out.append(fmt)
    # de-duplicate while preserving order
    seen: set[str] = set()
    return tuple(f for f in out if not (f in seen or seen.add(f)))


_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(value: str) -> str:
    return _SAFE_RE.sub("_", value).strip("_") or "unnamed"


def _save(fig, output_dir: Path, stem: str, formats: tuple[str, ...]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt)
        written.append(path)
    import matplotlib.pyplot as plt  # local re-import to close the figure

    plt.close(fig)
    return written


def plot_pass_at_k_curves(
    df: pd.DataFrame, output_dir: Path | str, formats: Iterable[str] = ("png",)
) -> list[Path]:
    """One figure per (model, benchmark); X=K (log), Y=Pass@K, hue=eval_type."""
    plt, sns = _require_plotting()
    formats_t = _normalise_formats(formats)
    output_dir = Path(output_dir)
    df = df.dropna(subset=["k", "pass_at_k"])
    written: list[Path] = []
    if df.empty:
        logger.warning("plot_pass_at_k_curves: no rows with k/pass_at_k; skipping.")
        return written
    grouped = df.groupby(["model", "benchmark"], sort=True)
    for (model, benchmark), sub in grouped:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        sns.lineplot(
            data=sub.sort_values("k"),
            x="k",
            y="pass_at_k",
            hue="eval_type",
            marker="o",
            ax=ax,
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K (log scale)")
        ax.set_ylabel("Pass@K")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Pass@K — {model} on {benchmark}")
        ax.legend(title="Eval type", loc="best")
        stem = _safe_name(f"pass_at_k__{model}__{benchmark}")
        written.extend(_save(fig, output_dir, stem, formats_t))
    logger.info(f"plot_pass_at_k_curves: wrote {len(written)} file(s)")
    return written


def plot_base_vs_cot_bars(
    df: pd.DataFrame, output_dir: Path | str, formats: Iterable[str] = ("png",)
) -> list[Path]:
    """Grouped bar chart per benchmark: base Pass@1 vs CoT-vetted Pass@1; hue=model."""
    plt, sns = _require_plotting()
    formats_t = _normalise_formats(formats)
    output_dir = Path(output_dir)
    sub = df[(df["k"] == 1) & df["pass_at_k"].notna()].copy()
    if sub.empty:
        logger.warning("plot_base_vs_cot_bars: no Pass@1 rows; skipping.")
        return []
    fig_height = max(4.0, 0.6 * sub["benchmark"].nunique() + 2.0)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    sns.barplot(
        data=sub,
        y="benchmark",
        x="pass_at_k",
        hue="model",
        ax=ax,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Pass@1")
    ax.set_ylabel("Benchmark")
    ax.set_title("Base vs CoT-vetted Pass@1")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    return _save(fig, output_dir, "base_vs_cot_pass_at_1", formats_t)


def plot_pass1_heatmap(
    df: pd.DataFrame, output_dir: Path | str, formats: Iterable[str] = ("png",)
) -> list[Path]:
    """Heatmap rows=model, cols=benchmark, value=Pass@1 (base eval only)."""
    plt, _sns = _require_plotting()
    formats_t = _normalise_formats(formats)
    output_dir = Path(output_dir)
    sub = df[(df["k"] == 1) & (df["eval_type"] == "base_eval") & df["pass_at_k"].notna()]
    if sub.empty:
        logger.warning("plot_pass1_heatmap: no base_eval Pass@1 rows; skipping.")
        return []
    pivot = sub.pivot_table(
        index="model", columns="benchmark", values="pass_at_k", aggfunc="mean"
    )
    import seaborn as sns

    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.6 * pivot.shape[1] + 2.0), max(4.0, 0.5 * pivot.shape[0] + 2.0))
    )
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title("Pass@1 heatmap (base eval)")
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Model")
    return _save(fig, output_dir, "pass_at_1_heatmap", formats_t)


def plot_cot_veto_rate(
    df: pd.DataFrame, output_dir: Path | str, formats: Iterable[str] = ("png",)
) -> list[Path]:
    """Bar chart of CoT veto rate (cot_false_count / total_generations) per run."""
    plt, sns = _require_plotting()
    formats_t = _normalise_formats(formats)
    output_dir = Path(output_dir)
    sub = df[(df["eval_type"] == "cot_eval") & df["total_generations"].notna()].copy()
    if sub.empty:
        logger.warning("plot_cot_veto_rate: no cot_eval rows; skipping.")
        return []
    # one record per run: keep a single row per (model, benchmark)
    sub = sub.drop_duplicates(subset=["model", "benchmark", "run_dir"])
    sub["veto_rate"] = sub["cot_false_count"].fillna(0) / sub["total_generations"].replace(
        {0: float("nan")}
    )
    sub = sub.dropna(subset=["veto_rate"])
    if sub.empty:
        logger.warning("plot_cot_veto_rate: all rows had zero generations; skipping.")
        return []
    fig_height = max(4.0, 0.6 * sub["benchmark"].nunique() + 2.0)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    sns.barplot(data=sub, y="benchmark", x="veto_rate", hue="model", ax=ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("CoT veto rate (cot_false / generations)")
    ax.set_ylabel("Benchmark")
    ax.set_title("CoT veto rate by model & benchmark")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    return _save(fig, output_dir, "cot_veto_rate", formats_t)


def render_all(
    df: pd.DataFrame, output_dir: Path | str, formats: Iterable[str] = ("png",)
) -> dict[str, list[Path]]:
    """Render the full publication set. Returns ``{plot_name: [paths,...]}``."""
    output_dir = Path(output_dir)
    return {
        "pass_at_k_curves": plot_pass_at_k_curves(df, output_dir, formats),
        "base_vs_cot_bars": plot_base_vs_cot_bars(df, output_dir, formats),
        "pass1_heatmap": plot_pass1_heatmap(df, output_dir, formats),
        "cot_veto_rate": plot_cot_veto_rate(df, output_dir, formats),
    }
