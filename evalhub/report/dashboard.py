"""Streamlit + Plotly dashboard for exploring an aggregated eval CSV.

The dashboard is launched indirectly via ``evalhub report dashboard``, which
invokes ``streamlit run -m evalhub.report.dashboard`` and exports two env
vars so this module can locate its inputs:

* ``EVALHUB_REPORT_CSV``  — required, absolute path to the long-form CSV.
* ``EVALHUB_REPORT_ROOT`` — optional, absolute path to the ``OUTPUT_ROOT``
  used for the "drill-down" tab. When unset, drill-down is disabled.

The module exposes :func:`main` so callers other than the Typer wrapper
(e.g. ``streamlit run <path>``) can also drive it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def _require_streamlit():
    try:
        import plotly.express as px
        import streamlit as st
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "evalhub.report.dashboard requires streamlit + plotly. Install with "
            "`pip install evalhub[report]`."
        ) from e
    return st, px


def _load_csv(csv_path: Path) -> pd.DataFrame:
    import pandas as pd

    df = pd.read_csv(csv_path)
    # k can legitimately be NaN for empty-pass_at_k stubs; cast non-null values to int.
    if "k" in df.columns:
        df["k"] = df["k"].astype("Int64")
    return df


def _filter_panel(st, df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    eval_types = st.sidebar.multiselect(
        "Eval type", sorted(df["eval_type"].dropna().unique()),
        default=sorted(df["eval_type"].dropna().unique()),
    )
    models = st.sidebar.multiselect(
        "Model", sorted(df["model"].dropna().unique()),
        default=sorted(df["model"].dropna().unique()),
    )
    states = st.sidebar.multiselect(
        "State", sorted(df["state"].dropna().unique()),
        default=sorted(df["state"].dropna().unique()),
    )
    benchmarks = st.sidebar.multiselect(
        "Benchmark", sorted(df["benchmark"].dropna().unique()),
        default=sorted(df["benchmark"].dropna().unique()),
    )
    out = df[
        df["eval_type"].isin(eval_types)
        & df["model"].isin(models)
        & df["state"].isin(states)
        & df["benchmark"].isin(benchmarks)
    ]
    return out


def _tab_overview(st, df: pd.DataFrame) -> None:
    st.subheader("Run summary")
    cols = st.columns(4)
    cols[0].metric("Runs", df["run_dir"].nunique())
    cols[1].metric("Models", df["model"].nunique())
    cols[2].metric("Benchmarks", df["benchmark"].nunique())
    cols[3].metric("CoT runs", df[df["eval_type"] == "cot_eval"]["run_dir"].nunique())
    st.dataframe(df, use_container_width=True, hide_index=True)


def _tab_pass_at_k(st, px, df: pd.DataFrame) -> None:
    st.subheader("Pass@K curves")
    sub = df.dropna(subset=["k", "pass_at_k"])
    if sub.empty:
        st.info("No Pass@K rows match the current filters.")
        return
    fig = px.line(
        sub.sort_values("k"),
        x="k",
        y="pass_at_k",
        color="model",
        line_dash="eval_type",
        symbol="state",
        facet_col="benchmark",
        facet_col_wrap=2,
        markers=True,
        log_x=True,
        range_y=[0.0, 1.0],
    )
    fig.update_layout(legend_title_text="Model")
    st.plotly_chart(fig, use_container_width=True)


def _tab_heatmap(st, px, df: pd.DataFrame) -> None:
    st.subheader("Pass@K heatmap")
    k_options = sorted({int(k) for k in df["k"].dropna().unique()})
    if not k_options:
        st.info("No K values to plot.")
        return
    k_pick = st.selectbox("K", k_options, index=0)
    sub = df[(df["k"] == k_pick) & df["pass_at_k"].notna()]
    if sub.empty:
        st.info(f"No rows with K={k_pick}.")
        return
    pivot = sub.pivot_table(index="model", columns="benchmark", values="pass_at_k", aggfunc="mean")
    fig = px.imshow(
        pivot, text_auto=".2f", color_continuous_scale="Viridis", zmin=0.0, zmax=1.0, aspect="auto"
    )
    fig.update_layout(title=f"Pass@{k_pick}")
    st.plotly_chart(fig, use_container_width=True)


def _tab_veto(st, px, df: pd.DataFrame) -> None:
    st.subheader("CoT veto rate")
    sub = df[(df["eval_type"] == "cot_eval") & df["total_generations"].notna()].drop_duplicates(
        subset=["model", "benchmark", "run_dir"]
    )
    if sub.empty:
        st.info("No CoT runs match the current filters.")
        return
    sub = sub.assign(
        veto_rate=sub["cot_false_count"].fillna(0)
        / sub["total_generations"].replace({0: float("nan")})
    ).dropna(subset=["veto_rate"])
    fig = px.bar(
        sub, x="veto_rate", y="benchmark", color="model", orientation="h", range_x=[0.0, 1.0]
    )
    fig.update_layout(xaxis_title="cot_false / generations", yaxis_title="Benchmark")
    st.plotly_chart(fig, use_container_width=True)


def _tab_drilldown(st, df: pd.DataFrame, results_root: Path | None) -> None:
    st.subheader("Drill-down")
    if results_root is None:
        st.info("Drill-down disabled (no --results-root passed). Re-launch with one.")
        return
    runs = df.drop_duplicates(subset=["run_dir"])[
        ["model", "state", "benchmark", "eval_type", "run_dir"]
    ]
    if runs.empty:
        st.info("No runs to drill into.")
        return
    selection = st.selectbox(
        "Pick a run",
        runs.itertuples(index=False),
        format_func=lambda r: f"[{r.eval_type}] {r.model}/{r.state} on {r.benchmark}",
    )
    run_dir = Path(selection.run_dir) / selection.benchmark
    if not run_dir.exists():
        st.warning(f"Run directory not found on disk: {run_dir}")
        return
    candidates = sorted(run_dir.glob("*.jsonl"))
    if not candidates:
        st.info("No JSONL files under this run.")
        return
    file_pick = st.selectbox("File", candidates, format_func=lambda p: p.name)
    max_rows = st.slider("Rows to load", min_value=10, max_value=2000, value=200, step=10)
    import pandas as pd

    records: list[dict] = []
    with file_pick.open() as f:
        import json

        for i, line in enumerate(f):
            if i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                st.warning(f"Skipping malformed line {i}: {exc}")
    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
    else:
        st.info("No records loaded.")


def main() -> None:
    st, px = _require_streamlit()
    csv_env = os.environ.get("EVALHUB_REPORT_CSV", "")
    if not csv_env:
        st.error(
            "EVALHUB_REPORT_CSV is not set. Launch via `evalhub report dashboard "
            "--csv <path>` or export the env var manually."
        )
        return
    csv_path = Path(csv_env)
    if not csv_path.exists():
        st.error(f"CSV not found: {csv_path}")
        return
    root_env = os.environ.get("EVALHUB_REPORT_ROOT", "")
    results_root = Path(root_env) if root_env else None

    st.set_page_config(page_title="EvalHub Report", layout="wide")
    st.title("EvalHub — Result Aggregation")
    st.caption(f"CSV: {csv_path}")

    df = _load_csv(csv_path)
    filtered = _filter_panel(st, df)
    tabs = st.tabs(["Overview", "Pass@K", "Heatmap", "CoT veto", "Drill-down"])
    with tabs[0]:
        _tab_overview(st, filtered)
    with tabs[1]:
        _tab_pass_at_k(st, px, filtered)
    with tabs[2]:
        _tab_heatmap(st, px, filtered)
    with tabs[3]:
        _tab_veto(st, px, filtered)
    with tabs[4]:
        _tab_drilldown(st, filtered, results_root)


if __name__ == "__main__":  # pragma: no cover - executed by `streamlit run`
    main()
