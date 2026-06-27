"""Smoke tests for the Pass@K vs CoT-Pass@K plot suite :mod:`evalhub.report.plots`.

Matplotlib/seaborn are optional, so the whole module is skipped when they (or
pandas) are missing. We build a tiny but *schema-faithful* wide DataFrame in
memory — two model families, base + instruct modes, all four benchmarks, every
judge ``think`` — and assert each family writes at least one non-empty file and
that the engine helpers behave.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

from evalhub.report import plots  # noqa: E402
from evalhub.report.aggregate import META_COLUMNS, metric_columns  # noqa: E402

KS = [1, 2, 4, 8, 16, 32, 64]
BENCHES = ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026"]
JUDGES = ["gemma-4-26B-A4B-it", "Qwen3.6-35B-A3B"]


def _metric_payload(base_pass: float, judged: bool) -> dict:
    """A monotone-in-K metric block; cot rows sit slightly below No-Judge."""
    cols: dict[str, float] = {}
    for k in KS:
        # rises with k, capped < 1, judged variant vetoed down a notch.
        v = min(0.95, base_pass + 0.04 * (k.bit_length() - 1))
        if judged:
            v = max(0.0, v - 0.08)
        cols[f"pass@{k}"] = v
        for tau in ("0.25", "0.5", "0.75", "1.0"):
            cols[f"gpass@{k}_t{tau}"] = max(0.0, v - float(tau) * 0.3)
        cols[f"mgpass@{k}"] = max(0.0, v - 0.15)
    return cols


def _row(model, state, family, size, is_base, benchmark, judge=None) -> dict:
    judged = judge is not None
    base_pass = 0.15 + 0.02 * size + (0.05 if state == "think" else 0.0)
    row = dict.fromkeys(META_COLUMNS, pd.NA)
    row.update(
        model=model,
        model_short=model.replace("Qwen3.5-", "Q-").replace("gemma-4-", "G4-"),
        model_family=family,
        model_size_b=float(size),
        is_base=is_base,
        state=state,
        mode={"base": "Pretrained", "non-think": "Instruct · Non-Think", "think": "Reasoning · Think"}[state],
        benchmark=benchmark,
        language={"aime2026": "EN", "aime2026_pt": "PT", "aime2026_tr": "TR", "tubitak_math2026": "TR-OL"}[benchmark],
        judged=judged,
        series=(f"cot:{judge}" if judged else "No-Judge"),
        judge_model=(judge if judged else pd.NA),
        judge_state=("think" if judged else pd.NA),
    )
    row.update(_metric_payload(base_pass, judged))
    return row


@pytest.fixture()
def wide_df() -> pd.DataFrame:
    rows: list[dict] = []
    specs = [
        ("Qwen3.5-9B-Base", "base", "Qwen", 9, True),
        ("Qwen3.5-2B-Base", "base", "Qwen", 2, True),
        ("Qwen3.5-9B", "non-think", "Qwen", 9, False),
        ("Qwen3.5-9B", "think", "Qwen", 9, False),
        ("Qwen3.5-2B", "non-think", "Qwen", 2, False),
        ("gemma-4-26B-A4B-it", "non-think", "gemma", 26, False),
        ("gemma-4-26B-A4B-it", "think", "gemma", 26, False),
    ]
    for model, state, fam, size, is_base in specs:
        for b in BENCHES:
            rows.append(_row(model, state, fam, size, is_base, b))  # No-Judge
            for j in JUDGES:
                rows.append(_row(model, state, fam, size, is_base, b, judge=j))
    cols = META_COLUMNS + metric_columns(KS)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]


# --- engine helpers --------------------------------------------------------


def test_series_extracts_sorted_kv(wide_df):
    r = wide_df[~wide_df["judged"]].iloc[0]
    s = plots.series(r, "pass", None, KS)
    assert [k for k, _ in s] == KS
    assert all(0.0 <= v <= 1.0 for _, v in s)
    # gpass with tau picks the τ column.
    sg = plots.series(r, "gpass", "0.5", KS)
    assert len(sg) == len(KS)


def test_col_mapping():
    assert plots._col("pass", 64, None) == "pass@64"
    assert plots._col("mgpass", 8, None) == "mgpass@8"
    assert plots._col("gpass", 16, "0.75") == "gpass@16_t0.75"


def test_k_axis_from_columns(wide_df):
    assert plots._k_axis(wide_df) == KS


def test_mm_label_modes():
    assert plots.mm_label("Qwen3.5-9B-Base", "base") == "Q-9B·Base"
    assert plots.mm_label("Qwen3.5-9B", "think").endswith("·TH")
    assert plots.mm_label("Qwen3.5-9B", "non-think").endswith("·NT")


# --- families --------------------------------------------------------------


@pytest.mark.parametrize(
    "fn_name",
    [
        "render_judge_effect",
        "render_bench_compare",
        "render_size_compare",
        "render_veto_curve",
        "render_mode_compare",
        "render_per_model",
        "render_tables",
        "render_headline",
        "render_comparisons",
    ],
)
def test_each_family_writes_nonempty_files(wide_df, tmp_path, fn_name):
    df = wide_df.copy()
    df["judged"] = df["judged"].astype(bool)
    ks = plots._k_axis(df)
    style = plots._judge_style(df)
    out = tmp_path / "plots"
    out.mkdir()
    paths = getattr(plots, fn_name)(df, out, ks, style)
    assert paths, f"{fn_name} produced no files"
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_render_all_covers_every_family(wide_df, tmp_path):
    written = plots.render_all(wide_df, tmp_path / "suite")
    # Every registered family ran.
    assert set(written) == set(plots._FAMILIES)
    # The RL-progression families (rl_progress, rl_table) require multi-step
    # ``@stepN`` checkpoint data, which this generic fixture intentionally omits
    # (mirroring test_each_family_writes_nonempty_files, which excludes them); for
    # such data they return [] by design, so they may be empty here.
    data_conditional = {"rl_progress", "rl_table"}
    for name, paths in written.items():
        if name in data_conditional:
            continue
        assert paths, f"family {name} wrote nothing"
    # The multilingual companion CSV is part of the contract.
    csvs = [p for p in written["comparisons"] if p.suffix == ".csv"]
    assert csvs, "expected pass_vs_cot companion CSV(s)"


def test_render_all_handles_string_judged_column(wide_df, tmp_path):
    """CSV round-trips can turn the bool discriminator into strings."""
    df = wide_df.copy()
    df["judged"] = df["judged"].map({True: "True", False: "False"})
    written = plots.render_all(df, tmp_path / "strbool")
    assert written["judge_effect"], "string 'judged' column broke No-Judge/cot split"
