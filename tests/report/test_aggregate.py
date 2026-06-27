"""Tests for the wide master-CSV aggregator :mod:`evalhub.report.aggregate`."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from evalhub.report.aggregate import (  # noqa: E402
    META_COLUMNS,
    TAUS,
    aggregate_results,
    build_wide_dataframe,
    metric_columns,
    wide_row_from_record,
    write_csv,
)
from evalhub.report.scan import scan_results  # noqa: E402


def test_meta_columns_lead_and_key_present():
    assert META_COLUMNS[0] == "model"
    for col in ("model_short", "is_base", "mode", "language", "judged", "series",
                "judge_model", "judge_state"):
        assert col in META_COLUMNS


def test_metric_columns_cover_all_k_and_tau():
    cols = metric_columns([1, 64])
    assert "pass@64" in cols
    assert "mgpass@64" in cols
    for t in TAUS:
        assert f"gpass@64_t{t}" in cols
    # Grouped per K: pass, then 4 g-pass τ, then mg → 6 columns per K.
    assert len(cols) == 2 * (1 + len(TAUS) + 1)


def test_one_row_per_run_with_metrics(fake_results_root):
    records = scan_results(fake_results_root)
    df = build_wide_dataframe(records)
    # fixture: 1 base + 1 cot run -> exactly 2 rows (NOT exploded per K).
    assert len(df) == 2
    # Pass@k landed in wide columns, not a long 'k' column.
    assert "k" not in df.columns
    assert df["pass@1"].notna().all()
    assert df["pass@4"].notna().all()
    # No-Judge vs judged discriminator.
    nj = df[~df["judged"]]
    jd = df[df["judged"]]
    assert len(nj) == 1 and len(jd) == 1
    assert nj.iloc[0]["judge_model"] in (None, "") or pd.isna(nj.iloc[0]["judge_model"])
    assert (nj["series"] == "No-Judge").all()
    assert jd.iloc[0]["series"].startswith("cot:")
    # cot row carries the veto stats.
    assert jd.iloc[0]["cot_false_count"] == 20


def test_unique_key_no_duplicates(v3_results_root):
    df = aggregate_results(v3_results_root, v3_results_root / "out.csv")
    key = ["model", "state", "benchmark", "judge_model", "judge_state"]
    assert df.duplicated(key).sum() == 0


def test_wide_row_flattens_gpass(fake_results_root):
    # Inject a record with g_pass/mg_pass and confirm tau columns appear.
    records = scan_results(fake_results_root)
    rec = records[0]
    object.__setattr__(rec, "g_pass_at_k", {"4": {"0.25": 0.7, "0.5": 0.6, "0.75": 0.5, "1.0": 0.3}})
    object.__setattr__(rec, "mg_pass_at_k", {"4": 0.42})
    row = wide_row_from_record(rec)
    assert row["gpass@4_t0.5"] == 0.6
    assert row["gpass@4_t1.0"] == 0.3
    assert row["mgpass@4"] == 0.42


def test_aggregate_results_writes_csv(fake_results_root, tmp_path):
    out = tmp_path / "nested" / "report.csv"
    df = aggregate_results(fake_results_root, out)
    assert out.exists()
    reloaded = pd.read_csv(out)
    assert len(reloaded) == len(df)
    assert list(reloaded.columns) == list(df.columns)


def test_write_csv_handles_missing_parent(tmp_path):
    df = pd.DataFrame({col: [] for col in META_COLUMNS})
    target = tmp_path / "deeper" / "out.csv"
    assert write_csv(df, target) == target
    assert target.exists()
