"""Tests for :mod:`evalhub.report.aggregate`."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from evalhub.report.aggregate import (  # noqa: E402  (import gated by pytest.importorskip)
    LONG_COLUMNS,
    aggregate_results,
    build_dataframe,
    write_csv,
)
from evalhub.report.scan import scan_results  # noqa: E402


def test_long_columns_are_canonical_order():
    assert LONG_COLUMNS[0:7] == [
        "model",
        "state",
        "temperature",
        "max_tokens",
        "n_samples",
        "benchmark",
        "eval_type",
    ]
    assert "pass_at_k" in LONG_COLUMNS
    assert "cot_false_count" in LONG_COLUMNS
    assert "n_samples" in LONG_COLUMNS
    assert "judge_n_samples" in LONG_COLUMNS


def test_build_dataframe_explodes_pass_at_k(fake_results_root):
    records = scan_results(fake_results_root)
    df = build_dataframe(records)
    # 3 K values per record × 2 records = 6 rows.
    assert len(df) == 6
    assert list(df.columns) == LONG_COLUMNS
    # Base rows must have NaN stats columns.
    base_rows = df[df["eval_type"] == "base_eval"]
    assert base_rows["cot_false_count"].isna().all()
    # CoT rows preserve the stats payload on every K.
    cot_rows = df[df["eval_type"] == "cot_eval"]
    assert (cot_rows["cot_false_count"] == 20).all()
    assert (cot_rows["judge_model"] == "qwen-judge").all()


def test_aggregate_results_writes_csv_with_header(fake_results_root, tmp_path):
    out_csv = tmp_path / "out" / "report.csv"
    df = aggregate_results(fake_results_root, out_csv)
    assert out_csv.exists()
    reloaded = pd.read_csv(out_csv)
    assert list(reloaded.columns) == LONG_COLUMNS
    assert len(reloaded) == len(df)


def test_write_csv_handles_missing_parent(tmp_path):
    df = pd.DataFrame({col: [] for col in LONG_COLUMNS})
    target = tmp_path / "nested" / "deeper" / "out.csv"
    written = write_csv(df, target)
    assert written == target
    assert target.exists()


def test_aggregate_results_empty_pass_at_k_keeps_one_stub_row(tmp_path):
    """The 'no base-correct samples' summary still surfaces in the CSV."""
    root = tmp_path / "results"
    benchmark_dir = (
        root
        / "judgments"
        / "qwen-mini_state-non-think_judged_by_qwen-judge_state-think_t0.6_max16384"
        / "aime2026"
    )
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "aime2026_cot_summary.json").write_text(
        '{"pass_at_k": {}, "cons_at_k": 0.0, "note": "no base-correct samples"}'
    )
    csv = tmp_path / "stub.csv"
    df = aggregate_results(root, csv)
    assert len(df) == 1
    assert df.iloc[0]["k"] is None or pd.isna(df.iloc[0]["k"])
