"""Smoke tests for the highlights PDF :mod:`evalhub.report.highlights`.

Reuses the schema-faithful wide DataFrame fixture from ``test_plots`` so the
matched-pair logic and every page exercise realistic columns. Skipped when the
plotting stack is absent.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from evalhub.report import highlights as H  # noqa: E402
from tests.report.test_plots import wide_df  # noqa: E402,F401  (fixture import)


def test_matched_pairs_pairs_nojudge_with_cot(wide_df):
    mp = H._matched_pairs(wide_df)
    assert not mp.empty
    # every row carries a No-Judge and a cot value plus their difference.
    assert (mp["delta"] == (mp["nojudge"] - mp["cot"])).all()
    # both lenient and strict metrics made it in.
    assert {"pass", "gpass", "mgpass"} <= set(mp["metric"])
    # judged side only — judge always present and labelled think.
    assert mp["judge"].notna().all()


def test_matched_pairs_handles_string_judged_column(wide_df):
    df = wide_df.copy()
    df["judged"] = df["judged"].map({True: "True", False: "False"})
    mp = H._matched_pairs(df)
    assert not mp.empty


def test_build_highlights_writes_multipage_pdf(wide_df, tmp_path):
    out = tmp_path / "nested" / "highlights.pdf"
    result = H.build_highlights(wide_df, out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0
    # one page per registered page function.
    data = out.read_bytes()
    n_pages = data.count(b"/Type /Page") - data.count(b"/Type /Pages")
    assert n_pages == len(H._PAGES)


def test_build_highlights_rejects_long_format(tmp_path):
    """A non-wide CSV (no matched pairs) should fail loudly, not write junk."""
    df = pd.DataFrame({"model": ["m"], "state": ["base"], "benchmark": ["b"],
                       "judged": [False], "judge_model": [None], "judge_state": [None],
                       "model_size_b": [1.0], "is_base": [True], "model_family": ["Qwen"],
                       "judge_n_samples": [None]})
    with pytest.raises(ValueError):
        H.build_highlights(df, tmp_path / "x.pdf")
