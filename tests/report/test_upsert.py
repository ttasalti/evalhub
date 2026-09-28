"""Tests for incremental single-row upsert into the wide master CSV."""

from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")

from evalhub.report.aggregate import upsert_summary  # noqa: E402

KEY = ["model", "state", "benchmark", "judge_model", "judge_state"]


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _base_summary(root, model_dir, benchmark, payload):
    p = root / model_dir / benchmark / f"{benchmark}_summary.json"
    _write(p, payload)
    return p


def test_upsert_creates_then_idempotent(tmp_path):
    root = tmp_path / "results"
    summary = _base_summary(
        root,
        "base/qwen-mini__t0.6__max2048__n64",
        "gsm8k",
        {"pass_at_k": {"1": 0.5, "2": 0.65}, "cons_at_k": 0.5},
    )
    csv = tmp_path / "report.csv"

    upsert_summary(summary, csv, root)
    df1 = pd.read_csv(csv)
    assert len(df1) == 1
    assert df1.iloc[0]["pass@1"] == 0.5
    assert not bool(df1.iloc[0]["judged"])
    # report_tasks.csv is a separate file; its task_id/n_generations columns
    # (the two names genuinely unique to it -- the count columns already
    # existed in report.csv as whole-run aggregates) must not leak in.
    for col in ("task_id", "n_generations"):
        assert col not in df1.columns

    # Re-run same summary -> still ONE row (replace, not duplicate).
    upsert_summary(summary, csv, root)
    df2 = pd.read_csv(csv)
    assert len(df2) == 1
    assert df2.duplicated(KEY).sum() == 0


def test_upsert_updates_changed_values(tmp_path):
    root = tmp_path / "results"
    mdir = "base/qwen-mini__t0.6__max2048__n64"
    summary = _base_summary(root, mdir, "gsm8k", {"pass_at_k": {"1": 0.5}, "cons_at_k": 0.5})
    csv = tmp_path / "report.csv"
    upsert_summary(summary, csv, root)

    # Overwrite the summary with a new value, upsert again.
    _write(summary, {"pass_at_k": {"1": 0.9}, "cons_at_k": 0.7})
    upsert_summary(summary, csv, root)
    df = pd.read_csv(csv)
    assert len(df) == 1
    assert df.iloc[0]["pass@1"] == 0.9


def test_upsert_distinct_keys_append(tmp_path):
    root = tmp_path / "results"
    csv = tmp_path / "report.csv"
    a = _base_summary(root, "base/qwen-mini__t0.6__max2048__n64", "gsm8k", {"pass_at_k": {"1": 0.5}, "cons_at_k": 0.5})
    b = _base_summary(
        root, "base/qwen-mini__t0.6__max2048__n64", "aime2026", {"pass_at_k": {"1": 0.1}, "cons_at_k": 0.1}
    )
    upsert_summary(a, csv, root)
    upsert_summary(b, csv, root)
    df = pd.read_csv(csv)
    assert len(df) == 2  # different benchmark -> distinct key
    assert set(df["benchmark"]) == {"gsm8k", "aime2026"}


def test_upsert_caps_k_like_aggregate(tmp_path):
    """upsert must trim K > REPORT_MAX_K exactly like aggregate_results, so the
    incremental and full-rescan paths never diverge for the same input."""
    root = tmp_path / "results"
    summary = _base_summary(
        root,
        "base/qwen-mini__t0.6__max2048__n64",
        "gsm8k",
        {"pass_at_k": {"1": 0.5, "64": 0.8, "128": 0.9}, "cons_at_k": 0.5},
    )
    csv = tmp_path / "report.csv"

    upsert_summary(summary, csv, root)  # default max_k=REPORT_MAX_K=64
    df = pd.read_csv(csv)
    assert len(df) == 1
    assert df.iloc[0]["pass@64"] == 0.8
    assert "pass@128" not in df.columns or pd.isna(df.iloc[0]["pass@128"])


def test_upsert_honors_exclude_patterns(tmp_path):
    root = tmp_path / "results"
    summary = _base_summary(
        root,
        "base/qwen-mini__t0.6__max2048__n64",
        "gsm8k",
        {"pass_at_k": {"1": 0.5}, "cons_at_k": 0.5},
    )
    csv = tmp_path / "report.csv"

    out = upsert_summary(summary, csv, root, exclude_patterns=("qwen-mini",))
    assert out is None
    assert not csv.exists()


def test_upsert_max_tokens_disambiguates(tmp_path):
    """Regression test: two judged runs of the same model/state/benchmark/judge
    at different max_tokens (e.g. a 16k and a 32k think-mode re-run of the same
    model) must both persist as distinct rows, not collide into one -- this was
    a real bug found in production (KEY_COLUMNS didn't include max_tokens, so
    upserting the second silently deleted the first's row)."""
    root = tmp_path / "results"
    csv = tmp_path / "report.csv"
    target16 = root / "base" / "qwen-mini__t0.6__max16384__n64"
    target32 = root / "base" / "qwen-mini__t0.6__max32768__n64"
    judged16 = (
        target16
        / "judged_by"
        / "qwen-judge__state-think__t0.6__max16384__basemax16384"
        / "gsm8k"
        / "gsm8k_cot_summary.json"
    )
    judged32 = (
        target32
        / "judged_by"
        / "qwen-judge__state-think__t0.6__max16384__basemax32768"
        / "gsm8k"
        / "gsm8k_cot_summary.json"
    )
    _write(judged16, {"pass_at_k": {"1": 0.5}, "cons_at_k": 0.5})
    _write(judged32, {"pass_at_k": {"1": 0.3}, "cons_at_k": 0.3})

    upsert_summary(judged16, csv, root)
    upsert_summary(judged32, csv, root)
    df = pd.read_csv(csv)
    assert len(df) == 2  # both max_tokens variants coexist
    assert set(df["max_tokens"]) == {16384, 32768}
    assert (df["judge_model"] == "qwen-judge").all()


def test_upsert_judge_empty_vs_filled(tmp_path):
    """A No-Judge row and a judged row for the same run are distinct rows."""
    root = tmp_path / "results"
    csv = tmp_path / "report.csv"
    target = root / "base" / "qwen-mini__t0.6__max2048__n64"
    nojudge = target / "gsm8k" / "gsm8k_summary.json"
    _write(nojudge, {"pass_at_k": {"1": 0.5}, "cons_at_k": 0.5})
    judged = target / "judged_by" / "qwen-judge__state-think__t0.6__max16384__n3" / "gsm8k" / "gsm8k_cot_summary.json"
    _write(judged, {"pass_at_k": {"1": 0.4}, "cons_at_k": 0.4})

    upsert_summary(nojudge, csv, root)
    upsert_summary(judged, csv, root)
    df = pd.read_csv(csv)
    assert len(df) == 2
    assert (~df["judged"]).sum() == 1 and df["judged"].sum() == 1
    jrow = df[df["judged"]].iloc[0]
    assert jrow["judge_model"] == "qwen-judge"
    assert jrow["series"].startswith("cot:")
