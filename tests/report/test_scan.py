"""Tests for :mod:`evalhub.report.scan`."""

from __future__ import annotations

from evalhub.report.scan import (
    parse_base_dirname,
    parse_benchmark_leaf,
    parse_judge_dirname,
    parse_judge_leaf_dirname,
    scan_results,
)


def test_parse_benchmark_leaf_v5():
    assert parse_benchmark_leaf("aime2026__t0.6__max16384__n64") == ("aime2026", 0.6, 16384, 64)
    # Single-underscore benchmark names must not be split at the wrong place.
    assert parse_benchmark_leaf("aime2026_tr__t0.6__max20480__n16") == ("aime2026_tr", 0.6, 20480, 16)
    assert parse_benchmark_leaf("tubitak_math2026__t0.6__max16384__n64") == (
        "tubitak_math2026", 0.6, 16384, 64,
    )


def test_parse_benchmark_leaf_bare_is_none():
    # A bare benchmark (old layouts) carries no sampling suffix → None (caller falls back).
    assert parse_benchmark_leaf("aime2026") is None
    assert parse_benchmark_leaf("tubitak_math2026") is None
    # A model dir name must never be mistaken for a benchmark leaf.
    assert parse_benchmark_leaf("DAPO-EN-Qwen3.5-2B-Base-t64g192-n8-m4") is None


def test_scan_results_v5_one_folder_per_model(v5_results_root):
    records = scan_results(v5_results_root)
    by_type = {r.eval_type: r for r in records}
    assert set(by_type) == {"base_eval", "cot_eval"}

    base = by_type["base_eval"]
    assert base.model == "qwen-mini"          # bare model dir name
    assert base.state == "base"               # from the state dir above the model
    assert base.benchmark == "gsm8k"          # suffix stripped off the leaf
    assert base.temperature == 0.6
    assert base.max_completion_tokens == 2048
    assert base.n_samples == 64               # from the benchmark leaf, not the model dir
    assert base.pass_at_k == {1: 0.50, 2: 0.65, 4: 0.78}
    assert base.stats is not None and base.stats["true_count"] == 1600

    cot = by_type["cot_eval"]
    assert cot.model == "qwen-mini"
    assert cot.state == "base"
    assert cot.benchmark == "gsm8k"
    assert cot.n_samples == 64                # target n from the benchmark leaf
    assert cot.judge_model == "qwen-judge"
    assert cot.judge_state == "think"
    assert cot.judge_max_completion_tokens == 16384
    assert cot.pass_at_k == {1: 0.40, 2: 0.55, 4: 0.70}
    assert cot.stats is not None and cot.stats["cot_false_count"] == 400


def test_parse_base_dirname_canonical():
    parsed = parse_base_dirname("qwen-mini_state-non-think_t0.6_max2048")
    assert parsed is not None
    assert parsed.model == "qwen-mini"
    assert parsed.state == "non-think"
    assert parsed.temperature == 0.6
    assert parsed.max_completion_tokens == 2048


def test_parse_base_dirname_legacy_fallback():
    parsed = parse_base_dirname("gemma-4-E4B_t0.6_max16384")
    assert parsed is not None
    assert parsed.model == "gemma-4-E4B"
    assert parsed.state == "unknown"
    assert parsed.temperature == 0.6
    assert parsed.max_completion_tokens == 16384


def test_parse_base_dirname_rejects_garbage():
    assert parse_base_dirname("totally_unrelated_directory_name") is None


def test_parse_judge_dirname_canonical():
    name = "qwen-mini_state-non-think_judged_by_qwen-judge_state-think_t0.6_max16384"
    parsed = parse_judge_dirname(name)
    assert parsed is not None
    assert parsed.target_model == "qwen-mini"
    assert parsed.target_state == "non-think"
    assert parsed.judge_model == "qwen-judge"
    assert parsed.judge_state == "think"
    assert parsed.max_completion_tokens == 16384


def test_scan_results_returns_one_base_and_one_cot(fake_results_root):
    records = scan_results(fake_results_root)
    by_type = {r.eval_type: r for r in records}
    assert set(by_type) == {"base_eval", "cot_eval"}

    base = by_type["base_eval"]
    assert base.model == "qwen-mini"
    assert base.benchmark == "gsm8k"
    assert base.state == "non-think"
    assert base.pass_at_k == {1: 0.50, 2: 0.65, 4: 0.78}
    assert base.cons_at_k == 0.55
    assert base.stats is None

    cot = by_type["cot_eval"]
    assert cot.model == "qwen-mini"
    assert cot.judge_model == "qwen-judge"
    assert cot.judge_state == "think"
    assert cot.pass_at_k == {1: 0.40, 2: 0.55, 4: 0.70}
    assert cot.stats is not None
    assert cot.stats["cot_false_count"] == 20


def test_scan_results_handles_legacy_layout(legacy_results_root):
    records = scan_results(legacy_results_root)
    assert len(records) == 1
    rec = records[0]
    assert rec.eval_type == "base_eval"
    assert rec.model == "gemma-4-E4B"
    assert rec.state == "unknown"
    assert rec.benchmark == "aime2026"


def test_parse_v3_leaf_uses_parent_state():
    parsed = parse_base_dirname("qwen-mini__t0.6__max2048__n64", parent_state="think")
    assert parsed is not None
    assert parsed.model == "qwen-mini"
    assert parsed.state == "think"
    assert parsed.n_samples == 64


def test_parse_v3_leaf_unknown_state_when_no_parent():
    parsed = parse_base_dirname("qwen-mini__t0.6__max2048__n64")
    assert parsed is not None
    assert parsed.state == "unknown"


def test_parse_judge_leaf_v4_no_n_samples():
    parsed = parse_judge_leaf_dirname("qwen-judge__state-think__t0.6__max16384")
    assert parsed is not None
    assert parsed.model == "qwen-judge"
    assert parsed.state == "think"
    assert parsed.temperature == 0.6
    assert parsed.max_completion_tokens == 16384
    assert parsed.n_samples is None


def test_parse_judge_leaf_v2_with_n_samples_still_works():
    parsed = parse_judge_leaf_dirname("qwen-judge__state-think__t0.6__max16384__n3")
    assert parsed is not None
    assert parsed.model == "qwen-judge"
    assert parsed.state == "think"
    assert parsed.n_samples == 3


def test_scan_results_v3_state_hoisted(v3_results_root):
    records = scan_results(v3_results_root)
    by_type = {r.eval_type: r for r in records}
    assert set(by_type) == {"base_eval", "cot_eval"}
    base = by_type["base_eval"]
    assert base.model == "qwen-mini"
    assert base.state == "base"  # from parent dir, not leaf
    assert base.n_samples == 64
    cot = by_type["cot_eval"]
    assert cot.model == "qwen-mini"
    assert cot.state == "base"
    assert cot.judge_model == "qwen-judge"
    assert cot.judge_state == "think"
    assert cot.judge_n_samples == 3


def test_parse_base_dirname_v2_with_n_samples():
    parsed = parse_base_dirname("qwen-mini__state-base__t0.6__max2048__n64")
    assert parsed is not None
    assert parsed.model == "qwen-mini"
    assert parsed.state == "base"
    assert parsed.temperature == 0.6
    assert parsed.max_completion_tokens == 2048
    assert parsed.n_samples == 64


def test_scan_results_v2_nested_layout(v2_results_root):
    records = scan_results(v2_results_root)
    by_type = {r.eval_type: r for r in records}
    assert set(by_type) == {"base_eval", "cot_eval"}

    base = by_type["base_eval"]
    assert base.model == "qwen-mini"
    assert base.state == "base"
    assert base.n_samples == 64
    assert base.benchmark == "gsm8k"
    # Summary aggregate counts populate record.stats automatically.
    assert base.stats is not None
    assert base.stats["true_count"] == 1600
    assert base.stats["invalid_count"] == 0

    cot = by_type["cot_eval"]
    assert cot.model == "qwen-mini"
    assert cot.judge_model == "qwen-judge"
    assert cot.judge_state == "think"
    assert cot.n_samples == 64
    assert cot.judge_n_samples == 3
    assert cot.stats is not None
    assert cot.stats["cot_false_count"] == 400
