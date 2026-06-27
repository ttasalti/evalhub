"""Shared fixtures for the report test suite.

Builds an on-the-fly results tree that mirrors what
``scripts/run_end_to_end.sh`` produces, so the scanner sees realistic inputs
without us having to commit binary artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture()
def fake_results_root(tmp_path: Path) -> Path:
    """Two-run fixture: one base eval + one CoT-vetted eval, same model/benchmark."""
    root = tmp_path / "results"

    base_dir = root / "qwen-mini_state-non-think_t0.6_max2048" / "gsm8k"
    _write_json(
        base_dir / "gsm8k_summary.json",
        {"pass_at_k": {"1": 0.50, "2": 0.65, "4": 0.78}, "cons_at_k": 0.55},
    )

    judge_dir = (
        root
        / "judgments"
        / "qwen-mini_state-non-think_judged_by_qwen-judge_state-think_t0.6_max16384"
        / "gsm8k"
    )
    _write_json(
        judge_dir / "gsm8k_cot_summary.json",
        {"pass_at_k": {"1": 0.40, "2": 0.55, "4": 0.70}, "cons_at_k": 0.50},
    )
    _write_json(
        judge_dir / "gsm8k_cot_stats.json",
        {
            "total_tasks": 50,
            "total_generations": 200,
            "true_count": 80,
            "false_count": 90,
            "cot_false_count": 20,
            "invalid_count": 10,
        },
    )
    return root


@pytest.fixture()
def legacy_results_root(tmp_path: Path) -> Path:
    """Directory using the pre-`state-` legacy naming."""
    root = tmp_path / "legacy"
    legacy_dir = root / "gemma-4-E4B_t0.6_max16384" / "aime2026"
    _write_json(
        legacy_dir / "aime2026_summary.json",
        {"pass_at_k": {"1": 0.0, "2": 0.0, "4": 0.0}, "cons_at_k": 0.0},
    )
    return root


@pytest.fixture()
def v3_results_root(tmp_path: Path) -> Path:
    """V3 layout: state hoisted to a top-level dir, leaf carries model + knobs only."""
    root = tmp_path / "v3"
    target_root = root / "base" / "qwen-mini__t0.6__max2048__n64"
    base_dir = target_root / "gsm8k"
    _write_json(
        base_dir / "gsm8k_summary.json",
        {
            "pass_at_k": {"1": 0.50, "2": 0.65, "4": 0.78},
            "cons_at_k": 0.55,
            "total_tasks": 50,
            "total_generations": 3200,
            "true_count": 1600,
            "false_count": 1600,
            "cot_false_count": 0,
            "invalid_format_count": 0,
        },
    )
    judge_dir = (
        target_root
        / "judged_by"
        / "qwen-judge__state-think__t0.6__max16384__n3"
        / "gsm8k"
    )
    _write_json(
        judge_dir / "gsm8k_cot_summary.json",
        {
            "pass_at_k": {"1": 0.40, "2": 0.55, "4": 0.70},
            "cons_at_k": 0.50,
            "total_tasks": 50,
            "total_generations": 3200,
            "true_count": 1200,
            "false_count": 1600,
            "cot_false_count": 400,
            "invalid_format_count": 0,
        },
    )
    return root


@pytest.fixture()
def v5_results_root(tmp_path: Path) -> Path:
    """V5 layout (current): ONE folder per model — model dir is bare, the sampling
    suffix lives on the benchmark leaf, on both the base side and under judged_by/."""
    root = tmp_path / "v5"
    model_root = root / "base" / "qwen-mini"           # bare model dir, no suffix
    base_dir = model_root / "gsm8k__t0.6__max2048__n64"  # suffix on the benchmark leaf
    _write_json(
        base_dir / "gsm8k_summary.json",
        {
            "pass_at_k": {"1": 0.50, "2": 0.65, "4": 0.78},
            "cons_at_k": 0.55,
            "total_tasks": 50,
            "total_generations": 3200,
            "true_count": 1600,
            "false_count": 1600,
            "cot_false_count": 0,
            "invalid_format_count": 0,
        },
    )
    judge_dir = (
        model_root
        / "judged_by"
        / "qwen-judge__state-think__t0.6__max16384"
        / "gsm8k__t0.6__max2048__n64"                   # same target suffix on the leaf
    )
    _write_json(
        judge_dir / "gsm8k_cot_summary.json",
        {
            "pass_at_k": {"1": 0.40, "2": 0.55, "4": 0.70},
            "cons_at_k": 0.50,
            "total_tasks": 50,
            "total_generations": 3200,
            "true_count": 1200,
            "false_count": 1600,
            "cot_false_count": 400,
            "invalid_format_count": 0,
        },
    )
    return root


@pytest.fixture()
def v2_results_root(tmp_path: Path) -> Path:
    """V2 layout: target → judged_by → judge → benchmark, with n_samples + aggregate counts."""
    root = tmp_path / "v2"
    target_root = root / "qwen-mini__state-base__t0.6__max2048__n64"
    base_dir = target_root / "gsm8k"
    _write_json(
        base_dir / "gsm8k_summary.json",
        {
            "pass_at_k": {"1": 0.50, "2": 0.65, "4": 0.78},
            "cons_at_k": 0.55,
            "total_tasks": 50,
            "total_generations": 3200,
            "true_count": 1600,
            "false_count": 1600,
            "cot_false_count": 0,
            "invalid_format_count": 0,
        },
    )

    judge_dir = (
        target_root
        / "judged_by"
        / "qwen-judge__state-think__t0.6__max16384__n3"
        / "gsm8k"
    )
    _write_json(
        judge_dir / "gsm8k_cot_summary.json",
        {
            "pass_at_k": {"1": 0.40, "2": 0.55, "4": 0.70},
            "cons_at_k": 0.50,
            "total_tasks": 50,
            "total_generations": 3200,
            "true_count": 1200,
            "false_count": 1600,
            "cot_false_count": 400,
            "invalid_format_count": 0,
        },
    )
    return root
