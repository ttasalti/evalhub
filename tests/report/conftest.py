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
