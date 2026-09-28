"""compose_target_dir / compose_judge_dir collision-freedom checks.

Sources pipeline_common.sh from a subshell so we exercise the real shell
functions. Each test sets the env vars that the composers depend on, then
captures stdout.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB = REPO_ROOT / "scripts" / "lib" / "pipeline_common.sh"


def _run(env: dict, fn: str, *args: str) -> str:
    cmd = f'source "{LIB}" && {fn} ' + " ".join(f'"{a}"' for a in args)
    full_env = {**os.environ, **env}
    proc = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
        env=full_env,
        check=True,
    )
    return proc.stdout.strip()


BASE_ENV = {
    "OUTPUT_ROOT": "results",
    "TARGET_MODEL": "Qwen/Qwen3.5-0.8B-Base",
    "TARGET_STATE": "base",
    "TARGET_TEMPERATURE": "0.6",
    "TARGET_MAX_COMPLETION_TOKENS": "16384",
    "TARGET_N_SAMPLES": "64",
    "JUDGE_MODEL": "Qwen/Qwen3.6-35B-A3B",
    "JUDGE_STATE": "think",
    "JUDGE_TEMPERATURE": "0.6",
    "JUDGE_MAX_COMPLETION_TOKENS": "16384",
    "JUDGE_N_SAMPLES": "3",
}


def test_target_dir_encodes_all_params():
    out = _run(BASE_ENV, "compose_target_dir", "aime2026")
    # V5 layout: one folder per model; the sampling suffix lives on the benchmark leaf.
    assert out == "results/base/Qwen3.5-0.8B-Base/aime2026__t0.6__max16384__n64"


def test_judge_dir_nests_under_target():
    out = _run(BASE_ENV, "compose_judge_dir", "aime2026")
    # V5 layout: model dir is bare; the judge keeps its own t/max; the benchmark
    # leaf carries the *target* sampling suffix (so base and judged_by stay aligned).
    # basemax always records the target's max_completion_tokens (may differ from
    # the judge's own); __re-/__eb- are omitted entirely when unset/"none", no
    # dead "__re-none__eb-none" noise in the common case.
    assert out == (
        "results/base/Qwen3.5-0.8B-Base/"
        "judged_by/Qwen3.6-35B-A3B__state-think__t0.6__max16384__basemax16384/"
        "aime2026__t0.6__max16384__n64"
    )


def test_judge_dir_omits_re_eb_when_explicitly_none():
    out = _run(
        {**BASE_ENV, "JUDGE_REASONING_EFFORT": "none", "JUDGE_EXTRA_BODY_TAG": "none"},
        "compose_judge_dir",
        "aime2026",
    )
    assert "__re-" not in out
    assert "__eb-" not in out
    assert "__basemax16384" in out


def test_judge_dir_keeps_basemax_when_it_differs_from_judge_max():
    out = _run(
        {**BASE_ENV, "JUDGE_MAX_COMPLETION_TOKENS": "32768", "TARGET_MAX_COMPLETION_TOKENS": "65536"},
        "compose_judge_dir",
        "aime2026",
    )
    assert "__max32768__basemax65536" in out
    assert "max65536__n64" in out.rsplit("/", 1)[-1]


def test_judge_dir_encodes_reasoning_and_extra_body():
    out = _run(
        {**BASE_ENV, "JUDGE_REASONING_EFFORT": "high", "JUDGE_EXTRA_BODY_TAG": "thinking-enabled"},
        "compose_judge_dir",
        "aime2026",
    )
    assert out == (
        "results/base/Qwen3.5-0.8B-Base/"
        "judged_by/Qwen3.6-35B-A3B__state-think__t0.6__max16384__basemax16384__re-high__eb-thinking-enabled/"
        "aime2026__t0.6__max16384__n64"
    )


def test_judge_dirs_differ_when_reasoning_effort_differs():
    out_a = _run({**BASE_ENV, "JUDGE_REASONING_EFFORT": "high"}, "compose_judge_dir", "aime2026")
    out_b = _run({**BASE_ENV, "JUDGE_REASONING_EFFORT": "none"}, "compose_judge_dir", "aime2026")
    assert out_a != out_b


def test_target_dirs_differ_when_n_samples_differs():
    out_a = _run({**BASE_ENV, "TARGET_N_SAMPLES": "8"}, "compose_target_dir", "aime2026")
    out_b = _run({**BASE_ENV, "TARGET_N_SAMPLES": "64"}, "compose_target_dir", "aime2026")
    assert out_a != out_b


def test_judge_dirs_differ_when_judge_state_differs():
    out_a = _run({**BASE_ENV, "JUDGE_STATE": "think"}, "compose_judge_dir", "aime2026")
    out_b = _run({**BASE_ENV, "JUDGE_STATE": "non-think"}, "compose_judge_dir", "aime2026")
    assert out_a != out_b


def test_same_tuple_same_path():
    """Same params must yield the same path (idempotent re-run)."""
    out_a = _run(BASE_ENV, "compose_judge_dir", "aime2026")
    out_b = _run(BASE_ENV, "compose_judge_dir", "aime2026")
    assert out_a == out_b
