"""Configuration for the error-injection judge study.

Single source of truth for paths, the model/benchmark grid, sampling knobs, the
wrong-number menu distribution, and the DeepSeek think-mode judge config.

Nothing here writes to disk; everything is read by the stage scripts. The study
is strictly read-only against ``results/``, the raw generations are only ever
opened for reading.
"""

from __future__ import annotations

import os

# paths
# Repo root = parent of this package's directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO_ROOT, "results", "think")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# the grid
MODELS = ["Qwen3.5-4B", "Qwen3.5-9B"]

# Sample ONLY from the highest max-token variant of each benchmark (no
# pooling across token budgets). At lower caps most think-mode
# solutions truncate (finish_reason=length -> no boxed -> not correct).
BENCHMARKS = ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026", "pt_exams_math"]
HIGHEST_VARIANT = {
    "aime2026": 65536,
    "aime2026_pt": 32768,
    "aime2026_tr": 32768,
    "tubitak_math2026": 32768,
    "pt_exams_math": 16384,
}
# n_samples in the leaf dir name (base run), needed only to locate the leaf.
LEAF_N_SAMPLES = {
    "aime2026": 64,
    "aime2026_pt": 64,
    "aime2026_tr": 64,
    "tubitak_math2026": 64,
    "pt_exams_math": 16,
}
LEAF_TEMPERATURE = "0.6"

# Judge prompt language / task per benchmark (question language).
JUDGE_TASK = {
    "aime2026": "cot_judge",  # English
    "aime2026_pt": "cot_judge_pt",  # Portuguese
    "aime2026_tr": "cot_judge_tr",  # Turkish
    "tubitak_math2026": "cot_judge_tr",  # Turkish
    "pt_exams_math": "cot_judge_pt",  # Portuguese
}

# sampling knobs
SEED = 42
MIN_TOKENS = 600  # keep solutions strictly longer than this
PER_GROUP = 150  # rows sampled per (model, benchmark) group
MAX_SOLS_PER_Q = 6  # <= this many solutions per question
MIN_SOLS_PER_Q = 1  # >= this many for any included question

# error injection
ERROR_TYPES = ["clean", "boxed_only", "intermediate_error", "consistent_error", "truncated"]
# Wrong-number menu target distribution (over draws that need a wrong number).
MENU_DISTRIBUTION = {"pm12": 0.40, "digit_swap": 0.30, "single_digit": 0.30}
MENU_ORDER = ["pm12", "digit_swap", "single_digit"]  # fallback order
# Spec says "digit-swap must be 2-digit (47->74)". False (default) = generalized
# transposition of any-length number (better serves the 30% target on a 3-digit-heavy
# corpus). True = literal spec: only 2-digit numbers qualify, 3+ digits fall through.
DIGIT_SWAP_STRICT_2DIGIT = False
TYPE2_REGION = (0.40, 0.70)  # fractional char-region for the intermediate error
TYPE4_KEEP_FRACTION = 0.75  # keep the first 75%, drop the last 25%
AIME_MIN, AIME_MAX = 0, 999  # integer answer range for AIME-family benchmarks
AIME_BENCHMARKS = {"aime2026", "aime2026_pt", "aime2026_tr"}

# DeepSeek think-mode judge (for build_judge_inputs / run / cost)
JUDGE_MODEL = "deepseek-v4-flash"
JUDGE_API_BASE = "https://api.deepseek.com/v1"
JUDGE_STATE = "think"
JUDGE_REASONING_EFFORT = "high"
JUDGE_EXTRA_BODY = '{"thinking": {"type": "enabled"}}'
JUDGE_EXTRA_BODY_TAG = "thinking-enabled"
JUDGE_MAX_COMPLETION_TOKENS = 20480
JUDGE_TEMPERATURE = "0.6"
JUDGE_TOP_P = "0.95"
JUDGE_N_SAMPLES = 3  # 3-generation majority
# DeepSeek pricing (USD per token).
PRICE_INPUT_MISS = 0.14 / 1e6
PRICE_INPUT_HIT = 0.0028 / 1e6
PRICE_OUTPUT = 0.28 / 1e6


def group_name(model: str, benchmark: str) -> str:
    return f"{model}|{benchmark}"


def leaf_dir(model: str, benchmark: str) -> str:
    """Absolute path to the highest-variant base-run leaf for a group."""
    mx = HIGHEST_VARIANT[benchmark]
    ns = LEAF_N_SAMPLES[benchmark]
    name = f"{benchmark}__t{LEAF_TEMPERATURE}__max{mx}__n{ns}"
    return os.path.join(RESULTS_ROOT, model, name)
