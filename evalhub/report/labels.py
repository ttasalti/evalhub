"""Semantic labelling helpers for report aggregation (no plotting deps).

Short model names, family/size parsing, language and mode mappings — used by the
aggregator to enrich the master CSV. Kept matplotlib-free so the data pipeline
never imports a plotting stack.
"""

from __future__ import annotations

import re

# Benchmark -> language code. aime2026 trilingual + the standalone TR olympiad.
LANG: dict[str, str] = {
    "aime2026": "EN",
    "aime2026_pt": "PT",
    "aime2026_tr": "TR",
    "tubitak_math2026": "TR-OL",
}

# state -> human mode label.
MODE: dict[str, str] = {
    "base": "Pretrained",
    "non-think": "Instruct · Non-Think",
    "think": "Reasoning · Think",
}

_SIZE_RE = re.compile(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])")
# gemma 3n "effective-param" naming: E2B ≈ 2B, E4B ≈ 4B (a letter precedes the
# digit, so _SIZE_RE skips it). Used as a fallback so these models still get a
# numeric size for scaling/mode plots instead of NaN.
_ESIZE_RE = re.compile(r"[Ee](\d+(?:\.\d+)?)B\b")
_FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Qwen", "Qwen"),
    ("gemma", "gemma"),
    ("Ministral", "Ministral"),
    ("Llama", "Llama"),
    ("Mistral", "Mistral"),
)
_BENCH_FAMILY_SUFFIX_RE = re.compile(r"_(?:tr|pt|es|en|fr|de|zh|ja|ko)$")


def language(benchmark: str | None) -> str:
    if not benchmark:
        return "unknown"
    return LANG.get(benchmark, benchmark)


def mode_label(state: str | None) -> str:
    if not state:
        return "unknown"
    return MODE.get(state, state)


def model_family(model: str | None) -> str:
    if not model:
        return "unknown"
    for needle, label in _FAMILY_PATTERNS:
        if needle.lower() in model.lower():
            return label
    return "other"


def model_size_b(model: str | None) -> float | None:
    if not model:
        return None
    m = _SIZE_RE.search(model)
    if m is None:
        m = _ESIZE_RE.search(model)  # gemma E2B/E4B effective-param fallback
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def benchmark_family(benchmark: str | None) -> str:
    if not benchmark:
        return "unknown"
    return _BENCH_FAMILY_SUFFIX_RE.sub("", benchmark)


def is_base_model(name: str | None) -> bool:
    """Pretrained (non-instruct) checkpoint?"""
    if not name:
        return False
    if name.endswith("-Base") or "-Base-" in name:
        return True
    # gemma-4-* without the "-it" instruct suffix is the pretrained variant.
    if name.startswith("gemma-4-") and not name.endswith("-it"):
        return True
    return False


def short_model(name: str | None) -> str:
    """Qwen3.5-9B-Base -> 'Q-9B·Base', gemma-4-E4B-it -> 'G4-E4B'.

    Model names with an ``@stepN`` suffix (intermediate RL checkpoints) get the
    step appended as ``·sN``, e.g. ``DAPO-EN-Q-2B-t16g48·Base·s120``.
    """
    if not name:
        return "?"
    # Strip @stepN suffix before processing; re-attach as ·sN at the end.
    step_suffix = ""
    step_m = re.match(r"^(.+?)@step(\d+)$", name)
    if step_m:
        name = step_m.group(1)
        step_suffix = f"·s{step_m.group(2)}"
    base = is_base_model(name)
    s = name
    s = s.replace("Qwen3.5-", "Q-").replace("gemma-4-", "G4-").replace("Ministral-3-", "M-")
    s = re.sub(r"-A\d+B\b", "", s)  # 26B-A4B -> 26B
    s = s.replace("-2512", "").replace("-it", "").replace("-Base", "")
    result = f"{s}·Base" if base else s
    return f"{result}{step_suffix}"


def short_judge(jm: str | None, js: str | None) -> str:
    """Compact judge tag: '<short model>·<state short>'."""
    if not jm:
        return ""
    tag = {"base": "BS", "non-think": "NT", "think": "TH"}.get(js or "", js or "?")
    return f"{short_model(jm)}·{tag}"


def series_label(jm: str | None, js: str | None) -> str:
    """One-token grouping key: 'No-Judge' or 'cot:<judge>·<state>'."""
    return f"cot:{short_judge(jm, js)}" if jm else "No-Judge"
