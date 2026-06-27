"""Discover and parse evaluation summary files under an ``OUTPUT_ROOT``.

The orchestrator scripts (``scripts/lib/pipeline_common.sh``) write the current
**V5** layout: ONE folder per model. The model dir carries only ``<state>/<model>``
and the full sampling suffix (``__t{T}__max{N}__n{NS}``) is moved onto the benchmark
leaf, so every sampling variant of a model sits side-by-side under a single model dir:

    ${OUTPUT_ROOT}/
        <state>/<model>/
            <benchmark>__t{T}__max{N}__n{NS}/<benchmark>_summary.json
            judged_by/<judge>__state-<jstate>__t{jT}__max{jN}/
                <benchmark>__t{T}__max{N}__n{NS}/<benchmark>_cot_summary.json
                <benchmark>__t{T}__max{N}__n{NS}/<benchmark>_cot_stats.json

(The benchmark leaf carries the TARGET's sampling on both the base side and under
``judged_by/``. Files inside a leaf keep the bare benchmark name, e.g.
``aime2026_summary.json``, since they are named from ``--tasks``/``--benchmark``.)

This module walks that tree and emits one :class:`RunRecord` per summary file.
Fallback regexes also accept the older V3 (sampling suffix on the model dir leaf),
V2 (``__state-..__`` in the leaf), V1 (single-underscore ``_state-..``) and V0
(no ``state-`` annotation, flat ``judgments/`` / ``_evaluated_by_``) layouts so
historical runs still parse.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import orjson

from evalhub.utils.logger import logger
from evalhub.utils.model_state import MODEL_STATES

EvalType = Literal["base_eval", "cot_eval"]

# V5 (current) benchmark leaf — ONE folder per model: the model dir carries only
# "<state>/<model>", and the sampling suffix moved onto the benchmark leaf:
# "<benchmark>__t<T>__max<N>__n<NS>". The model dir is the leaf's parent (or its
# grandparent across a "step_NNN"/"judged_by" level); the state is one level above
# the model dir. Non-greedy "<benchmark>.+?" + the "__t" double-underscore boundary
# splits cleanly even for single-underscore benchmark names (aime2026_tr, tubitak_math2026).
_BENCH_LEAF_RE_V5 = re.compile(
    r"^(?P<benchmark>.+?)__t(?P<temp>[0-9.]+)__max(?P<max>\d+)__n(?P<n_samples>\d+)$"
)

# V3 base run directory leaf — state hoisted to a parent dir, leaf carries
# model + sampling knobs: "<model>__t<T>__max<N>__n<NS>".
# The state lives in the parent dir name ({base, non-think, think, unknown}).
_BASE_DIR_RE_V3 = re.compile(
    r"^(?P<model>.+?)__t(?P<temp>[0-9.]+)__max(?P<max>\d+)__n(?P<n_samples>\d+)$"
)

# Recognised state dir names sitting one level above a V3 base leaf.
_V3_STATE_DIRS: tuple[str, ...] = ("base", "non-think", "think", "unknown")

# V2 base run directory (state embedded in leaf via "__state-<s>__").
_BASE_DIR_RE_V2 = re.compile(
    r"^(?P<model>.+?)__state-(?P<state>base|non-think|think|unknown)"
    r"__t(?P<temp>[0-9.]+)__max(?P<max>\d+)__n(?P<n_samples>\d+)$"
)

# V1 base run directory (single-underscore separator, no n_samples).
_BASE_DIR_RE = re.compile(
    r"^(?P<model>.+?)_state-(?P<state>base|non-think|think|unknown)"
    r"_t(?P<temp>[0-9.]+)_max(?P<max>\d+)$"
)

# V0 legacy: directories that pre-date the "_state-" annotation.
_LEGACY_BASE_DIR_RE = re.compile(r"^(?P<model>.+?)_t(?P<temp>[0-9.]+)_max(?P<max>\d+)$")

# V4 (current) judgment leaf directory — the leaf name under judged_by/:
# "<judge>__state-<jstate>__t<jT>__max<jN>" (no "__n<jNS>" trailer).
# JUDGE_N_SAMPLES is consolidated out of the path; the actual value lives in
# each benchmark's summary file. The regex still accepts the V2 trailing
# "__n<jNS>" optionally so older on-disk runs can be parsed transparently.
_JUDGE_DIR_RE_V4 = re.compile(
    r"^(?P<judge>.+?)__state-(?P<judge_state>base|non-think|think|unknown)"
    r"__t(?P<temp>[0-9.]+)__max(?P<max>\d+)(?:__n(?P<n_samples>\d+))?$"
)
# V2 legacy alias — same trailer form as V4 for backwards-compat references.
_JUDGE_DIR_RE_V2 = _JUDGE_DIR_RE_V4

# V1 judgment directory (flat under judgments/):
# "<target>_state-<state>_judged_by_<judge>_state-<state>_t<T>_max<N>".
_JUDGE_DIR_RE = re.compile(
    r"^(?P<target>.+?)_state-(?P<target_state>base|non-think|think|unknown)"
    r"_judged_by_(?P<judge>.+?)_state-(?P<judge_state>base|non-think|think|unknown)"
    r"_t(?P<temp>[0-9.]+)_max(?P<max>\d+)$"
)

# V0 legacy judgment dir produced by the original compose_judge_dir:
# "<target>_evaluated_by_<judge>_<max>", with the benchmark subdir holding
# the temperature as "<benchmark>_t<T>". Both fields are extracted in
# _build_cot_record so the scanner picks up CoT runs from these directories.
_LEGACY_JUDGE_DIR_RE = re.compile(
    r"^(?P<target>.+?)_evaluated_by_(?P<judge>.+?)_(?P<max>\d+)$"
)
_LEGACY_BENCHMARK_TEMP_RE = re.compile(
    r"^(?P<benchmark>.+?)_t(?P<temp>[0-9.]+)$"
)

# Intermediate RL checkpoint directory: "step_<N>" sitting inside a model run dir.
_STEP_DIR_RE = re.compile(r"^step_(\d+)$")


@dataclass(frozen=True)
class ParsedBaseDir:
    """Structured view of a base-run directory name."""

    model: str
    state: str
    temperature: float
    max_completion_tokens: int
    n_samples: int | None = None


@dataclass(frozen=True)
class ParsedJudgeDir:
    """Structured view of a judgment directory name."""

    target_model: str
    target_state: str
    judge_model: str
    judge_state: str
    temperature: float
    max_completion_tokens: int
    target_n_samples: int | None = None
    judge_n_samples: int | None = None


@dataclass(frozen=True)
class RunRecord:
    """A single parsed evaluation summary.

    One :class:`RunRecord` corresponds to one ``*_summary.json`` (base eval) or
    ``*_cot_summary.json`` (CoT-vetted eval) file. All path fields are
    absolute; ``stats`` is populated only for ``cot_eval`` records that have a
    sibling ``*_cot_stats.json`` written by ``evalhub cot metrics``.
    """

    run_dir: Path
    summary_path: Path
    stats_path: Path | None
    eval_type: EvalType
    model: str
    state: str
    temperature: float
    max_completion_tokens: int
    benchmark: str
    judge_model: str | None
    judge_state: str | None
    judge_temperature: float | None
    judge_max_completion_tokens: int | None
    pass_at_k: dict[int, float]
    cons_at_k: float
    stats: dict[str, int] | None
    source_root: Path
    note: str | None = field(default=None)
    n_samples: int | None = field(default=None)
    judge_n_samples: int | None = field(default=None)
    # G-Pass@k blocks, str-keyed exactly as written to the summary JSON:
    #   g_pass_at_k = {"<k>": {"<tau>": value}}   mg_pass_at_k = {"<k>": value}
    g_pass_at_k: dict | None = field(default=None)
    mg_pass_at_k: dict | None = field(default=None)
    # RL training step this checkpoint was evaluated at (None = pretrained or final).
    rl_step: int | None = field(default=None)


def _read_json(path: Path) -> dict:
    with path.open("rb") as f:
        return orjson.loads(f.read())


def parse_benchmark_leaf(name: str) -> tuple[str, float, int, int] | None:
    """Parse a V5 benchmark leaf "<benchmark>__t<T>__max<N>__n<NS>".

    Returns ``(benchmark, temperature, max_completion_tokens, n_samples)`` or
    ``None`` when ``name`` is a bare benchmark (old layouts) and carries no
    sampling suffix — in which case the caller falls back to the V3/V2/.. path.
    """
    match = _BENCH_LEAF_RE_V5.match(name)
    if match is None:
        return None
    return (
        match.group("benchmark"),
        float(match.group("temp")),
        int(match.group("max")),
        int(match.group("n_samples")),
    )


def parse_base_dirname(name: str, parent_state: str | None = None) -> ParsedBaseDir | None:
    """Parse a base run directory name. Returns ``None`` if it doesn't match.

    Tries V2 (state in leaf) first because its prefix overlaps V3's. Then V3
    (state-less leaf, state from ``parent_state``), V1 (single-underscore),
    and finally V0 (no state — returns "unknown").
    """
    match_v2 = _BASE_DIR_RE_V2.match(name)
    if match_v2 is not None:
        return ParsedBaseDir(
            model=match_v2.group("model"),
            state=match_v2.group("state"),
            temperature=float(match_v2.group("temp")),
            max_completion_tokens=int(match_v2.group("max")),
            n_samples=int(match_v2.group("n_samples")),
        )
    match_v3 = _BASE_DIR_RE_V3.match(name)
    if match_v3 is not None:
        # V3 leaf — caller supplies the state from the parent dir. Fall back to
        # "unknown" so a V3 leaf encountered outside a known state parent still
        # yields a valid record.
        state = parent_state if parent_state in _V3_STATE_DIRS else "unknown"
        return ParsedBaseDir(
            model=match_v3.group("model"),
            state=state,
            temperature=float(match_v3.group("temp")),
            max_completion_tokens=int(match_v3.group("max")),
            n_samples=int(match_v3.group("n_samples")),
        )
    match = _BASE_DIR_RE.match(name)
    if match is not None:
        return ParsedBaseDir(
            model=match.group("model"),
            state=match.group("state"),
            temperature=float(match.group("temp")),
            max_completion_tokens=int(match.group("max")),
        )
    legacy = _LEGACY_BASE_DIR_RE.match(name)
    if legacy is not None:
        return ParsedBaseDir(
            model=legacy.group("model"),
            state="unknown",
            temperature=float(legacy.group("temp")),
            max_completion_tokens=int(legacy.group("max")),
        )
    return None


def parse_judge_leaf_dirname(name: str) -> ParsedBaseDir | None:
    """Parse the V4 judge leaf directory (``judge_clean__state-..__t..__max..``).

    The V2 variant with a trailing ``__n<jNS>`` is still accepted for
    backwards-compat with on-disk runs that pre-date the V4 layout. Returns
    None when ``name`` doesn't match either form.
    """
    match = _JUDGE_DIR_RE_V4.match(name)
    if match is None:
        return None
    n_samples_raw = match.group("n_samples")
    return ParsedBaseDir(
        model=match.group("judge"),
        state=match.group("judge_state"),
        temperature=float(match.group("temp")),
        max_completion_tokens=int(match.group("max")),
        n_samples=int(n_samples_raw) if n_samples_raw else None,
    )


def parse_judge_dirname(name: str) -> ParsedJudgeDir | None:
    """Parse a flat (V1/V0) judgment directory name. Returns ``None`` if it
    doesn't match. V2 judgment dirs are nested (target_dir/judged_by/judge_dir)
    and are handled directly by :func:`_build_cot_record`, not here.
    """
    match = _JUDGE_DIR_RE.match(name)
    if match is not None:
        return ParsedJudgeDir(
            target_model=match.group("target"),
            target_state=match.group("target_state"),
            judge_model=match.group("judge"),
            judge_state=match.group("judge_state"),
            temperature=float(match.group("temp")),
            max_completion_tokens=int(match.group("max")),
        )
    legacy = _LEGACY_JUDGE_DIR_RE.match(name)
    if legacy is not None:
        # Temperature is encoded in the benchmark subdir, not here; the real
        # value is filled in by _build_cot_record from _LEGACY_BENCHMARK_TEMP_RE.
        return ParsedJudgeDir(
            target_model=legacy.group("target"),
            target_state="unknown",
            judge_model=legacy.group("judge"),
            judge_state="unknown",
            temperature=0.0,
            max_completion_tokens=int(legacy.group("max")),
        )
    return None


def _summary_to_pass_at_k(summary: dict) -> dict[int, float]:
    raw = summary.get("pass_at_k") or {}
    out: dict[int, float] = {}
    for k, value in raw.items():
        try:
            out[int(k)] = float(value)
        except (TypeError, ValueError):
            logger.warning(f"Skipping non-numeric pass_at_k entry {k!r}={value!r}")
    return out


def _validate_state(state: str) -> str:
    if state in MODEL_STATES or state == "unknown":
        return state
    return "unknown"


def _stats_from_summary(summary: dict) -> dict[str, int] | None:
    """Pull aggregate count fields out of a summary.json (V2+ writes them).

    Normalises ``invalid_format_count`` (summary key) to ``invalid_count``
    (stats.json + aggregate column) for downstream consistency.
    """
    keys = ("total_tasks", "total_generations", "true_count", "false_count",
            "cot_false_count")
    if not any(k in summary for k in keys + ("invalid_format_count", "invalid_count")):
        return None
    out: dict[str, int] = {k: int(summary[k]) for k in keys if k in summary}
    if "invalid_format_count" in summary:
        out["invalid_count"] = int(summary["invalid_format_count"])
    elif "invalid_count" in summary:
        out["invalid_count"] = int(summary["invalid_count"])
    return out


def _build_base_record(
    summary_path: Path, source_root: Path
) -> RunRecord | None:
    """Build a :class:`RunRecord` for a ``{benchmark}_summary.json`` file."""
    benchmark_dir = summary_path.parent
    run_dir = benchmark_dir.parent

    # Detect "step_NNN" sub-directory: an intermediate RL checkpoint sitting
    # inside the model run dir.  Navigate up to the real model dir and record
    # the step number; the model name gets an "@stepN" suffix to make each
    # checkpoint a distinct row in the aggregate CSV.
    rl_step: int | None = None
    step_m = _STEP_DIR_RE.match(run_dir.name)
    if step_m:
        rl_step = int(step_m.group(1))
        run_dir = run_dir.parent  # up to the actual model run directory

    # V5 (current): sampling suffix lives on the benchmark leaf, model dir is bare.
    leaf = parse_benchmark_leaf(benchmark_dir.name)
    if leaf is not None:
        benchmark, temperature, max_completion_tokens, n_samples = leaf
        state = run_dir.parent.name if run_dir.parent.name in _V3_STATE_DIRS else "unknown"
        model = run_dir.name
    else:
        # V3/V2/V1/V0 fallback: sampling suffix lives on the model run dir leaf.
        benchmark = benchmark_dir.name
        parent_state = run_dir.parent.name if run_dir.parent.name in _V3_STATE_DIRS else None
        parsed = parse_base_dirname(run_dir.name, parent_state=parent_state)
        if parsed is None:
            logger.debug(f"Skipping unrecognised base run dir: {run_dir}")
            return None
        model = parsed.model
        state = parsed.state
        temperature = parsed.temperature
        max_completion_tokens = parsed.max_completion_tokens
        n_samples = parsed.n_samples

    model_name = model if rl_step is None else f"{model}@step{rl_step}"
    summary = _read_json(summary_path)
    return RunRecord(
        run_dir=benchmark_dir.parent.resolve(),  # points to step_NNN or model run dir
        summary_path=summary_path.resolve(),
        stats_path=None,
        eval_type="base_eval",
        model=model_name,
        state=_validate_state(state),
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        benchmark=benchmark,
        judge_model=None,
        judge_state=None,
        judge_temperature=None,
        judge_max_completion_tokens=None,
        pass_at_k=_summary_to_pass_at_k(summary),
        cons_at_k=float(summary.get("cons_at_k", 0.0) or 0.0),
        stats=_stats_from_summary(summary),
        source_root=source_root.resolve(),
        note=summary.get("note"),
        n_samples=n_samples,
        g_pass_at_k=summary.get("g_pass_at_k") or None,
        mg_pass_at_k=summary.get("mg_pass_at_k") or None,
        rl_step=rl_step,
    )


def _build_cot_record(
    summary_path: Path, source_root: Path
) -> RunRecord | None:
    """Build a :class:`RunRecord` for a ``{benchmark}_cot_summary.json`` file.

    Supports three layouts:

    * **V5 (nested, current)** — ``<state>/<model>/judged_by/<judge>__state-../<benchmark>__t..__max..__n../``
    * **V2/V3 (nested)** — ``<target>__t..__n.../judged_by/<judge>__state-..../<benchmark>/``
    * **V1/V0 (flat)** — ``<...>/<flat_judge_dir>/<benchmark>[_t<T>]/``
    """
    benchmark_dir = summary_path.parent
    parent_dir = benchmark_dir.parent

    # --- V5/V3/V2 nested layout detection ---------------------------------
    if parent_dir.parent.name == "judged_by":
        target_dir = parent_dir.parent.parent

        # Detect "step_NNN" sub-directory: the judged_by/ sits inside step_NNN/.
        rl_step: int | None = None
        step_m = _STEP_DIR_RE.match(target_dir.name)
        if step_m:
            rl_step = int(step_m.group(1))
            target_dir = target_dir.parent  # up to the actual model run directory

        judge_parsed = parse_judge_leaf_dirname(parent_dir.name)
        # V5: sampling suffix on the benchmark leaf, target dir is the bare model.
        leaf = parse_benchmark_leaf(benchmark_dir.name)
        target_parsed: ParsedBaseDir | None = None
        if leaf is not None and judge_parsed is not None:
            benchmark, temperature, max_completion_tokens, n_samples = leaf
            state = target_dir.parent.name if target_dir.parent.name in _V3_STATE_DIRS else "unknown"
            model = target_dir.name
        else:
            # V3/V2 fallback: sampling suffix on the target dir leaf, bare benchmark.
            target_parent_state = (
                target_dir.parent.name if target_dir.parent.name in _V3_STATE_DIRS else None
            )
            target_parsed = parse_base_dirname(target_dir.name, parent_state=target_parent_state)
            if target_parsed is not None:
                benchmark = benchmark_dir.name
                temperature = target_parsed.temperature
                max_completion_tokens = target_parsed.max_completion_tokens
                n_samples = target_parsed.n_samples
                state = target_parsed.state
                model = target_parsed.model

        if judge_parsed is not None and (leaf is not None or target_parsed is not None):
            model_name = model if rl_step is None else f"{model}@step{rl_step}"
            summary = _read_json(summary_path)
            stats_path = benchmark_dir / f"{benchmark}_cot_stats.json"
            stats = _read_json(stats_path) if stats_path.exists() else _stats_from_summary(summary)
            return RunRecord(
                run_dir=parent_dir.resolve(),
                summary_path=summary_path.resolve(),
                stats_path=stats_path.resolve() if stats_path.exists() else None,
                eval_type="cot_eval",
                model=model_name,
                state=_validate_state(state),
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                benchmark=benchmark,
                judge_model=judge_parsed.model,
                judge_state=_validate_state(judge_parsed.state),
                judge_temperature=judge_parsed.temperature,
                judge_max_completion_tokens=judge_parsed.max_completion_tokens,
                pass_at_k=_summary_to_pass_at_k(summary),
                cons_at_k=float(summary.get("cons_at_k", 0.0) or 0.0),
                stats=stats,
                source_root=source_root.resolve(),
                note=summary.get("note"),
                n_samples=n_samples,
                judge_n_samples=judge_parsed.n_samples,
                g_pass_at_k=summary.get("g_pass_at_k") or None,
                mg_pass_at_k=summary.get("mg_pass_at_k") or None,
                rl_step=rl_step,
            )

    # --- V1/V0 flat layout fallback ---------------------------------------
    parsed = parse_judge_dirname(parent_dir.name)
    if parsed is None:
        logger.debug(f"Skipping unrecognised judgment dir: {parent_dir}")
        return None

    # Legacy benchmark dir embeds the judge temperature: "<benchmark>_t<T>".
    bench_match = _LEGACY_BENCHMARK_TEMP_RE.match(benchmark_dir.name)
    if bench_match is not None:
        benchmark = bench_match.group("benchmark")
        temperature = float(bench_match.group("temp"))
    else:
        benchmark = benchmark_dir.name
        temperature = parsed.temperature

    summary = _read_json(summary_path)
    stats_path = benchmark_dir / f"{benchmark}_cot_stats.json"
    stats = _read_json(stats_path) if stats_path.exists() else _stats_from_summary(summary)
    return RunRecord(
        run_dir=parent_dir.resolve(),
        summary_path=summary_path.resolve(),
        stats_path=stats_path.resolve() if stats_path.exists() else None,
        eval_type="cot_eval",
        model=parsed.target_model,
        state=_validate_state(parsed.target_state),
        temperature=temperature,
        max_completion_tokens=parsed.max_completion_tokens,
        benchmark=benchmark,
        judge_model=parsed.judge_model,
        judge_state=_validate_state(parsed.judge_state),
        judge_temperature=temperature,
        judge_max_completion_tokens=parsed.max_completion_tokens,
        pass_at_k=_summary_to_pass_at_k(summary),
        cons_at_k=float(summary.get("cons_at_k", 0.0) or 0.0),
        stats=stats,
        source_root=source_root.resolve(),
        note=summary.get("note"),
        g_pass_at_k=summary.get("g_pass_at_k") or None,
        mg_pass_at_k=summary.get("mg_pass_at_k") or None,
    )


def record_from_summary(
    summary_path: Path | str, source_root: Path | str | None = None
) -> RunRecord | None:
    """Parse a single ``*_summary.json`` / ``*_cot_summary.json`` into a record.

    ``source_root`` only sets ``RunRecord.source_root``; pass the results root if
    you have it, otherwise the summary's own directory is used. Returns ``None``
    when the surrounding directory names don't match any known layout.
    """
    path = Path(summary_path)
    root = Path(source_root) if source_root is not None else path.parent
    if path.name.endswith("_cot_summary.json"):
        return _build_cot_record(path, root)
    return _build_base_record(path, root)


def scan_results(results_root: Path | str) -> list[RunRecord]:
    """Walk ``results_root`` and parse every summary file into a :class:`RunRecord`.

    Unrecognised directory names are skipped with a debug log entry — this keeps
    the scanner robust to hand-crafted output dirs sitting next to the
    canonical ones.
    """
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"results_root does not exist: {root}")

    records: list[RunRecord] = []
    seen: set[Path] = set()
    for summary_path in sorted(root.rglob("*_summary.json")):
        if summary_path in seen:
            continue
        seen.add(summary_path)
        record = record_from_summary(summary_path, root)
        if record is not None:
            records.append(record)
    logger.info(f"Parsed {len(records)} run record(s) under {root}")
    return records


def records_to_dicts(records: Iterable[RunRecord]) -> list[dict]:
    """Flatten records to plain dicts (useful for JSON dumps in tests)."""
    out: list[dict] = []
    for r in records:
        out.append(
            {
                "run_dir": str(r.run_dir),
                "summary_path": str(r.summary_path),
                "stats_path": str(r.stats_path) if r.stats_path else None,
                "eval_type": r.eval_type,
                "model": r.model,
                "state": r.state,
                "temperature": r.temperature,
                "max_completion_tokens": r.max_completion_tokens,
                "benchmark": r.benchmark,
                "judge_model": r.judge_model,
                "judge_state": r.judge_state,
                "judge_temperature": r.judge_temperature,
                "judge_max_completion_tokens": r.judge_max_completion_tokens,
                "pass_at_k": dict(r.pass_at_k),
                "cons_at_k": r.cons_at_k,
                "stats": dict(r.stats) if r.stats else None,
                "note": r.note,
                "n_samples": r.n_samples,
                "judge_n_samples": r.judge_n_samples,
            }
        )
    return out
