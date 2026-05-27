"""Discover and parse evaluation summary files under an ``OUTPUT_ROOT``.

The orchestrator scripts write artefacts into a two-tier layout:

    ${OUTPUT_ROOT}/
        {target}_state-{state}_t{T}_max{N}/{benchmark}/{benchmark}_summary.json
        judgments/
            {target}_state-{state}_judged_by_{judge}_state-{state}_t{T}_max{N}/
                {benchmark}/{benchmark}_cot_summary.json
                {benchmark}/{benchmark}_cot_stats.json

This module walks that tree and emits one :class:`RunRecord` per summary file.
A legacy fallback regex also accepts the older ``{model}_t{T}_max{N}`` layout
that pre-dates the ``state-`` annotation.
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

# Canonical base run directory: "<model>_state-<state>_t<T>_max<N>".
_BASE_DIR_RE = re.compile(
    r"^(?P<model>.+?)_state-(?P<state>base|non-think|think|unknown)"
    r"_t(?P<temp>[0-9.]+)_max(?P<max>\d+)$"
)

# Legacy fallback: directories that pre-date the "_state-" annotation.
_LEGACY_BASE_DIR_RE = re.compile(r"^(?P<model>.+?)_t(?P<temp>[0-9.]+)_max(?P<max>\d+)$")

# Judgment directory:
# "<target>_state-<state>_judged_by_<judge>_state-<state>_t<T>_max<N>".
_JUDGE_DIR_RE = re.compile(
    r"^(?P<target>.+?)_state-(?P<target_state>base|non-think|think|unknown)"
    r"_judged_by_(?P<judge>.+?)_state-(?P<judge_state>base|non-think|think|unknown)"
    r"_t(?P<temp>[0-9.]+)_max(?P<max>\d+)$"
)

# Legacy judgment dir produced by compose_judge_dir in pipeline_common.sh:
# "<target>_evaluated_by_<judge>_<max>", with the benchmark subdir holding
# the temperature as "<benchmark>_t<T>". Both fields are extracted in
# _build_cot_record so the scanner picks up CoT runs from these directories.
_LEGACY_JUDGE_DIR_RE = re.compile(
    r"^(?P<target>.+?)_evaluated_by_(?P<judge>.+?)_(?P<max>\d+)$"
)
_LEGACY_BENCHMARK_TEMP_RE = re.compile(
    r"^(?P<benchmark>.+?)_t(?P<temp>[0-9.]+)$"
)


@dataclass(frozen=True)
class ParsedBaseDir:
    """Structured view of a base-run directory name."""

    model: str
    state: str
    temperature: float
    max_completion_tokens: int


@dataclass(frozen=True)
class ParsedJudgeDir:
    """Structured view of a judgment directory name."""

    target_model: str
    target_state: str
    judge_model: str
    judge_state: str
    temperature: float
    max_completion_tokens: int


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


def _read_json(path: Path) -> dict:
    with path.open("rb") as f:
        return orjson.loads(f.read())


def parse_base_dirname(name: str) -> ParsedBaseDir | None:
    """Parse a base run directory name. Returns ``None`` if it doesn't match."""
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


def parse_judge_dirname(name: str) -> ParsedJudgeDir | None:
    """Parse a judgment directory name. Returns ``None`` if it doesn't match."""
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


def _build_base_record(
    summary_path: Path, source_root: Path
) -> RunRecord | None:
    """Build a :class:`RunRecord` for a ``{benchmark}_summary.json`` file."""
    benchmark_dir = summary_path.parent
    run_dir = benchmark_dir.parent
    benchmark = benchmark_dir.name
    parsed = parse_base_dirname(run_dir.name)
    if parsed is None:
        logger.debug(f"Skipping unrecognised base run dir: {run_dir}")
        return None
    summary = _read_json(summary_path)
    return RunRecord(
        run_dir=run_dir.resolve(),
        summary_path=summary_path.resolve(),
        stats_path=None,
        eval_type="base_eval",
        model=parsed.model,
        state=_validate_state(parsed.state),
        temperature=parsed.temperature,
        max_completion_tokens=parsed.max_completion_tokens,
        benchmark=benchmark,
        judge_model=None,
        judge_state=None,
        judge_temperature=None,
        judge_max_completion_tokens=None,
        pass_at_k=_summary_to_pass_at_k(summary),
        cons_at_k=float(summary.get("cons_at_k", 0.0) or 0.0),
        stats=None,
        source_root=source_root.resolve(),
        note=summary.get("note"),
    )


def _build_cot_record(
    summary_path: Path, source_root: Path
) -> RunRecord | None:
    """Build a :class:`RunRecord` for a ``{benchmark}_cot_summary.json`` file."""
    benchmark_dir = summary_path.parent
    run_dir = benchmark_dir.parent
    parsed = parse_judge_dirname(run_dir.name)
    if parsed is None:
        logger.debug(f"Skipping unrecognised judgment dir: {run_dir}")
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
    stats = _read_json(stats_path) if stats_path.exists() else None
    return RunRecord(
        run_dir=run_dir.resolve(),
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
    )


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
        if summary_path.name.endswith("_cot_summary.json"):
            record = _build_cot_record(summary_path, root)
        else:
            record = _build_base_record(summary_path, root)
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
            }
        )
    return out
