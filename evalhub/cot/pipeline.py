"""Composite ``finalize`` step: extract -> aggregate -> metrics on local files."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from evalhub.cot.aggregate import aggregate_judge_votes
from evalhub.cot.extract import extract_correct_generations
from evalhub.cot.metrics import apply_cot_metrics


@dataclass
class FinalizeResult:
    extracted_count: int
    aggregated_count: int
    summary: dict[str, Any]
    output_results: Path
    summary_path: Path
    majority_path: Path
    judge_input_path: Path


def finalize_cot_pipeline(
    base_results_path: PathLike,
    base_raw_path: PathLike,
    judge_solutions_path: PathLike,
    output_dir: PathLike,
    benchmark: str,
    stats_path: PathLike | None = None,
) -> FinalizeResult:
    """Run extract, aggregate, and metric application as one local step.

    This expects the upstream generations (base + judge) to already exist on
    disk; it does no model inference of its own.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_input_path = output_dir / f"{benchmark}_cot_judge_input.jsonl"
    majority_path = output_dir / f"{benchmark}_cot_majority.jsonl"
    output_results_path = output_dir / f"{benchmark}_cot_results.jsonl"
    summary_path = output_dir / f"{benchmark}_cot_summary.json"
    resolved_stats_path = (
        Path(stats_path) if stats_path is not None else output_dir / f"{benchmark}_cot_stats.json"
    )

    extracted = extract_correct_generations(
        base_results_path=base_results_path,
        base_raw_path=base_raw_path,
        output_path=judge_input_path,
    )
    aggregated = aggregate_judge_votes(
        judge_solutions_path=judge_solutions_path,
        output_path=majority_path,
    )
    summary = apply_cot_metrics(
        base_results_path=base_results_path,
        majority_path=majority_path,
        output_results_path=output_results_path,
        summary_path=summary_path,
        stats_path=resolved_stats_path,
    )

    return FinalizeResult(
        extracted_count=extracted,
        aggregated_count=aggregated,
        summary=summary,
        output_results=output_results_path,
        summary_path=summary_path,
        majority_path=majority_path,
        judge_input_path=judge_input_path,
    )
