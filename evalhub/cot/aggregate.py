"""Majority-vote aggregation of per-generation judge verdicts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import orjson

from evalhub.utils.logger import logger

YES_TOKEN = "yes"
NO_TOKEN = "no"


@dataclass
class GenerationVerdict:
    generation_id: str
    yes_count: int
    no_count: int
    invalid_count: int
    majority_correct: bool

    def to_record(self) -> dict:
        return {
            "task_id": self.generation_id,
            "yes_count": self.yes_count,
            "no_count": self.no_count,
            "invalid_count": self.invalid_count,
            "majority_correct": self.majority_correct,
        }


def _classify(token: str) -> str:
    norm = (token or "").strip().lower()
    if norm == YES_TOKEN:
        return YES_TOKEN
    if norm == NO_TOKEN:
        return NO_TOKEN
    return "invalid"


def aggregate_judge_votes(
    judge_solutions_path: PathLike,
    output_path: PathLike,
) -> int:
    """Group judge outputs by generation id and apply strict majority voting.

    A generation is approved if and only if the count of ``"yes"`` verdicts is
    strictly greater than the count of ``"no"`` verdicts (invalid extractions
    abstain rather than implicitly voting "no").
    """
    judge_solutions_path = Path(judge_solutions_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not judge_solutions_path.exists():
        raise FileNotFoundError(f"Judge solutions file missing: {judge_solutions_path}")

    buckets: dict[str, list[str]] = defaultdict(list)
    with judge_solutions_path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = orjson.loads(line)
            task_id = record.get("task_id")
            if not task_id:
                continue
            solution = record.get("solution", "")
            if isinstance(solution, list):
                buckets[task_id].extend(_classify(str(s)) for s in solution)
            else:
                buckets[task_id].append(_classify(str(solution)))

    written = 0
    with output_path.open("wb") as f_out:
        for generation_id, verdicts in buckets.items():
            yes_count = sum(1 for v in verdicts if v == YES_TOKEN)
            no_count = sum(1 for v in verdicts if v == NO_TOKEN)
            invalid_count = sum(1 for v in verdicts if v == "invalid")
            verdict = GenerationVerdict(
                generation_id=generation_id,
                yes_count=yes_count,
                no_count=no_count,
                invalid_count=invalid_count,
                majority_correct=yes_count > no_count,
            )
            f_out.write(orjson.dumps(verdict.to_record()) + b"\n")
            written += 1

    logger.info(f"Aggregated {written} generation verdicts to {output_path}")
    return written
