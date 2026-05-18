"""Extract base-model generations marked correct, ready for judge evaluation."""

from __future__ import annotations

from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Any

import orjson

from evalhub.cot.ids import encode as encode_generation_id
from evalhub.utils.logger import logger


def _iter_jsonl(path: Path):
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def extract_correct_generations(
    base_results_path: PathLike,
    base_raw_path: PathLike,
    output_path: PathLike,
    max_tasks: int | None = None,
) -> int:
    """Filter the base raw generations down to only those marked correct.

    The base-evaluation step writes two files in lockstep:
      * ``*_results.jsonl`` — one record per task with a ``correct`` array
        aligned to ``solutions``.
      * ``*_raw.jsonl`` — one record per generation, in the same order, where
        the n-th occurrence of a given ``task_id`` corresponds to the n-th
        entry of that task's ``correct`` array.

    Returns the number of records written.
    """
    base_results_path = Path(base_results_path)
    base_raw_path = Path(base_raw_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not base_results_path.exists():
        raise FileNotFoundError(f"Base results file missing: {base_results_path}")
    if not base_raw_path.exists():
        raise FileNotFoundError(f"Base raw file missing: {base_raw_path}")

    correct_indices: dict[str, set[int]] = defaultdict(set)
    ground_truths: dict[str, Any] = {}
    correct_solutions: dict[str, dict[int, Any]] = defaultdict(dict)

    tasks_seen = 0
    for record in _iter_jsonl(base_results_path):
        task_id = record.get("task_id")
        if not task_id:
            continue
        if max_tasks is not None and tasks_seen >= max_tasks:
            break
        tasks_seen += 1

        ground_truths[task_id] = record.get("ground_truth", "")
        solutions = record.get("solutions", []) or []
        for idx, is_correct in enumerate(record.get("correct", []) or []):
            if is_correct is True:
                correct_indices[task_id].add(idx)
                if idx < len(solutions):
                    correct_solutions[task_id][idx] = solutions[idx]

    counters: dict[str, int] = defaultdict(int)
    written = 0
    with output_path.open("wb") as f_out:
        for record in _iter_jsonl(base_raw_path):
            task_id = record.get("task_id")
            if not task_id or task_id not in correct_indices:
                if task_id:
                    counters[task_id] += 1
                continue

            current_idx = counters[task_id]
            counters[task_id] += 1
            if current_idx not in correct_indices[task_id]:
                continue

            judge_input = {
                "task_id": encode_generation_id(task_id, current_idx),
                "original_task_id": task_id,
                "generation_idx": current_idx,
                "ground_truth": ground_truths.get(task_id, ""),
                "generated_answer": correct_solutions[task_id].get(current_idx, ""),
                "raw_response": record.get("response", {}),
            }
            f_out.write(orjson.dumps(judge_input) + b"\n")
            written += 1

    logger.info(f"Extracted {written} correct generations to {output_path}")
    return written
