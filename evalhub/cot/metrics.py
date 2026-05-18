"""Apply CoT judge verdicts to base results and recompute Pass@K / Cons@K."""

from __future__ import annotations

from collections import Counter, defaultdict
from os import PathLike
from pathlib import Path
from typing import Any

import orjson

from evalhub.cot.ids import encode as encode_generation_id
from evalhub.utils.logger import logger
from evalhub.utils.metrics import compute_pass_at_k

DEFAULT_KS: list[int] = [2**i for i in range(11)]
COT_FALSE_LABEL = "cot_false"


def _load_majority_map(majority_path: Path) -> dict[str, bool]:
    mapping: dict[str, bool] = {}
    with majority_path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = orjson.loads(line)
            mapping[record["task_id"]] = bool(record["majority_correct"])
    return mapping


def _ks_from_record(record: dict[str, Any], n_generations: int) -> list[int]:
    declared = record.get("pass_at_k") or {}
    if declared:
        return sorted({int(k) for k in declared.keys() if int(k) <= n_generations})
    return [k for k in DEFAULT_KS if k <= n_generations]


def apply_cot_metrics(
    base_results_path: PathLike,
    majority_path: PathLike,
    output_results_path: PathLike,
    summary_path: PathLike,
    stats_path: PathLike | None = None,
) -> dict[str, Any]:
    """Re-evaluate base results under the CoT veto.

    Each generation that the base evaluator marked correct is downgraded to
    ``"cot_false"`` if the judge's majority verdict is negative. Pass@K is then
    recomputed against the surviving true count for every K up to ``n_samples``,
    and Cons@K (consensus correctness on the majority answer) is recomputed on
    the post-veto ``correct`` array.
    """
    base_results_path = Path(base_results_path)
    majority_path = Path(majority_path)
    output_results_path = Path(output_results_path)
    summary_path = Path(summary_path)
    for p in (output_results_path, summary_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    if not base_results_path.exists():
        raise FileNotFoundError(f"Base results file missing: {base_results_path}")
    if not majority_path.exists():
        raise FileNotFoundError(f"Majority file missing: {majority_path}")

    majority_map = _load_majority_map(majority_path)

    sum_pass_at_k: dict[str, float] = defaultdict(float)
    sum_cons_at_k = 0.0
    total_tasks = 0

    stats = {
        "total_tasks": 0,
        "total_generations": 0,
        "true_count": 0,
        "false_count": 0,
        "cot_false_count": 0,
        "invalid_count": 0,
    }

    with base_results_path.open("rb") as f_in, output_results_path.open("wb") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = orjson.loads(line)
            task_id = record["task_id"]
            correct_arr: list[Any] = list(record.get("correct", []) or [])
            solutions: list[Any] = list(record.get("solutions", []) or [])
            n_generations = len(correct_arr)

            for i in range(n_generations):
                gen_id = encode_generation_id(task_id, i)
                if correct_arr[i] is True and gen_id in majority_map and not majority_map[gen_id]:
                    correct_arr[i] = COT_FALSE_LABEL

            true_count = sum(1 for x in correct_arr if x is True)
            false_count = sum(1 for x in correct_arr if x is False)
            cot_false_count = sum(1 for x in correct_arr if x == COT_FALSE_LABEL)
            invalid_count = n_generations - true_count - false_count - cot_false_count

            stats["total_tasks"] += 1
            stats["total_generations"] += n_generations
            stats["true_count"] += true_count
            stats["false_count"] += false_count
            stats["cot_false_count"] += cot_false_count
            stats["invalid_count"] += invalid_count

            new_pass_at_k: dict[str, float] = {}
            for k in _ks_from_record(record, n_generations):
                value = compute_pass_at_k(n_generations, true_count, k)
                new_pass_at_k[str(k)] = value
                sum_pass_at_k[str(k)] += value

            if solutions:
                sol_strs = ["" if s is None else str(s) for s in solutions]
                majority_answer, _ = Counter(sol_strs).most_common(1)[0]
                is_consensus_correct = any(
                    correct_arr[i] is True for i, sol in enumerate(sol_strs) if sol == majority_answer
                )
                if is_consensus_correct:
                    sum_cons_at_k += 1.0

            total_tasks += 1
            record["correct"] = correct_arr
            record["pass_at_k"] = new_pass_at_k
            f_out.write(orjson.dumps(record) + b"\n")

    if total_tasks == 0:
        raise ValueError(f"Base results file produced no records: {base_results_path}")

    pass_at_k_summary = {k: v / total_tasks for k, v in sum_pass_at_k.items()}
    cons_at_k = sum_cons_at_k / total_tasks
    summary = {"pass_at_k": pass_at_k_summary, "cons_at_k": cons_at_k}

    with summary_path.open("wb") as f_sum:
        f_sum.write(orjson.dumps(summary))

    if stats_path is not None:
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("wb") as f_stats:
            f_stats.write(orjson.dumps(stats))

    for k, value in pass_at_k_summary.items():
        logger.info(f"CoT-Pass@{k}: {value:.4f}")
    logger.info(f"CoT-Cons@K:   {cons_at_k:.4f} over {total_tasks} tasks")

    return summary
