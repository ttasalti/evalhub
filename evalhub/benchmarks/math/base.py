import csv
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Any

import orjson

from evalhub.benchmarks.base import Dataset
from evalhub.benchmarks.math.verifier import extract_answer, grade_answer
from evalhub.utils.logger import logger
from evalhub.utils.metrics import compute_pass_at_k, get_majority_vote
from evalhub.utils.pbar import get_progress_bar

DEFAULT_KS = [2**i for i in range(11)]


class MathDataset(Dataset):
    r"""Dataset class for math reasoning problems."""

    def __init__(self, name: str = "math", **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self) -> None:
        r"""Load tasks from math reasoning dataset."""
        raise NotImplementedError

    def format_prompt(self, item: dict[str, Any]) -> str:
        r"""Format the prompt for math reasoning task."""
        raise NotImplementedError

    def extract_solution(self, task_id: str, response: str | None) -> str:
        r"""Extract the solution from the response."""
        return extract_answer(response) or ""

    def check_correct(self, extracted_answer: str, ground_truth: str, task_id: str = None) -> bool:
        r"""Check if the extracted answer is correct."""
        return grade_answer(extracted_answer, ground_truth) or self.patch(extracted_answer, ground_truth, task_id)

    def patch(self, extracted_answer: str, ground_truth: str, task_id: str) -> bool:
        r"""Patch the extracted answer."""
        return False

    def _load_solutions(self, solution_path: PathLike) -> dict[str, list[str]]:
        r"""Load predictions from solution file."""
        solutions = defaultdict(list)
        solution_path = Path(solution_path)

        with open(solution_path, "rb") as f:
            for line in f:
                sample = orjson.loads(line)
                task_id = sample["task_id"]
                solution = sample["solution"]
                solutions[task_id].append(solution)

        return solutions

    def evaluate(self, solution: PathLike, output_dir: PathLike) -> None:
        r"""Evaluate the solution."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        id2solutions = self._load_solutions(solution)
        assert len(id2solutions) == len(self.groundtruth), (
            f"Predictions ({len(id2solutions)}) must match groundtruths ({len(self.groundtruth)})"
        )

        results, correct, total = [], 0, len(id2solutions)
        progress = get_progress_bar()
        with progress:
            eval_task = progress.add_task("[bold blue]Evaluating", total=total)

            for task_id, solutions in id2solutions.items():
                ground_truth = self.groundtruth[task_id].answer
                is_correct = [self.check_correct(solution, ground_truth, task_id) for solution in solutions]

                # Calculate pass@k metrics
                pass_at_k = defaultdict(float)
                for k in DEFAULT_KS:
                    if k > len(solutions):
                        continue
                    pass_at_k[str(k)] = compute_pass_at_k(len(solutions), sum(is_correct), k)

                # Calculate majority vote
                majority_vote = get_majority_vote(solutions)
                is_correct_majority = self.check_correct(majority_vote, ground_truth, task_id)

                # Per-task 4-way breakdown of the K generations.  At base-eval
                # time the judge hasn't run yet, so cot_false / invalid_format
                # are always 0; they exist to keep the schema consistent with
                # CoT records produced by `evalhub cot metrics`.
                true_count = sum(1 for x in is_correct if x is True)
                false_count = sum(1 for x in is_correct if x is False)
                per_task_counts = {
                    "true": true_count,
                    "false": false_count,
                    "cot_false": 0,
                    "invalid_format": 0,
                }

                result = {
                    "task_id": task_id,
                    "solutions": solutions,
                    "ground_truth": self.groundtruth[task_id].answer,
                    "correct": is_correct,
                    "pass_at_k": pass_at_k,
                    "majority_vote": majority_vote,
                    "is_correct_majority": is_correct_majority,
                    "per_task_counts": per_task_counts,
                }

                progress.update(eval_task, advance=1)
                results.append(result)
                correct += int(is_correct_majority)

        # Calculate aggregate metrics
        pass_at_k = {
            k: sum(result["pass_at_k"].get(k, 0) for result in results) / total for k in results[0]["pass_at_k"]
        }
        cons_at_k = correct / total
        n_generations = len(results[0]["solutions"])

        # Aggregate per-task counts so summary.json carries totals (true/false/
        # cot_false/invalid_format) instead of forcing downstream tooling to
        # re-derive them from the per-task results.jsonl. cot_false /
        # invalid_format are always 0 at base time; they exist for schema
        # symmetry with the CoT-eval summary.
        true_count_total = sum(r["per_task_counts"]["true"] for r in results)
        false_count_total = sum(r["per_task_counts"]["false"] for r in results)

        # Log metrics
        for k, value in pass_at_k.items():
            logger.info(f"Pass@{k}: {value:.2%}")
        logger.info(f"Cons@{n_generations}: {cons_at_k:.2%}")

        # Save detailed results
        result_path = output_dir / f"{self.name}_results.jsonl"
        with open(result_path, "wb") as f:
            for result in results:
                try:
                    f.write(orjson.dumps(result) + b"\n")
                except Exception as e:
                    logger.error(f"Error dumping result: {result.keys()}")
                    logger.error(f"Error: {e}")
                    exit(1)
        logger.info(f"Evaluation results saved to {result_path}")

        # Save summary
        summary_path = output_dir / f"{self.name}_summary.json"
        summary = {
            "pass_at_k": pass_at_k,
            "cons_at_k": cons_at_k,
            "total_tasks": total,
            "total_generations": total * n_generations,
            "true_count": true_count_total,
            "false_count": false_count_total,
            "cot_false_count": 0,
            "invalid_format_count": 0,
        }
        with open(summary_path, "wb") as f:
            f.write(orjson.dumps(summary))
        logger.info(f"Evaluation summary saved to {summary_path}")

        # Per-task CSV — one row per task_id with all the data a researcher
        # typically wants in Excel: per-K pass rates, the four count buckets,
        # ground truth, majority vote, and whether the majority was correct.
        csv_path = output_dir / f"{self.name}_per_task.csv"
        write_per_task_csv(results, csv_path, has_cot=False)
        logger.info(f"Per-task CSV saved to {csv_path}")


def write_per_task_csv(
    results: list[dict[str, Any]],
    csv_path: PathLike,
    has_cot: bool,
) -> None:
    """Write per-task CSV with K-axis pass@k columns, counts, and ground truth.

    When ``has_cot`` is True the CSV's count columns reflect post-judge values
    (cot_false / invalid_format may be > 0). When False they are always 0 in
    those columns — convenient for diff-ing base vs CoT runs side by side.
    """
    if not results:
        return
    k_keys = sorted(results[0].get("pass_at_k", {}).keys(), key=lambda x: int(x))
    fieldnames = (
        ["task_id", "true", "false", "cot_false", "invalid_format"]
        + [f"pass@{k}" for k in k_keys]
        + ["ground_truth", "majority_vote", "is_correct_majority"]
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            counts = r.get("per_task_counts", {})
            row = {
                "task_id": r["task_id"],
                "true": counts.get("true", 0),
                "false": counts.get("false", 0),
                "cot_false": counts.get("cot_false", 0),
                "invalid_format": counts.get("invalid_format", 0),
                "ground_truth": r.get("ground_truth", ""),
                "majority_vote": r.get("majority_vote", ""),
                "is_correct_majority": r.get("is_correct_majority", ""),
            }
            for k in k_keys:
                row[f"pass@{k}"] = r["pass_at_k"].get(k, "")
            writer.writerow(row)
