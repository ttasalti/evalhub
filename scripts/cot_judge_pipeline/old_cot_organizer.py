import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize and filter correct CoT responses.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing results")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name (e.g., aime2025)")
    parser.add_argument("--max_tasks", type=int, default=1024, help="Maximum number of tasks to process")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    results_file = base_dir / f"{args.benchmark}_results.jsonl"
    raw_file = base_dir / f"{args.benchmark}_raw.jsonl"
    output_file = base_dir / f"{args.benchmark}_correct_filtered_{args.max_tasks}.jsonl"

    if not results_file.exists() or not raw_file.exists():
        logger.error("Required input files not found in %s", base_dir)
        return

    correct_indices: Dict[str, List[int]] = defaultdict(list)
    ground_truths: Dict[str, str] = {}
    generated_solutions: Dict[str, Dict[int, str]] = defaultdict(dict)

    with results_file.open("r", encoding="utf-8") as f_res:
        for i, line in enumerate(f_res):
            if i >= args.max_tasks:
                break
            
            data = json.loads(line)
            if not (task_id := data.get("task_id")):
                continue
                
            ground_truths[task_id] = data.get("ground_truth", "")
            
            for idx, is_correct in enumerate(data.get("correct", [])):
                if is_correct:
                    correct_indices[task_id].append(idx)
                    solutions = data.get("solutions", [])
                    generated_solutions[task_id][idx] = solutions[idx] if idx < len(solutions) else ""

    current_task_indices: Dict[str, int] = defaultdict(int)
    saved_count = 0

    with raw_file.open("r", encoding="utf-8") as f_raw, output_file.open("w", encoding="utf-8") as f_out:
        for line in f_raw:
            data = json.loads(line)
            if not (task_id := data.get("task_id")) or task_id not in correct_indices:
                continue
                
            current_idx = current_task_indices[task_id]
            if current_idx in correct_indices[task_id]:
                output_data = {
                    "task_id": task_id,
                    "ground_truth": ground_truths.get(task_id, ""),
                    "generated_answer": generated_solutions[task_id].get(current_idx, ""),
                    "raw_response": data.get("response", {})
                }
                f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                saved_count += 1
                
            current_task_indices[task_id] += 1

    logger.info("Processed %d tasks. Saved %d correct responses to %s", args.max_tasks, saved_count, output_file.name)

if __name__ == "__main__":
    main()
