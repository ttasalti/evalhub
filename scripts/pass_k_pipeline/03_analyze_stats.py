import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze generation statistics from EvalHub results.")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    parser.add_argument("--base_dir", type=str, default="results", help="Base directory for results")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    results_dir = Path(args.base_dir) / args.model / args.benchmark
    input_file = results_dir / f"{args.benchmark}_results.jsonl"
    output_file = results_dir / f"{args.benchmark}_generation_stats.json"

    if not input_file.exists():
        logger.error("Input file not found: %s", input_file)
        return

    stats_list: List[Dict[str, Any]] = []

    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                if not (task_id := data.get("task_id")):
                    continue
                    
                correct_list = data.get("correct", [])
                generated_k = len(correct_list)
                correct_answers = sum(1 for x in correct_list if x is True)
                
                stats_list.append({
                    "task_id": task_id,
                    "generated_answers_k": generated_k,
                    "correct_answers": correct_answers,
                    "wrong_answers": generated_k - correct_answers
                })
            except json.JSONDecodeError:
                continue

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f_out:
        json.dump(stats_list, f_out, indent=4, ensure_ascii=False)
        
    logger.info("Processed %d tasks. Generation statistics saved to %s", len(stats_list), output_file.name)

if __name__ == "__main__":
    main()
