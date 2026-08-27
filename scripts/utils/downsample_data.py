import json
import logging
import argparse
from collections import defaultdict
from evalhub.benchmarks.registry import DATASET_MAP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample dataset and generate solution files.")
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--input", required=True, help="Input raw JSONL file")
    parser.add_argument("--out_raw", required=True, help="Output downsampled raw JSONL")
    parser.add_argument("--out_sol", required=True, help="Output extracted solutions JSONL")
    parser.add_argument("--n", type=int, default=256, help="Maximum samples per task ID")
    return parser.parse_args()

def extract_content(response_data: dict) -> str:
    if content := response_data.get("content"):
        return content
    try:
        return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except (IndexError, AttributeError):
        return ""

def main() -> None:
    args = parse_args()
    
    if args.task not in DATASET_MAP:
        logger.error("Task '%s' not found in DATASET_MAP.", args.task)
        return

    dataset_class = DATASET_MAP[args.task]()
    task_counts = defaultdict(int)
    processed_count = 0

    with open(args.input, "r", encoding="utf-8") as file_in, \
         open(args.out_raw, "w", encoding="utf-8") as file_raw, \
         open(args.out_sol, "w", encoding="utf-8") as file_sol:
        
        for line in file_in:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                if not (task_id := data.get("task_id")):
                    continue
                    
                if task_counts[task_id] < args.n:
                    file_raw.write(json.dumps(data, ensure_ascii=False) + "\n")
                    
                    content = extract_content(data.get("response", {}))
                    solution_data = {
                        "task_id": task_id,
                        "solution": dataset_class.extract_solution(task_id, content)
                    }
                    file_sol.write(json.dumps(solution_data, ensure_ascii=False) + "\n")
                    
                    task_counts[task_id] += 1
                    processed_count += 1
                    
            except json.JSONDecodeError:
                continue

    logger.info("Downsampling complete. Processed %d valid entries.", processed_count)

if __name__ == "__main__":
    main()
