import json
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    
    if not input_path.exists():
        print(f"[UYARI] {input_path} bulunamadı, atlanıyor.")
        return

    task_groups = defaultdict(list)

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                task_id = data.get("task_id", "")
                
                # Eğer data "solution" array ise
                sol = data.get("solution", "")
                if isinstance(sol, list):
                    task_groups[task_id].extend([str(s).lower() for s in sol])
                elif task_id:
                    task_groups[task_id].append(str(sol).lower())
                    
            except json.JSONDecodeError:
                continue
                
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for task_id, solutions in task_groups.items():
            yes_count = sum(1 for s in solutions if "yes" in s)
            majority_correct = yes_count > (len(solutions) / 2) if len(solutions) > 0 else False
            
            output_data = {
                "task_id": task_id,
                "solutions": solutions,
                "majority_correct": majority_correct
            }
            f_out.write(json.dumps(output_data) + '\n')

if __name__ == "__main__":
    main()