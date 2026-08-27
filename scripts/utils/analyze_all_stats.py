import json
import argparse
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="results")
    return parser.parse_args()

def main():
    args = parse_args()
    base_path = Path(args.base_dir).resolve()

    if not base_path.exists():
        print(f"ERROR: Directory not found: {base_path}", file=sys.stderr)
        sys.exit(1)

    processed_files = 0
    total_lines = 0
    skipped_json_error = 0
    skipped_no_task_id = 0
    successful_writes = 0

    for file_path in base_path.rglob("*_results.jsonl"):
        processed_files += 1
        
        # Orijinal dosyanın bulunduğu klasörü ve yeni dosya adını belirliyoruz
        # Örn: aime2026_results.jsonl -> aime2026_generation_stats.jsonl
        output_filename = file_path.name.replace("_results.jsonl", "_generation_stats.jsonl")
        output_file_path = file_path.parent / output_filename
        
        stats_list = []
        
        with file_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                total_lines += 1
                
                try:
                    data = json.loads(line)
                    task_id = data.get("task_id")
                    
                    if not task_id:
                        skipped_no_task_id += 1
                        continue
                        
                    correct_list = data.get("correct", [])
                    gen_count = len(correct_list)
                    true_count = sum(1 for x in correct_list if x is True)
                    
                    # Burada file_path'i sildik çünkü zaten kendi klasörüne kaydediyoruz
                    output = {
                        "task_id": task_id,
                        "generation_count": gen_count,
                        "true_count": true_count
                    }
                    stats_list.append(output)
                    
                except json.JSONDecodeError:
                    skipped_json_error += 1
                    continue
        
        # Elde edilen istatistikleri doğrudan o klasördeki yeni dosyaya yazıyoruz
        if stats_list:
            with output_file_path.open("w", encoding="utf-8") as f_out:
                for stat in stats_list:
                    f_out.write(json.dumps(stat, ensure_ascii=False) + "\n")
                    successful_writes += 1

    print("\n--- EXECUTION REPORT ---", file=sys.stderr)
    print(f"Total Files Processed  : {processed_files}", file=sys.stderr)
    print(f"Total Lines Read       : {total_lines}", file=sys.stderr)
    print(f"Skipped (JSON Error)   : {skipped_json_error}", file=sys.stderr)
    print(f"Skipped (No task_id)   : {skipped_no_task_id}", file=sys.stderr)
    print(f"Successful Writes      : {successful_writes}", file=sys.stderr)
    print("------------------------\n", file=sys.stderr)

if __name__ == "__main__":
    main()