import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import math

# Pass@K formülü
def compute_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_results_file", type=str, required=True)
    parser.add_argument("--judge_majority_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--summary_file", type=str, required=True)
    parser.add_argument("--stats_file", type=str, required=False) # Geriye dönük uyumluluk
    args = parser.parse_args()
    
    # 1. Majority oylarını oku
    judge_dict = {}
    with open(args.judge_majority_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            judge_dict[data["task_id"]] = data["majority_correct"]

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    sum_pass_at_k = defaultdict(float)
    sum_cons_at_k = 0.0
    total_tasks = 0

    stats = {
        "total_tasks": 0,
        "total_generations": 0,
        "true_count": 0,
        "false_count": 0,
        "cot_false_count": 0,
    }
    
    # 2. Base sonuçları güncelle ve Pass@K / Cons@K hesapla
    with open(args.base_results_file, "r", encoding="utf-8") as f_in, \
         open(args.output_file, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            data = json.loads(line)
            task_id = data["task_id"]
            correct_arr = data.get("correct", [])
            solutions = data.get("solutions", [])
            n_generations = len(correct_arr)
            
            # True ama hakem tarafından reddedilenleri "cot_false" yap
            for i in range(n_generations):
                gen_id = f"{task_id}_gen_{i}"
                if correct_arr[i] is True and gen_id in judge_dict:
                    if not judge_dict[gen_id]:
                        correct_arr[i] = "cot_false"
            
            # İstatistikleri güncelle
            true_count = sum(1 for x in correct_arr if x is True)
            false_count = sum(1 for x in correct_arr if x is False)
            cot_false_count = sum(1 for x in correct_arr if x == "cot_false")

            stats["total_tasks"] += 1
            stats["total_generations"] += n_generations
            stats["true_count"] += true_count
            stats["false_count"] += false_count
            stats["cot_false_count"] += cot_false_count
            
            # Yeni Pass@K değerlerini hesapla
            new_pass_at_k = {}
            for k_str in data.get("pass_at_k", {}).keys():
                k_val = int(k_str)
                if k_val <= n_generations:
                    val = compute_pass_at_k(n_generations, true_count, k_val)
                else:
                    val = 1.0
                new_pass_at_k[k_str] = val
                sum_pass_at_k[k_str] += val
                
            # Consensus@K (Çoğunluk Oylaması) Hesaplama
            if solutions:
                sol_strs = [str(s) if s is not None else "" for s in solutions]
                counter = Counter(sol_strs)
                if counter:
                    # En çok tekrar eden cevabı bul
                    best_sol, _ = counter.most_common(1)[0]
                    # Bu çoğunluk cevabı aynı zamanda doğru (ve CoT onaylı) bir cevap mı?
                    is_consensus_correct = any(correct_arr[i] is True for i, sol in enumerate(sol_strs) if sol == best_sol)
                    if is_consensus_correct:
                        sum_cons_at_k += 1.0
                
            total_tasks += 1
            data["correct"] = correct_arr
            data["pass_at_k"] = new_pass_at_k
            
            f_out.write(json.dumps(data) + "\n")
            
    # 3. İstenilen Özel Formatta Summary Dosyasını oluştur (JSONL satırı)
    if total_tasks > 0:
        pass_at_k_summary = {k: (v / total_tasks) for k, v in sum_pass_at_k.items()}
        cons_at_k_val = sum_cons_at_k / total_tasks
        
        summary_data = {
            "pass_at_k": pass_at_k_summary,
            "cons_at_k": cons_at_k_val
        }
        
        # JSONL formatı (indent yok, tek satır)
        with open(args.summary_file, "w", encoding="utf-8") as f_sum:
            f_sum.write(json.dumps(summary_data) + "\n")
            
        # İstek olursa detaylı stats dosyasını da yaz (Ana summary'i bozmadan)
        if args.stats_file:
            with open(args.stats_file, "w", encoding="utf-8") as f_stats:
                json.dump(stats, f_stats, indent=4)
                
        print(f"[BAŞARILI] {args.summary_file} istenilen formatta oluşturuldu. Toplam Task: {total_tasks}")

if __name__ == "__main__":
    main()