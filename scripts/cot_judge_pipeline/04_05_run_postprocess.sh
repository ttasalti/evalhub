#!/bin/bash
#SBATCH --job-name=judge_postprocess
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate evalhub_env

JUDGMENT_ROOT="results/judgments"
BASE_DIR="results"

echo "Post-processing pipeline başlatılıyor (Dinamik K Eşleştirme)..."
echo "--------------------------------------------------------"

find "$JUDGMENT_ROOT" -type d | while read -r TARGET_DIR; do
    
    # Hedef klasörde raw veya normal jsonl var mı kontrol et
    if [[ -f "${TARGET_DIR}/math_judge.jsonl" ]]; then
        INPUT_FILE="${TARGET_DIR}/math_judge.jsonl"
    elif [[ -f "${TARGET_DIR}/math_judge_raw.jsonl" ]]; then
        INPUT_FILE="${TARGET_DIR}/math_judge_raw.jsonl"
    else
        continue
    fi
    
    EVAL_DIR_NAME=$(basename $(dirname "$TARGET_DIR")) 
    BASE_MODEL="${EVAL_DIR_NAME%%_evaluated_by_*}"    
    BENCH_FULL=$(basename "$TARGET_DIR")              
    
    # Regex ile BENCH_BASE ve K_VAL ayıklama (örn: aime2026_t1.2_k64 -> aime2026 ve 64)
    if [[ "$BENCH_FULL" =~ ^(.*)_t[0-9\.]+_k([0-9]+)$ ]]; then
        BENCH_BASE="${BASH_REMATCH[1]}"
        K_VAL="${BASH_REMATCH[2]}"
        BASE_RESULTS_DIR="${BASE_DIR}/${BASE_MODEL}/${BENCH_BASE}_${K_VAL}"
    elif [[ "$BENCH_FULL" =~ ^(.*)_k([0-9]+)$ ]]; then
        BENCH_BASE="${BASH_REMATCH[1]}"
        K_VAL="${BASH_REMATCH[2]}"
        BASE_RESULTS_DIR="${BASE_DIR}/${BASE_MODEL}/${BENCH_BASE}_${K_VAL}"
    else
        BENCH_BASE="$BENCH_FULL"
        BASE_RESULTS_DIR="${BASE_DIR}/${BASE_MODEL}/${BENCH_BASE}"
    fi
    
    # O klasörün içindeki _results.jsonl ile biten asıl dosyayı bul
    BASE_RESULTS_FILE=$(find "$BASE_RESULTS_DIR" -maxdepth 1 -name "*_results.jsonl" 2>/dev/null | head -n 1 || true)
    
    MAJORITY_FILE="${TARGET_DIR}/math_judge_majority.jsonl"
    JUDGED_RESULTS_FILE="${TARGET_DIR}/${BENCH_FULL}_results_judged.jsonl"
    SUMMARY_FILE="${TARGET_DIR}/summary.json"
    
    if [[ -n "$BASE_RESULTS_FILE" ]] && [[ -f "$BASE_RESULTS_FILE" ]]; then
        echo "[EŞLEŞTİ] Judge : $BENCH_FULL"
        echo "          Base  : $BASE_RESULTS_DIR"
        
        # Adım 4: Majority Vote Hesapla
        python scripts/cot_judge_pipeline/04_aggregate_votes.py \
            --input_file "$INPUT_FILE" \
            --output_file "$MAJORITY_FILE"
        
        # Adım 5: Base dosyayı güncelle ve metrikleri Pass@K formatında hesapla
        python scripts/cot_judge_pipeline/05_apply_metrics.py \
            --base_results_file "$BASE_RESULTS_FILE" \
            --judge_majority_file "$MAJORITY_FILE" \
            --output_file "$JUDGED_RESULTS_FILE" \
            --summary_file "$SUMMARY_FILE"
            
        echo "          -> Tamamlandı."
        echo "--------------------------------------------------------"
    else
        echo "[ATLANDI] Ana sonuç bulunamadı! Aranan path: $BASE_RESULTS_DIR/*_results.jsonl"
    fi
done

echo "Tüm dinamik eşleştirmeler ve post-process işlemleri başarıyla tamamlandı!"