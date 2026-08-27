#!/bin/bash
#SBATCH --job-name=eval_64
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err           

set -e
eval "$(conda shell.bash hook)"
conda activate evalhub_env

# Yapılandırma
MODEL="Qwen3.5-4B-Base"
ORIG_TASK="aime2026"
N=64
NEW_TASK="${ORIG_TASK}_${N}"

# Klasör ve Dosya Yolları
BASE_RESULTS="results"
SOURCE_RAW="${BASE_RESULTS}/${MODEL}/${ORIG_TASK}/${ORIG_TASK}_raw.jsonl"
TARGET_DIR="${BASE_RESULTS}/${MODEL}/${NEW_TASK}"

mkdir -p "$TARGET_DIR" logs

# 1. Ham veriyi filtrele ve solution dosyasını oluştur
python3 scripts/utils/downsample_data.py \
    --task "$ORIG_TASK" \
    --input "$SOURCE_RAW" \
    --out_raw "${TARGET_DIR}/${NEW_TASK}_raw.jsonl" \
    --out_sol "${TARGET_DIR}/${NEW_TASK}.jsonl" \
    --n $N

# 2. Mevcut evalhub CLI ile değerlendirme yap
# analyze_generations.py'nin bulabilmesi için dosya ismini NEW_TASK formatında üretiyoruz
evalhub eval \
    --tasks "$ORIG_TASK" \
    --solutions "${TARGET_DIR}/${NEW_TASK}.jsonl" \
    --output-dir "$TARGET_DIR"

# evalhub eval çıktı ismini analyze_generations.py'nin beklediği formata taşı
mv "${TARGET_DIR}/${ORIG_TASK}_results.jsonl" "${TARGET_DIR}/${NEW_TASK}_results.jsonl" 2>/dev/null || true
mv "${TARGET_DIR}/${ORIG_TASK}_summary.json" "${TARGET_DIR}/${NEW_TASK}_summary.json" 2>/dev/null || true

# 3. Kendi analyze_generations.py scriptinizi gerekli parametrelerle çalıştır
python3 scripts/pass_k_pipeline/03_analyze_stats.py \
    --model "$MODEL" \
    --benchmark "$NEW_TASK" \
    --base_dir "$BASE_RESULTS"