#!/bin/bash
set -euo pipefail

CONFIG_FILE="scripts/configs/cot_judge.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Configuration file not found: $CONFIG_FILE"
    exit 1
fi
source "$CONFIG_FILE"

IFS=' ' read -r -a SUFFIX_LIST <<< "${SUFFIXES}"
IFS=' ' read -r -a TEMP_LIST <<< "${TEMPERATURES}"

mkdir -p logs

echo "[INFO] Starting job submission process for LLM-as-a-Judge (Sequential mode with dependencies)..."

PREV_JOB_ID=""

for base_model in ${BASE_MODELS}; do
    BASE_SAFE_NAME=$(basename "$base_model")
    
    for judge_model in ${JUDGE_MODELS}; do
        for benchmark in ${BENCHMARKS}; do
            for suffix in "${SUFFIX_LIST[@]}"; do
                K_VAL="${suffix#_}"

                for temp in "${TEMP_LIST[@]}"; do
                    INPUT_FILE="data/passatk_filtered/${BASE_SAFE_NAME}/${benchmark}${suffix}_corrects.jsonl"
                    
                    echo "[INFO] Submitting -> Benchmark: ${benchmark}${suffix} | Temp: ${temp} | Base: ${BASE_SAFE_NAME} | K: ${K_VAL}"
                    
                    DEPENDENCY_ARG=""
                    if [[ -n "$PREV_JOB_ID" ]]; then
                        DEPENDENCY_ARG="--dependency=afterany:${PREV_JOB_ID}"
                    fi
                    
                    PREV_JOB_ID=$(sbatch $DEPENDENCY_ARG \
                        --parsable \
                        --job-name="judge_$(basename "$judge_model")" \
                        --output="logs/judge_%j.out" \
                        --error="logs/judge_%j.err" \
                        scripts/cot_judge_pipeline/03_run_judge_worker.sh \
                        "$judge_model" "$base_model" "$benchmark" "$INPUT_FILE" "$temp" "$K_VAL")
                    
                    echo "[INFO] Submitted Job ID: $PREV_JOB_ID (Waiting for: ${DEPENDENCY_ARG:-None})"
                done
            done
        done
    done
done

echo "[INFO] All sequential judge jobs have been submitted to the cluster."