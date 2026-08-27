#!/bin/bash
set -euo pipefail

CONFIG_FILE="scripts/configs/pass_k_eval.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Configuration file not found: $CONFIG_FILE"
    exit 1
fi

source "$CONFIG_FILE"

echo "[INFO] Starting job submission process for Pass@K evaluation..."

for MODEL in $MODELS; do
    echo "[INFO] Submitting Slurm job for model: $MODEL"
    sbatch --export=ALL,TARGET_MODEL="$MODEL" scripts/pass_k_pipeline/02_run_eval_worker.sh
done

echo "[INFO] All evaluation jobs have been submitted to the cluster."