#!/bin/bash
#SBATCH --job-name=evalhub_run
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32  
#SBATCH --mem=256G            
#SBATCH --nodelist=nsdl2
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate evalhub_env

CONFIG_FILE="${SLURM_SUBMIT_DIR}/scripts/configs/pass_k_eval.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Configuration file not found: $CONFIG_FILE"
    exit 1
fi
source "$CONFIG_FILE"

HF_MODEL_ID=${TARGET_MODEL}
MODEL_SAFE_NAME=$(basename "$HF_MODEL_ID")
DYNAMIC_PORT=$((30000 + SLURM_JOB_ID % 10000))
NUM_GPUS=1

export HF_TOKEN
export SGLANG_DISABLE_CUDNN_CHECK=1
export HOSTED_VLLM_API_BASE="http://127.0.0.1:${DYNAMIC_PORT}/v1"
export HOSTED_VLLM_API_KEY="EMPTY"

echo "[INFO] Starting sglang_router for model $HF_MODEL_ID with $NUM_GPUS GPUs on port $DYNAMIC_PORT"

python -m sglang_router.launch_server \
    --model-path "$HF_MODEL_ID" \
    --host 127.0.0.1 \
    --port "$DYNAMIC_PORT" \
    --dp "$NUM_GPUS" \
    --router-balance-abs-threshold 1 &
SERVER_PID=$!

echo "[INFO] Waiting for server initialization..."
while ! curl -s "http://127.0.0.1:${DYNAMIC_PORT}/v1/models" > /dev/null; do
    sleep 10
done
sleep 300
echo "[INFO] Server is online."

for TASK in $TASKS; do
    BASE_DIR="${HOME}/evalhub/results/${MODEL_SAFE_NAME}/${TASK}"
    mkdir -p "$BASE_DIR"

    GEN_ARGS=(
        --model "hosted_vllm/$HF_MODEL_ID"
        --tasks "$TASK"
        --output-dir "$BASE_DIR"
        --temperature "$TEMPERATURE"
        --top-p "$TOP_P"
        --max-completion-tokens "$MAX_COMPLETION_TOKENS"
        --frequency-penalty "$FREQUENCY_PENALTY"
        --presence-penalty "$PRESENCE_PENALTY"
        --n-samples "$N_SAMPLES"
        --num-workers "$NUM_WORKERS"
        --timeout "$TIMEOUT"
    )

    [[ -n "${STOP:-}" ]] && GEN_ARGS+=(--stop "$STOP")
    [[ -n "${SYSTEM_PROMPT:-}" ]] && GEN_ARGS+=(--system-prompt "$SYSTEM_PROMPT")
    [[ -n "${OVERRIDE_ARGS:-}" ]] && GEN_ARGS+=(--override-args "$OVERRIDE_ARGS")
    
    echo "[INFO] Generating responses for task: $TASK"
    evalhub gen "${GEN_ARGS[@]}"

    echo "[INFO] Evaluating task: $TASK"
    evalhub eval \
        --tasks "$TASK" \
        --solutions "$BASE_DIR/${TASK}.jsonl" \
        --output-dir "$BASE_DIR"
done

echo "[INFO] Shutting down server..."
kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
echo "[INFO] Evaluation worker completed successfully."