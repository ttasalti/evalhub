#!/bin/bash
#SBATCH --job-name=evalhub_judge_run
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --nodelist=nsdl2

set -euo pipefail

# Map positional arguments to standard pipeline variables
HF_MODEL_ID=$1
BASE_MODEL_ID=$2
TASK=$3
INPUT_FILE=$4
TEMPERATURE=$5
K_VAL=$6

eval "$(conda shell.bash hook)"
conda activate evalhub_env

CONFIG_FILE="${SLURM_SUBMIT_DIR}/scripts/configs/cot_judge.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Configuration file not found: $CONFIG_FILE"
    exit 1
fi
source "$CONFIG_FILE"

MODEL_SAFE_NAME=$(basename "$HF_MODEL_ID")
BASE_SAFE_NAME=$(basename "$BASE_MODEL_ID")

DYNAMIC_PORT=$((30000 + SLURM_JOB_ID % 10000))
NUM_GPUS=1

export HF_TOKEN
export SGLANG_DISABLE_CUDNN_CHECK=1
export HOSTED_VLLM_API_BASE="http://127.0.0.1:${DYNAMIC_PORT}/v1"
export HOSTED_VLLM_API_KEY="EMPTY"

echo "[INFO] Starting sglang_router for judge model $HF_MODEL_ID with $NUM_GPUS GPUs on port $DYNAMIC_PORT"

# Enforcing fail-fast memory boundaries for stability on H200s.
# --mem-fraction-static 0.85 ensures immediate failure on OOM rather than hanging when handling high NUM_WORKERS concurrency.
python -m sglang_router.launch_server \
    --model-path "$HF_MODEL_ID" \
    --host 127.0.0.1 \
    --port "$DYNAMIC_PORT" \
    --dp "$NUM_GPUS" \
    --router-balance-abs-threshold 1 \
    --mem-fraction-static 0.85 \
    --trust-remote-code &
SERVER_PID=$!

echo "[INFO] Waiting for server initialization..."
while ! curl -s "http://127.0.0.1:${DYNAMIC_PORT}/v1/models" > /dev/null; do
    sleep 10
done
sleep 300
echo "[INFO] Server is online."

# Ensure output path complies strictly with the required structure
OUT_DIR="results/judgments/${BASE_SAFE_NAME}_evaluated_by_${MODEL_SAFE_NAME}_${MAX_COMPLETION_TOKENS}/${TASK}_t${TEMPERATURE}_k${K_VAL}"
mkdir -p "$OUT_DIR"

# Merge additional arguments with the target file path dynamically
if [[ -n "${OVERRIDE_ARGS:-}" ]] && [[ "$OVERRIDE_ARGS" == *\} ]]; then
    FINAL_OVERRIDE="${OVERRIDE_ARGS%\}}, \"file_path\": \"$INPUT_FILE\"}"
else
    FINAL_OVERRIDE="{\"file_path\": \"$INPUT_FILE\"}"
fi

GEN_ARGS=(
    --model "hosted_vllm/$HF_MODEL_ID"
    --tasks "math_judge"
    --output-dir "$OUT_DIR"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --max-completion-tokens "$MAX_COMPLETION_TOKENS"
    --frequency-penalty "$FREQUENCY_PENALTY"
    --presence-penalty "$PRESENCE_PENALTY"
    --n-samples "$N_SAMPLES"
    --num-workers "$NUM_WORKERS"
    --timeout "$TIMEOUT"
    --override-args "$FINAL_OVERRIDE"
)

[[ -n "${STOP:-}" ]] && GEN_ARGS+=(--stop "$STOP")
[[ -n "${SYSTEM_PROMPT:-}" ]] && GEN_ARGS+=(--system-prompt "$SYSTEM_PROMPT")
[[ -n "${TOOL_CONFIG:-}" ]] && GEN_ARGS+=(--tool-config "$TOOL_CONFIG")

echo "[INFO] Initiating Judge Generation Phase for task: $TASK"
evalhub gen "${GEN_ARGS[@]}"

echo "[INFO] Initiating Judge Evaluation Phase for task: $TASK"
evalhub eval \
    --tasks "math_judge" \
    --solutions "$OUT_DIR/math_judge.jsonl" \
    --output-dir "$OUT_DIR"

echo "[INFO] Shutting down server..."
kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
echo "[INFO] Evaluation worker completed successfully."