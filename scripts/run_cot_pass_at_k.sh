#!/usr/bin/env bash
# ============================================================================
# End-to-end CoT-Pass@K pipeline.
#
# Pipeline stages
#   1. start vLLM server with the target model + its chat template
#   2. evalhub gen + evalhub eval  (standard Pass@K data)
#   3. shut target server down
#   4. evalhub cot extract         (filter correct generations)
#   5. start vLLM server with the judge model
#   6. evalhub gen + evalhub eval  (judge yes/no per generation)
#   7. shut judge server down
#   8. evalhub cot aggregate + cot metrics  (majority vote + CoT-Pass@K)
#
# All knobs come from an env file; pass it as $1 or set EVALHUB_PIPELINE_ENV.
# A defaults sample lives at scripts/cot_pipeline.env.example.
# ============================================================================
set -euo pipefail

# ----- locate the project root regardless of where the script was invoked
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ----- load env
ENV_FILE="${1:-${EVALHUB_PIPELINE_ENV:-${SCRIPT_DIR}/cot_pipeline.env}}"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[ERROR] env file not found: ${ENV_FILE}" >&2
    echo "        copy scripts/cot_pipeline.env.example and edit it." >&2
    exit 1
fi
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

# ----- required keys
: "${TARGET_MODEL:?TARGET_MODEL must be set}"
: "${JUDGE_MODEL:?JUDGE_MODEL must be set}"
: "${BENCHMARK:?BENCHMARK must be set}"

# ----- defaults (mirror evalhub's own defaults so we never silently drift)
TARGET_STATE="${TARGET_STATE:-non-think}"
JUDGE_STATE="${JUDGE_STATE:-think}"

TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-0.6}"
TARGET_TOP_P="${TARGET_TOP_P:-0.95}"
TARGET_N_SAMPLES="${TARGET_N_SAMPLES:-8}"
TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-2048}"
TARGET_NUM_WORKERS="${TARGET_NUM_WORKERS:-256}"
TARGET_TIMEOUT="${TARGET_TIMEOUT:-3600}"
TARGET_FREQUENCY_PENALTY="${TARGET_FREQUENCY_PENALTY:-0}"
TARGET_PRESENCE_PENALTY="${TARGET_PRESENCE_PENALTY:-0}"
TARGET_SYSTEM_PROMPT="${TARGET_SYSTEM_PROMPT:-}"
TARGET_STOP="${TARGET_STOP:-}"
TARGET_OVERRIDE_ARGS="${TARGET_OVERRIDE_ARGS:-}"
TARGET_TOOL_CONFIG="${TARGET_TOOL_CONFIG:-}"
TARGET_CALLBACK="${TARGET_CALLBACK:-}"
TARGET_MAX_TURNS="${TARGET_MAX_TURNS:-3}"
TARGET_ENABLE_MULTITURN="${TARGET_ENABLE_MULTITURN:-false}"
TARGET_RESUME="${TARGET_RESUME:-false}"
TARGET_PARALLEL_COUNT="${TARGET_PARALLEL_COUNT:-1}"

JUDGE_TASK="${JUDGE_TASK:-cot_judge}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.6}"
JUDGE_TOP_P="${JUDGE_TOP_P:-0.95}"
JUDGE_N_SAMPLES="${JUDGE_N_SAMPLES:-3}"
JUDGE_MAX_COMPLETION_TOKENS="${JUDGE_MAX_COMPLETION_TOKENS:-16384}"
JUDGE_NUM_WORKERS="${JUDGE_NUM_WORKERS:-256}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-3600}"
JUDGE_FREQUENCY_PENALTY="${JUDGE_FREQUENCY_PENALTY:-0}"
JUDGE_PRESENCE_PENALTY="${JUDGE_PRESENCE_PENALTY:-0}"
JUDGE_SYSTEM_PROMPT="${JUDGE_SYSTEM_PROMPT:-}"
JUDGE_STOP="${JUDGE_STOP:-}"
JUDGE_TOOL_CONFIG="${JUDGE_TOOL_CONFIG:-}"
JUDGE_CALLBACK="${JUDGE_CALLBACK:-}"
JUDGE_MAX_TURNS="${JUDGE_MAX_TURNS:-3}"
JUDGE_ENABLE_MULTITURN="${JUDGE_ENABLE_MULTITURN:-false}"
JUDGE_RESUME="${JUDGE_RESUME:-false}"
JUDGE_PARALLEL_COUNT="${JUDGE_PARALLEL_COUNT:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results}"
TARGET_PORT="${TARGET_PORT:-30000}"
JUDGE_PORT="${JUDGE_PORT:-30001}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"

mkdir -p "${OUTPUT_ROOT}" "${PROJECT_ROOT}/logs"

# ----- output layout
target_clean="$(basename "${TARGET_MODEL}")_state-${TARGET_STATE}"
judge_clean="$(basename "${JUDGE_MODEL}")_state-${JUDGE_STATE}"
TARGET_DIR="${OUTPUT_ROOT}/${target_clean}_t${TARGET_TEMPERATURE}_max${TARGET_MAX_COMPLETION_TOKENS}/${BENCHMARK}"
JUDGE_DIR="${OUTPUT_ROOT}/judgments/${target_clean}_judged_by_${judge_clean}_t${JUDGE_TEMPERATURE}_max${JUDGE_MAX_COMPLETION_TOKENS}/${BENCHMARK}"
mkdir -p "${TARGET_DIR}" "${JUDGE_DIR}"

# ----- helpers
resolve_template() {
    local model="$1" state="$2"
    python -m evalhub.utils.model_state --model "${model}" --state "${state}" --allow-missing
}

start_vllm() {
    local model="$1" port="$2" tp="$3" state="$4" log_file="$5"
    local template
    template="$(resolve_template "${model}" "${state}" || true)"
    local args=(--model "${model}" --port "${port}" --tensor-parallel-size "${tp}" --trust-remote-code)
    if [[ -n "${template}" ]]; then
        echo "[INFO] Using chat template: ${template}"
        args+=(--chat-template "${template}")
    else
        echo "[INFO] No registered template for ${model}/${state} — using tokenizer default."
    fi
    python -m vllm.entrypoints.openai.api_server "${args[@]}" >>"${log_file}" 2>&1 &
    SERVER_PID=$!
    local waited=0
    until curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[ERROR] vLLM died. See ${log_file}" >&2
            exit 1
        fi
        sleep 5
        waited=$((waited + 5))
        if (( waited >= HEALTH_TIMEOUT )); then
            echo "[ERROR] vLLM health timeout after ${HEALTH_TIMEOUT}s on port ${port}" >&2
            kill "${SERVER_PID}" 2>/dev/null || true
            exit 1
        fi
    done
    echo "[INFO] vLLM healthy on port ${port} (pid ${SERVER_PID})"
}

stop_vllm() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap stop_vllm EXIT

# Append every populated optional arg to a gen call.
build_gen_args() {
    local role="$1"
    local -n out_array="$2"
    local prefix="${role}_"
    local top_p_var="${prefix}TOP_P"
    local fp_var="${prefix}FREQUENCY_PENALTY"
    local pp_var="${prefix}PRESENCE_PENALTY"
    local timeout_var="${prefix}TIMEOUT"
    local stop_var="${prefix}STOP"
    local sys_var="${prefix}SYSTEM_PROMPT"
    local override_var="${prefix}OVERRIDE_ARGS"
    local tool_var="${prefix}TOOL_CONFIG"
    local cb_var="${prefix}CALLBACK"
    local mt_var="${prefix}MAX_TURNS"
    local emt_var="${prefix}ENABLE_MULTITURN"
    local resume_var="${prefix}RESUME"
    local state_var="${prefix}STATE"

    [[ -n "${!top_p_var:-}" ]]         && out_array+=(--top-p "${!top_p_var}")
    [[ -n "${!fp_var:-}" ]]            && out_array+=(--frequency-penalty "${!fp_var}")
    [[ -n "${!pp_var:-}" ]]            && out_array+=(--presence-penalty "${!pp_var}")
    [[ -n "${!timeout_var:-}" ]]       && out_array+=(--timeout "${!timeout_var}")
    [[ -n "${!stop_var:-}" ]]          && out_array+=(--stop "${!stop_var}")
    [[ -n "${!sys_var:-}" ]]           && out_array+=(--system-prompt "${!sys_var}")
    [[ -n "${!tool_var:-}" ]]          && out_array+=(--tool-config "${!tool_var}")
    [[ -n "${!cb_var:-}" ]]            && out_array+=(--callback "${!cb_var}")
    [[ -n "${!mt_var:-}" ]]            && out_array+=(--max-turns "${!mt_var}")
    [[ -n "${!state_var:-}" ]]         && out_array+=(--model-state "${!state_var}")
    [[ "${!emt_var:-false}" == "true" ]]    && out_array+=(--enable-multiturn)
    [[ "${!resume_var:-false}" == "true" ]] && out_array+=(--resume)
}

# ============================================================================
# Stage 1 — base generation + base evaluation
# ============================================================================
echo "==[1/3]== Base generation & evaluation =================================="
start_vllm "${TARGET_MODEL}" "${TARGET_PORT}" "${TARGET_PARALLEL_COUNT}" "${TARGET_STATE}" \
    "${PROJECT_ROOT}/logs/vllm_target_${TARGET_PORT}.log"
export HOSTED_VLLM_API_BASE="http://127.0.0.1:${TARGET_PORT}/v1"
export HOSTED_VLLM_API_KEY="EMPTY"

target_args=(
    --model "hosted_vllm/${TARGET_MODEL}"
    --tasks "${BENCHMARK}"
    --temperature "${TARGET_TEMPERATURE}"
    --n-samples "${TARGET_N_SAMPLES}"
    --num-workers "${TARGET_NUM_WORKERS}"
    --max-completion-tokens "${TARGET_MAX_COMPLETION_TOKENS}"
    --output-dir "${TARGET_DIR}"
)
build_gen_args "TARGET" target_args
[[ -n "${TARGET_OVERRIDE_ARGS}" ]] && target_args+=(--override-args "${TARGET_OVERRIDE_ARGS}")
evalhub gen "${target_args[@]}"

target_solutions="${TARGET_DIR}/${BENCHMARK}.jsonl"
[[ -f "${target_solutions}" ]] || target_solutions="${TARGET_DIR}/${BENCHMARK}_raw.jsonl"
target_eval_args=(
    --tasks "${BENCHMARK}"
    --solutions "${target_solutions}"
    --output-dir "${TARGET_DIR}"
)
[[ -n "${TARGET_OVERRIDE_ARGS}" ]] && target_eval_args+=(--override-args "${TARGET_OVERRIDE_ARGS}")
evalhub eval "${target_eval_args[@]}"

stop_vllm

# ============================================================================
# Stage 2 — extract correct generations, then judge them
# ============================================================================
echo "==[2/3]== Extract correct generations & judge ============================"
JUDGE_INPUT="${JUDGE_DIR}/${BENCHMARK}_cot_judge_input.jsonl"
evalhub cot extract \
    --base-results "${TARGET_DIR}/${BENCHMARK}_results.jsonl" \
    --base-raw "${TARGET_DIR}/${BENCHMARK}_raw.jsonl" \
    --output "${JUDGE_INPUT}"

if [[ ! -s "${JUDGE_INPUT}" ]]; then
    echo "[INFO] No correct base generations; CoT-Pass@K = 0 by definition. Stopping."
    echo '{"pass_at_k": {}, "cons_at_k": 0.0, "note": "no base-correct samples"}' \
        > "${JUDGE_DIR}/${BENCHMARK}_cot_summary.json"
    exit 0
fi

start_vllm "${JUDGE_MODEL}" "${JUDGE_PORT}" "${JUDGE_PARALLEL_COUNT}" "${JUDGE_STATE}" \
    "${PROJECT_ROOT}/logs/vllm_judge_${JUDGE_PORT}.log"
export HOSTED_VLLM_API_BASE="http://127.0.0.1:${JUDGE_PORT}/v1"
export HOSTED_VLLM_API_KEY="EMPTY"

judge_override="{\"file_path\": \"${JUDGE_INPUT}\"}"
judge_args=(
    --model "hosted_vllm/${JUDGE_MODEL}"
    --tasks "${JUDGE_TASK}"
    --temperature "${JUDGE_TEMPERATURE}"
    --n-samples "${JUDGE_N_SAMPLES}"
    --num-workers "${JUDGE_NUM_WORKERS}"
    --max-completion-tokens "${JUDGE_MAX_COMPLETION_TOKENS}"
    --output-dir "${JUDGE_DIR}"
    --override-args "${judge_override}"
)
build_gen_args "JUDGE" judge_args
evalhub gen "${judge_args[@]}"

judge_solutions="${JUDGE_DIR}/${JUDGE_TASK}.jsonl"
[[ -f "${judge_solutions}" ]] || judge_solutions="${JUDGE_DIR}/${JUDGE_TASK}_raw.jsonl"
evalhub eval \
    --tasks "${JUDGE_TASK}" \
    --solutions "${judge_solutions}" \
    --output-dir "${JUDGE_DIR}" \
    --override-args "${judge_override}"

stop_vllm

# ============================================================================
# Stage 3 — aggregate majority vote, apply CoT veto, produce summary
# ============================================================================
echo "==[3/3]== CoT-Pass@K aggregation =========================================="
evalhub cot finalize \
    --base-results "${TARGET_DIR}/${BENCHMARK}_results.jsonl" \
    --base-raw "${TARGET_DIR}/${BENCHMARK}_raw.jsonl" \
    --judge-solutions "${judge_solutions}" \
    --output-dir "${JUDGE_DIR}" \
    --benchmark "${BENCHMARK}"

echo "[DONE] CoT-Pass@K summary written under ${JUDGE_DIR}"
