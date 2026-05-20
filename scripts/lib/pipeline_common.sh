#!/usr/bin/env bash
# ============================================================================
# scripts/lib/pipeline_common.sh
#
# Shared helpers for the EvalHub orchestrator scripts:
#   * run_eval_only.sh   — base generation + base evaluation
#   * run_judge_only.sh  — CoT judge stage over an existing base run
#   * run_end_to_end.sh  — full CoT-Pass@K pipeline
#
# All three sources this file. Functions are designed to be safe to call from
# `set -euo pipefail` scripts; failures bubble up via `exit` rather than
# returning non-zero, so the caller does not need to wrap every call in `||`.
# ============================================================================

# This file is meant to be sourced, not executed.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[ERROR] pipeline_common.sh must be sourced, not executed." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Logging helpers — single timestamp format used across every orchestrator.
# ---------------------------------------------------------------------------
pipeline_log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
pipeline_die() { printf '[%s] [ERROR] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Env file loader and required-var enforcement.
# ---------------------------------------------------------------------------
pipeline_load_env() {
    local env_file="${1:-}"
    if [[ -z "${env_file}" ]]; then
        pipeline_log "No env file supplied; relying on existing environment."
        return 0
    fi
    if [[ ! -f "${env_file}" ]]; then
        pipeline_die "env file not found: ${env_file}"
    fi
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
    pipeline_log "Loaded env file: ${env_file}"
}

require_env() {
    local missing=()
    for name in "$@"; do
        if [[ -z "${!name:-}" ]]; then
            missing+=("${name}")
        fi
    done
    if (( ${#missing[@]} > 0 )); then
        pipeline_die "Missing required env vars: ${missing[*]}"
    fi
}

# ---------------------------------------------------------------------------
# Path setup. Callers must set PROJECT_ROOT before sourcing this file.
# ---------------------------------------------------------------------------
pipeline_init_paths() {
    : "${PROJECT_ROOT:?PROJECT_ROOT must be set before calling pipeline_init_paths}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results}"
    LOG_DIR_LOCAL="${LOG_DIR:-${PROJECT_ROOT}/logs}"
    mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR_LOCAL}"
    export OUTPUT_ROOT
    export LOG_DIR_LOCAL
}

# ---------------------------------------------------------------------------
# Template resolution & vLLM lifecycle. All three orchestrators share this.
# ---------------------------------------------------------------------------
resolve_template() {
    local model="$1" state="$2"
    python -m evalhub.utils.model_state --model "${model}" --state "${state}" --allow-missing
}

# Starts the vLLM OpenAI server in the background. Sets SERVER_PID in the
# caller's scope so the cleanup trap can reap it.
start_vllm() {
    local model="$1" port="$2" tp="$3" state="$4" log_file="$5"
    local template
    template="$(resolve_template "${model}" "${state}" || true)"
    local args=(--model "${model}" --port "${port}" --tensor-parallel-size "${tp}" --trust-remote-code)
    if [[ -n "${template}" ]]; then
        pipeline_log "Using chat template: ${template}"
        args+=(--chat-template "${template}")
    else
        pipeline_log "No registered template for ${model}/${state} — using tokenizer default."
    fi
    python -m vllm.entrypoints.openai.api_server "${args[@]}" >>"${log_file}" 2>&1 &
    SERVER_PID=$!
    local waited=0
    local timeout="${HEALTH_TIMEOUT:-1800}"
    until curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            pipeline_die "vLLM died on port ${port}. See ${log_file}"
        fi
        sleep 5
        waited=$((waited + 5))
        if (( waited >= timeout )); then
            kill "${SERVER_PID}" 2>/dev/null || true
            pipeline_die "vLLM health timeout after ${timeout}s on port ${port}"
        fi
    done
    pipeline_log "vLLM healthy on port ${port} (pid ${SERVER_PID})"
}

stop_vllm() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
}

pipeline_register_cleanup() {
    trap stop_vllm EXIT
}

# ---------------------------------------------------------------------------
# Generation argument builder — appends every populated optional flag to an
# array passed by name. `role` is "TARGET" or "JUDGE"; the function reads
# ${role}_TOP_P, ${role}_STOP, etc.
# ---------------------------------------------------------------------------
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
    local tool_var="${prefix}TOOL_CONFIG"
    local cb_var="${prefix}CALLBACK"
    local mt_var="${prefix}MAX_TURNS"
    local emt_var="${prefix}ENABLE_MULTITURN"
    local resume_var="${prefix}RESUME"
    local state_var="${prefix}STATE"

    [[ -n "${!top_p_var:-}" ]]              && out_array+=(--top-p "${!top_p_var}")
    [[ -n "${!fp_var:-}" ]]                 && out_array+=(--frequency-penalty "${!fp_var}")
    [[ -n "${!pp_var:-}" ]]                 && out_array+=(--presence-penalty "${!pp_var}")
    [[ -n "${!timeout_var:-}" ]]            && out_array+=(--timeout "${!timeout_var}")
    [[ -n "${!stop_var:-}" ]]               && out_array+=(--stop "${!stop_var}")
    [[ -n "${!sys_var:-}" ]]                && out_array+=(--system-prompt "${!sys_var}")
    [[ -n "${!tool_var:-}" ]]               && out_array+=(--tool-config "${!tool_var}")
    [[ -n "${!cb_var:-}" ]]                 && out_array+=(--callback "${!cb_var}")
    [[ -n "${!mt_var:-}" ]]                 && out_array+=(--max-turns "${!mt_var}")
    [[ -n "${!state_var:-}" ]]              && out_array+=(--model-state "${!state_var}")
    [[ "${!emt_var:-false}" == "true" ]]    && out_array+=(--enable-multiturn)
    [[ "${!resume_var:-false}" == "true" ]] && out_array+=(--resume)
}

# ---------------------------------------------------------------------------
# Default-population blocks. Centralising them here keeps the three scripts
# from drifting in their tuned values.
# ---------------------------------------------------------------------------
apply_target_defaults() {
    TARGET_STATE="${TARGET_STATE:-non-think}"
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
    TARGET_PORT="${TARGET_PORT:-30000}"
}

apply_judge_defaults() {
    JUDGE_STATE="${JUDGE_STATE:-think}"
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
    JUDGE_PORT="${JUDGE_PORT:-30001}"
}

apply_common_defaults() {
    HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
}

# Compose the canonical target output directory:
#   ${OUTPUT_ROOT}/<model>_state-<state>_t<T>_max<N>/<benchmark>
compose_target_dir() {
    local benchmark="$1"
    local model_clean
    model_clean="$(basename "${TARGET_MODEL}")_state-${TARGET_STATE}"
    echo "${OUTPUT_ROOT}/${model_clean}_t${TARGET_TEMPERATURE}_max${TARGET_MAX_COMPLETION_TOKENS}/${benchmark}"
}

# Compose the canonical judgment output directory:
#   ${OUTPUT_ROOT}/judgments/<target>_judged_by_<judge>_t<T>_max<N>/<benchmark>
compose_judge_dir() {
    local benchmark="$1"
    local target_clean judge_clean
    target_clean="$(basename "${TARGET_MODEL}")_state-${TARGET_STATE}"
    judge_clean="$(basename "${JUDGE_MODEL}")_state-${JUDGE_STATE}"
    echo "${OUTPUT_ROOT}/judgments/${target_clean}_judged_by_${judge_clean}_t${JUDGE_TEMPERATURE}_max${JUDGE_MAX_COMPLETION_TOKENS}/${benchmark}"
}

# ---------------------------------------------------------------------------
# Stage runners. These are the highest-level building blocks; the
# orchestrator scripts simply call them in sequence.
# ---------------------------------------------------------------------------

# Run base generation + base evaluation. Assumes vLLM is already healthy and
# HOSTED_VLLM_API_BASE / _API_KEY are exported.
pipeline_run_target_gen_eval() {
    local target_dir="$1" benchmark="$2"

    local target_args=(
        --model "hosted_vllm/${TARGET_MODEL}"
        --tasks "${benchmark}"
        --temperature "${TARGET_TEMPERATURE}"
        --n-samples "${TARGET_N_SAMPLES}"
        --num-workers "${TARGET_NUM_WORKERS}"
        --max-completion-tokens "${TARGET_MAX_COMPLETION_TOKENS}"
        --output-dir "${target_dir}"
    )
    build_gen_args "TARGET" target_args
    [[ -n "${TARGET_OVERRIDE_ARGS}" ]] && target_args+=(--override-args "${TARGET_OVERRIDE_ARGS}")
    evalhub gen "${target_args[@]}"

    local target_solutions="${target_dir}/${benchmark}.jsonl"
    [[ -f "${target_solutions}" ]] || target_solutions="${target_dir}/${benchmark}_raw.jsonl"
    local target_eval_args=(
        --tasks "${benchmark}"
        --solutions "${target_solutions}"
        --output-dir "${target_dir}"
    )
    [[ -n "${TARGET_OVERRIDE_ARGS}" ]] && target_eval_args+=(--override-args "${TARGET_OVERRIDE_ARGS}")
    evalhub eval "${target_eval_args[@]}"
}

# Run judge generation + judge evaluation. JUDGE_INPUT and JUDGE_DIR must be
# pre-populated. Assumes vLLM is already healthy and the env exports point at
# the judge port.
pipeline_run_judge_gen_eval() {
    local judge_dir="$1" judge_input="$2"

    local judge_override="{\"file_path\": \"${judge_input}\"}"
    local judge_args=(
        --model "hosted_vllm/${JUDGE_MODEL}"
        --tasks "${JUDGE_TASK}"
        --temperature "${JUDGE_TEMPERATURE}"
        --n-samples "${JUDGE_N_SAMPLES}"
        --num-workers "${JUDGE_NUM_WORKERS}"
        --max-completion-tokens "${JUDGE_MAX_COMPLETION_TOKENS}"
        --output-dir "${judge_dir}"
        --override-args "${judge_override}"
    )
    build_gen_args "JUDGE" judge_args
    evalhub gen "${judge_args[@]}"

    local judge_solutions="${judge_dir}/${JUDGE_TASK}.jsonl"
    [[ -f "${judge_solutions}" ]] || judge_solutions="${judge_dir}/${JUDGE_TASK}_raw.jsonl"
    evalhub eval \
        --tasks "${JUDGE_TASK}" \
        --solutions "${judge_solutions}" \
        --output-dir "${judge_dir}" \
        --override-args "${judge_override}"
    echo "${judge_solutions}"
}

# Emit the canonical "no base-correct samples" stub. Used by both
# run_judge_only.sh and run_end_to_end.sh.
pipeline_write_empty_cot_summary() {
    local judge_dir="$1" benchmark="$2"
    mkdir -p "${judge_dir}"
    echo '{"pass_at_k": {}, "cons_at_k": 0.0, "note": "no base-correct samples"}' \
        > "${judge_dir}/${benchmark}_cot_summary.json"
}
