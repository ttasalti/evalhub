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

    # Optional secrets file — gitignored, holds HF_TOKEN and similar. Located
    # next to the main env file (or at scripts/secrets.env). Silently skipped
    # when absent so the pipeline still runs in environments that don't need
    # HF auth (e.g. anonymous Hub access for public datasets).
    local env_dir secrets_candidates secrets_file
    env_dir="$(dirname "${env_file}")"
    secrets_candidates=(
        "${env_dir}/secrets.env"
        "${SCRIPT_DIR:-}/secrets.env"
    )
    for secrets_file in "${secrets_candidates[@]}"; do
        if [[ -n "${secrets_file}" && -f "${secrets_file}" ]]; then
            set -a
            # shellcheck disable=SC1090
            source "${secrets_file}"
            set +a
            pipeline_log "Loaded secrets file: ${secrets_file}"
            break
        fi
    done

    # Per-job override file produced by scripts/submit.sh. Sourced LAST so
    # CLI overrides win over both the main env file and secrets.env.
    if [[ -n "${EVALHUB_OVERRIDES_FILE:-}" && -f "${EVALHUB_OVERRIDES_FILE}" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "${EVALHUB_OVERRIDES_FILE}"
        set +a
        pipeline_log "Loaded CLI overrides file: ${EVALHUB_OVERRIDES_FILE}"
    fi
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
    OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULTS_ROOT_DIR:-${RESULTS_BASE_DIR:-${PROJECT_ROOT}/results}}}"
    LOG_DIR_LOCAL="${LOG_DIR:-${PROJECT_ROOT}/logs}"
    mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR_LOCAL}"
    export OUTPUT_ROOT
    export LOG_DIR_LOCAL
}

# ---------------------------------------------------------------------------
# Legacy env aliases — accept the variable names used by the old vllm3.sh
# orchestrator verbatim. New TARGET_*/JUDGE_* names take precedence when set;
# otherwise we fall back to the legacy BASE_*/PORT/TEMPERATURE/etc. spellings.
# Also maps THINK_MODE -> TARGET_STATE/JUDGE_STATE.
# ---------------------------------------------------------------------------
apply_legacy_env_aliases() {
    : "${TARGET_MODEL:=${MODEL:-}}"
    : "${TARGET_TEMPERATURE:=${TEMPERATURE:-${B_TEMP:-}}}"
    : "${TARGET_N_SAMPLES:=${N_SAMPLES:-${BASE_N_SAMPLES:-}}}"
    : "${TARGET_MAX_COMPLETION_TOKENS:=${MAX_COMPLETION_TOKENS:-${BASE_MAX_COMPLETION_TOKENS:-}}}"
    : "${TARGET_NUM_WORKERS:=${BASE_NUM_WORKERS:-}}"
    : "${TARGET_TOP_P:=${BASE_TOP_P:-}}"
    : "${TARGET_FREQUENCY_PENALTY:=${BASE_FREQUENCY_PENALTY:-}}"
    : "${TARGET_PRESENCE_PENALTY:=${BASE_PRESENCE_PENALTY:-}}"
    : "${TARGET_TIMEOUT:=${BASE_TIMEOUT:-}}"
    : "${TARGET_STOP:=${BASE_STOP:-}}"
    : "${TARGET_SYSTEM_PROMPT:=${BASE_SYSTEM_PROMPT:-}}"
    : "${TARGET_OVERRIDE_ARGS:=${BASE_OVERRIDE_ARGS:-}}"
    : "${TARGET_TOOL_CONFIG:=${BASE_TOOL_CONFIG:-}}"
    : "${TARGET_CALLBACK:=${BASE_CALLBACK:-}}"
    : "${TARGET_MAX_TURNS:=${BASE_MAX_TURNS:-}}"
    : "${TARGET_ENABLE_MULTITURN:=${BASE_ENABLE_MULTITURN:-}}"
    : "${TARGET_RESUME:=${BASE_RESUME:-}}"
    : "${TARGET_PARALLEL_COUNT:=${BASE_PARALLEL_COUNT:-}}"
    : "${TARGET_PORT:=${PORT:-}}"
    : "${JUDGE_TEMPERATURE:=${JUDGE_TEMP:-}}"

    # THINK_MODE was the old binary toggle. Honour it when TARGET_STATE is unset.
    if [[ -n "${THINK_MODE:-}" && -z "${TARGET_STATE:-}" ]]; then
        local tm
        tm="$(echo "${THINK_MODE}" | tr '[:upper:]' '[:lower:]')"
        if [[ "${tm}" == "true" ]]; then
            TARGET_STATE="think"
        else
            TARGET_STATE="non-think"
        fi
    fi

    # Soft defaults for vLLM runtime knobs that the old script exported.
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-0}"
    # vLLM 0.19 defaults V1 engine to multiprocess (forks EngineCore as a
    # separate subprocess). For TP=1 servers that doubles host RAM because
    # torch + CUDA libs + Inductor cache get loaded twice. Force in-process
    # V1 engine to match the legacy V0 single-process footprint.
    export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
    # Force Python output to be unbuffered so evalhub gen/eval failures show
    # up in Slurm .err immediately rather than getting lost when a subprocess
    # crashes before flushing its line buffer.
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
}

# ---------------------------------------------------------------------------
# Model class detection — mirrors the old get_clean_model_name(): models whose
# name contains "base", "e2b", or "e4b" are treated as base models.
# ---------------------------------------------------------------------------
detect_model_class() {
    local m_lower
    m_lower="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    if [[ "${m_lower}" == *"base"* || "${m_lower}" == *"e2b"* || "${m_lower}" == *"e4b"* ]]; then
        echo "base"
    else
        echo "instruct"
    fi
}

# V4 clean name: always just the basename. The state is encoded by the parent
# directory (results/<state>/...) so embedding "_think-<true|false>" in the
# leaf is redundant. Previous V3 emitted "${basename}_think-${flag}" for
# instruct models; V4 drops this so every (state, model) pair lives under a
# single directory regardless of model class.
target_clean_name() {
    basename "$1"
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

    # Optional vLLM runtime knobs — TARGET_* applied to the target server,
    # JUDGE_* to the judge server. Role decided by matching the port.
    local role="TARGET"
    [[ -n "${JUDGE_PORT:-}" && "${port}" == "${JUDGE_PORT}" ]] && role="JUDGE"
    local _gmu="${role}_GPU_MEMORY_UTILIZATION"
    local _mml="${role}_MAX_MODEL_LEN"
    local _ee="${role}_ENFORCE_EAGER"
    local _ss="${role}_SWAP_SPACE"
    local _dt="${role}_DTYPE"
    local _kvd="${role}_KV_CACHE_DTYPE"
    local _extra="${role}_VLLM_EXTRA_ARGS"

    # TARGET_REVISION pins the target server to a specific HF revision (e.g. a
    # mid-training checkpoint branch like "step-120"). Judge servers always
    # use their default revision — there is no JUDGE_REVISION knob.
    if [[ "${role}" == "TARGET" && -n "${TARGET_REVISION:-}" ]]; then
        pipeline_log "Pinning target model to revision: ${TARGET_REVISION}"
        args+=(--revision "${TARGET_REVISION}" --tokenizer-revision "${TARGET_REVISION}")
    fi

    [[ -n "${!_gmu:-}" ]] && args+=(--gpu-memory-utilization "${!_gmu}")
    [[ -n "${!_mml:-}" ]] && args+=(--max-model-len "${!_mml}")
    [[ -n "${!_ss:-}" ]]  && args+=(--swap-space "${!_ss}")
    [[ -n "${!_dt:-}" ]]  && args+=(--dtype "${!_dt}")
    [[ -n "${!_kvd:-}" ]] && args+=(--kv-cache-dtype "${!_kvd}")
    [[ "${!_ee:-false}" == "true" ]] && args+=(--enforce-eager)
    local _extra_val="${!_extra:-}"
    if [[ -n "${_extra_val}" ]]; then
        # shellcheck disable=SC2206
        args+=( ${_extra_val} )
    fi

    # Kill any leftover process holding the port (Slurm cgroup cleanup is not
    # always reliable on shared nodes; a previous job's vLLM may linger).
    if command -v fuser >/dev/null 2>&1; then
        fuser -k -TERM "${port}/tcp" 2>/dev/null || true
        sleep 2
        fuser -k -KILL "${port}/tcp" 2>/dev/null || true
    fi

    python -m vllm.entrypoints.openai.api_server "${args[@]}" >>"${log_file}" 2>&1 &
    SERVER_PID=$!
    local waited=0
    local timeout="${HEALTH_TIMEOUT:-1800}"
    # /health returns 200 as soon as the API server binds — model may still be
    # downloading or loading. We must also verify the model is registered (via
    # /v1/models returning non-empty) AND that a real inference call succeeds.
    until curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1 \
       && curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null \
            | grep -q '"id"'; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            {
                echo ""
                echo "=== vLLM died on port ${port} — full log dump (${log_file}) ==="
                cat "${log_file}" 2>/dev/null || true
                echo "=== end vLLM log dump ==="
            } >&2
            pipeline_die "vLLM died on port ${port}."
        fi
        sleep 5
        waited=$((waited + 5))
        if (( waited >= timeout )); then
            kill "${SERVER_PID}" 2>/dev/null || true
            pipeline_die "vLLM health timeout after ${timeout}s on port ${port}"
        fi
    done
    # Final probe: model must actually answer an inference call. Some vLLM
    # configurations register the model id before the engine is fully warm.
    local probe_model probe_status
    probe_model="$(curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null \
        | grep -o '"id"[^,]*' | head -1 | sed -E 's/.*"id"[^"]*"([^"]+)".*/\1/')"

    # Verify the served model matches what we asked vLLM to load. A stale
    # leftover process on the port could be serving a different model.
    if [[ "${probe_model}" != "${model}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        pipeline_die "vLLM port ${port} is serving '${probe_model}' but we requested '${model}' — likely a stale process. Aborting."
    fi
    local probe_waited=0
    local probe_timeout=600
    while true; do
        probe_status=$(curl -fsS -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${probe_model}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
            -o /dev/null -w "%{http_code}" 2>/dev/null || true)
        if [[ "${probe_status}" == "200" ]]; then break; fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            pipeline_die "vLLM died during inference probe on port ${port}."
        fi
        sleep 5
        probe_waited=$((probe_waited + 5))
        if (( probe_waited >= probe_timeout )); then
            kill "${SERVER_PID}" 2>/dev/null || true
            pipeline_die "vLLM inference probe timeout after ${probe_timeout}s (last status: ${probe_status})"
        fi
    done
    pipeline_log "vLLM healthy on port ${port} (pid ${SERVER_PID}, model=${probe_model})"
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
    # Final `return 0` is critical: when both bool flags are "false" (the
    # default), the last `[[ ]] && cmd` short-circuits and the function would
    # otherwise return 1, tripping `set -e` in the caller and killing the
    # pipeline silently right after `start_vllm` reported healthy.
    return 0
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
    TARGET_GPU_MEMORY_UTILIZATION="${TARGET_GPU_MEMORY_UTILIZATION:-}"
    TARGET_MAX_MODEL_LEN="${TARGET_MAX_MODEL_LEN:-}"
    TARGET_ENFORCE_EAGER="${TARGET_ENFORCE_EAGER:-false}"
    TARGET_SWAP_SPACE="${TARGET_SWAP_SPACE:-}"
    TARGET_DTYPE="${TARGET_DTYPE:-}"
    TARGET_KV_CACHE_DTYPE="${TARGET_KV_CACHE_DTYPE:-}"
    TARGET_VLLM_EXTRA_ARGS="${TARGET_VLLM_EXTRA_ARGS:-}"
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
    JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-}"
    JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-}"
    JUDGE_ENFORCE_EAGER="${JUDGE_ENFORCE_EAGER:-false}"
    JUDGE_SWAP_SPACE="${JUDGE_SWAP_SPACE:-}"
    JUDGE_DTYPE="${JUDGE_DTYPE:-}"
    JUDGE_KV_CACHE_DTYPE="${JUDGE_KV_CACHE_DTYPE:-}"
    JUDGE_VLLM_EXTRA_ARGS="${JUDGE_VLLM_EXTRA_ARGS:-}"
}

apply_common_defaults() {
    HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
}

# Compose the canonical target output directory. The V5 layout gives ONE folder
# per model: the model dir carries only "<state>/<model>", and the full sampling
# suffix is moved onto the benchmark leaf so all sampling variants of a model sit
# side-by-side under a single model dir:
#   ${OUTPUT_ROOT}/<state>/<target_clean>/<benchmark>__t<T>__max<N>__n<NS>
# Same param tuple = same path (idempotent re-run). Different tuple = different
# leaf (collision-free), but still under the same model dir.
compose_target_dir() {
    local benchmark="$1"
    local clean step_dir=""
    clean="$(target_clean_name "${TARGET_MODEL}")"
    [[ -n "${TARGET_REVISION_TAG:-}" ]] && step_dir="${TARGET_REVISION_TAG}/"
    echo "${OUTPUT_ROOT}/${TARGET_STATE}/${clean}/${step_dir}${benchmark}__t${TARGET_TEMPERATURE}__max${TARGET_MAX_COMPLETION_TOKENS}__n${TARGET_N_SAMPLES}"
}

# Compose the canonical judgment output directory. Nested under the target
# model dir (which already sits under <state>/) so drill-down is natural:
#   ${OUTPUT_ROOT}/<state>/<target_clean>/judged_by/<judge_clean>__state-<jstate>__t<jT>__max<jN>/<benchmark>__t<T>__max<N>__n<NS>
# V5: the model dir drops its sampling suffix; the benchmark leaf carries the
# TARGET's sampling (t/max/n) — identical to the base side, keeping them aligned.
# The judge leaf still carries its own state + jT/jmax (independent of the target,
# and there is no parent state dir on the judge side). JUDGE_N_SAMPLES stays out
# of the path; its actual value is recorded inside each benchmark's summary file.
compose_judge_dir() {
    local benchmark="$1"
    local target_clean judge_clean step_dir=""
    target_clean="$(target_clean_name "${TARGET_MODEL}")"
    judge_clean="$(basename "${JUDGE_MODEL}")"
    [[ -n "${TARGET_REVISION_TAG:-}" ]] && step_dir="${TARGET_REVISION_TAG}/"
    echo "${OUTPUT_ROOT}/${TARGET_STATE}/${target_clean}/${step_dir}judged_by/${judge_clean}__state-${JUDGE_STATE}__t${JUDGE_TEMPERATURE}__max${JUDGE_MAX_COMPLETION_TOKENS}/${benchmark}__t${TARGET_TEMPERATURE}__max${TARGET_MAX_COMPLETION_TOKENS}__n${TARGET_N_SAMPLES}"
}

# ---------------------------------------------------------------------------
# Stage runners. These are the highest-level building blocks; the
# orchestrator scripts simply call them in sequence.
# ---------------------------------------------------------------------------

# Run base generation + base evaluation. Assumes vLLM is already healthy and
# HOSTED_VLLM_API_BASE / _API_KEY are exported.
pipeline_run_target_gen_eval() {
    local target_dir="$1" benchmark="$2"

    if [[ -f "${target_dir}/${benchmark}_summary.json" && "${EVALHUB_OVERWRITE:-0}" != "1" ]]; then
        echo "[pipeline] skip target gen+eval: ${target_dir}/${benchmark}_summary.json exists (EVALHUB_OVERWRITE=1 to force)"
        return 0
    fi

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
    local judge_dir="$1" judge_input="$2" benchmark="$3"

    # Overwrite guard: if the upstream CoT finalize already produced the real
    # metric file for this benchmark, skip judge gen+eval entirely.
    # V5: the leaf dir name now carries the sampling suffix (<benchmark>__t..__max..__n..),
    # so it no longer equals the benchmark — the real benchmark must be passed in
    # explicitly (cot finalize writes "<benchmark>_cot_summary.json" via --benchmark).
    local _bm="${benchmark}"
    if [[ -f "${judge_dir}/${_bm}_cot_summary.json" && "${EVALHUB_OVERWRITE:-0}" != "1" ]]; then
        echo "[pipeline] skip judge gen+eval: ${judge_dir}/${_bm}_cot_summary.json exists (EVALHUB_OVERWRITE=1 to force)"
        JUDGE_SOLUTIONS_OUT="${judge_dir}/${JUDGE_TASK}.jsonl"
        return 0
    fi

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
    # judge eval emits "<JUDGE_TASK>_summary.json" — raw judge yes-rate, NOT
    # the real CoT-Pass@K. Remove it so downstream summary globs don't pick up
    # this misleading file; the real metric lives in <benchmark>_cot_summary.json
    # produced by `evalhub cot finalize`.
    rm -f "${judge_dir}/${JUDGE_TASK}_summary.json"
    # Return via global var instead of stdout capture — evalhub gen/eval write
    # GenerationConfig dumps and progress bars to stdout, which would
    # contaminate `$(...)` command substitution in the caller.
    JUDGE_SOLUTIONS_OUT="${judge_solutions}"
}

# Emit the canonical "no base-correct samples" stub. Used by both
# run_judge_only.sh and run_end_to_end.sh.
pipeline_write_empty_cot_summary() {
    local judge_dir="$1" benchmark="$2"
    mkdir -p "${judge_dir}"
    echo '{"pass_at_k": {}, "cons_at_k": 0.0, "note": "no base-correct samples"}' \
        > "${judge_dir}/${benchmark}_cot_summary.json"
}

# Aggregate everything under OUTPUT_ROOT into the master CSV and render the
# plot suite, the highlights PDF, and the plot atlas. Used by the single-run
# Stage 4 and by the BENCHMARKS-sweep tail so both produce an identical report.
pipeline_run_report() {
    mkdir -p "${OUTPUT_ROOT}/report_plots"
    evalhub report aggregate \
        --results-root "${OUTPUT_ROOT}" \
        --output "${OUTPUT_ROOT}/report.csv"
    evalhub report plot \
        --csv "${OUTPUT_ROOT}/report.csv" \
        --output-dir "${OUTPUT_ROOT}/report_plots"
    evalhub report highlights \
        --csv "${OUTPUT_ROOT}/report.csv" \
        --output "${OUTPUT_ROOT}/report_highlights.pdf"
    evalhub report atlas \
        --plot-dir "${OUTPUT_ROOT}/report_plots" \
        --output "${OUTPUT_ROOT}/report_plots_atlas.pdf"
    pipeline_log "[DONE] Report CSV + plots + highlights + atlas written under ${OUTPUT_ROOT}"
}
