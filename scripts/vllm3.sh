#!/bin/bash
# ==============================================================================
# EvalHub Master Orchestrator: Monolithic Pipeline Implementation
# ==============================================================================
set -euo pipefail

# 1. Path Independence: Establish project root dynamically
if [[ -z "${PROJECT_ROOT:-}" ]]; then
    export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_TRITON_FLASH_ATTN=0 

# Conda activation
eval "$(conda shell.bash hook 2>/dev/null || echo '')"
conda activate evalhub_env || echo "[WARNING] Conda environment failed to activate, falling back to system Python."

CONFIG_FILE="${PROJECT_ROOT}/scripts/vllm.env"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Configuration file $CONFIG_FILE not found at $PROJECT_ROOT/scripts." >&2
    exit 1
fi

# 2. Environment Variable Integration: Force export of all variables in the .env
set -a
source "$CONFIG_FILE"
set +a

export RESULTS_BASE_DIR="${RESULTS_ROOT_DIR:-${PROJECT_ROOT}/results}"
export FILTERED_BASE_DIR="${FILTERED_DATA_ROOT_DIR:-${PROJECT_ROOT}/data/passatk_filtered}"

export HF_TOKEN="${HF_TOKEN:-}"
RUN_MODE="${RUN_MODE:-orchestrator}"
BASE_PORT="${PORT:-30000}"
export THINK_MODE="${THINK_MODE:-false}"
export WAIT_FOR_JOB_ID="${WAIT_FOR_JOB_ID:-}"

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

get_clean_model_name() {
    local model_path="$1"
    local model_lower=$(echo "$model_path" | tr '[:upper:]' '[:lower:]')
    local base_name=$(basename "$model_path")

    # Base model tespiti: Adında "base", "e2b" veya "e4b" geçenler Base kabul edilir.
    if [[ "$model_lower" == *"base"* ]] || [[ "$model_lower" == *"e2b"* ]] || [[ "$model_lower" == *"e4b"* ]]; then
        echo "${base_name}"
    else
        # Instruct/Judge modeller için think modunu isme ekle
        echo "${base_name}_think-${THINK_MODE}"
    fi
}

start_server_and_wait() {
    local model_path="$1"
    local port="$2"
    local p_count="$3"
    
    mkdir -p logs
    local log_file="logs/vllm_server_error_${port}.log"
    touch "$log_file"
    
    echo "[INFO] Initializing vLLM server for model: $model_path on port $port"
    echo "[INFO] Hardware Strategy: --tensor-parallel-size $p_count"
    echo "[INFO] Think Mode is set to: $THINK_MODE"

    local vllm_args=(
        --model "$model_path"
        --port "$port"
        --tensor-parallel-size "$p_count"
        --trust-remote-code
    )

    local template_dir="${PROJECT_ROOT}/scripts/templates"
    local template_file=""
    local model_lower=$(echo "$model_path" | tr '[:upper:]' '[:lower:]')
    local think_mode_lower=$(echo "$THINK_MODE" | tr '[:upper:]' '[:lower:]')
    
    # Modele göre Base olup olmadığını ortak değişkene atayalım
    local is_base=false
    if [[ "$model_lower" == *"base"* ]] || [[ "$model_lower" == *"e2b"* ]] || [[ "$model_lower" == *"e4b"* ]]; then
        is_base=true
    fi

    if [[ "$model_lower" == *"qwen"* ]] && [[ "$is_base" == true ]]; then
        template_file="${template_dir}/qwen3.5-base.jinja"
    elif [[ "$model_lower" == *"gemma"* ]] && [[ "$is_base" == true ]]; then
        template_file="${template_dir}/gemma4-base.jinja"
    elif [[ "$model_lower" == *"ministral"* ]] && [[ "$is_base" == true ]]; then
        template_file="${template_dir}/ministral3-base.jinja"
    elif [[ "$model_lower" == *"ministral"* ]] && [[ "$model_lower" == *"instruct"* ]]; then
        template_file="${template_dir}/ministral3-instruct.jinja"
    elif [[ "$model_lower" == *"ministral"* ]] && [[ "$model_lower" == *"reasoning"* ]]; then
        template_file="${template_dir}/ministral3-reasoning.jinja"
    elif [[ "$model_lower" == *"qwen"* ]]; then
        if [[ "$think_mode_lower" == "true" ]]; then
            template_file="${template_dir}/qwen3.5-think.jinja"
        else
            template_file="${template_dir}/qwen3.5-no-think.jinja"
        fi
    elif [[ "$model_lower" == *"gemma"* ]]; then
        if [[ "$think_mode_lower" == "true" ]]; then
            template_file="${template_dir}/gemma4-think.jinja"
        else
            template_file="${template_dir}/gemma4-no-think.jinja"
        fi
    else
        echo "[ERROR] Desteklenmeyen/Bilinmeyen model tespit edildi: $model_path" >&2
        exit 1
    fi

    if [[ ! -f "$template_file" ]]; then
        echo "[ERROR] Template dosyası diskte bulunamadı: $template_file" >&2
        exit 1
    fi

    echo "[INFO] Model için belirlenen chat template yükleniyor: $template_file"
    vllm_args+=( --chat-template "$template_file" )

    python -m vllm.entrypoints.openai.api_server "${vllm_args[@]}" >> "$log_file" 2>&1 &
    
    SERVER_PID=$!
    local retries=0
    local max_retries=360
    
    echo "[INFO] Waiting for server healthcheck... (Logs are in $log_file)"
    while ! curl -s "http://127.0.0.1:$port/health" > /dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[ERROR] vLLM server process ($SERVER_PID) died unexpectedly." >&2
            cat "$log_file" >&2
            exit 1
        fi
        sleep 5
        retries=$((retries + 1))
        if [ "$retries" -ge "$max_retries" ]; then
            echo "[ERROR] Server failed health check timeout on port $port." >&2
            kill "$SERVER_PID" 2>/dev/null || true
            exit 1
        fi
    done
    echo "[INFO] Server is healthy and accepting requests."
}

cleanup_server_and_cache() {
    echo "[INFO] Cleaning up server (PID: ${SERVER_PID:-}) and VRAM cache..."
    if [[ -n "${SERVER_PID:-}" ]]; then
        # 1. Ana vLLM sürecini ve TÜM alt/çocuk süreçlerini acımasızca anında öldür
        pkill -P "$SERVER_PID" -9 2>/dev/null || true
        kill -9 "$SERVER_PID" 2>/dev/null || true
        SERVER_PID="" 
    fi

    # 2. VRAM'i zorla boşaltmak için Ray/PyTorch Python worker'larını temizle 
    # (Önceki modelden kalan hiçbir şeyin VRAM'i bloke etmediğinden emin ol)
    pkill -9 -f "vllm.engine.arg_utils" 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true

    if [[ -n "${EVALHUB_CACHE_DIR:-}" ]]; then
        rm -rf "$EVALHUB_CACHE_DIR"
    fi
    
    # 3. İşletim sistemine RAM/VRAM'i toparlaması için 5 saniye nefes payı ver
    sleep 5
}

# ------------------------------------------------------------------------------
# Workers
# ------------------------------------------------------------------------------
run_eval_worker() {
    echo "[INFO] --- Starting Base Evaluation ---"
    
    local worker_port=$((BASE_PORT + RANDOM % 10000))
    export EVALHUB_CACHE_DIR="${PROJECT_ROOT}/data/cache/eval_${RANDOM}"
    mkdir -p "$EVALHUB_CACHE_DIR" "${PROJECT_ROOT}/logs"
    
    trap cleanup_server_and_cache EXIT

    start_server_and_wait "$TARGET_MODEL" "$worker_port" "$BASE_PARALLEL_COUNT"
    
    export HOSTED_VLLM_API_BASE="http://127.0.0.1:$worker_port/v1"
    export HOSTED_VLLM_API_KEY="EMPTY"

    local clean_model=$(get_clean_model_name "$TARGET_MODEL")
    local out_dir="${RESULTS_BASE_DIR}/${clean_model}_t${TEMPERATURE}_max${BASE_MAX_COMPLETION_TOKENS}/${BENCHMARK}"
    mkdir -p "$out_dir"
    
    local dynamic_workers=192
    local model_lower=$(echo "$TARGET_MODEL" | tr '[:upper:]' '[:lower:]')
    
    if [[ "$model_lower" == *"0.8b"* ]]; then dynamic_workers=512;
    elif [[ "$model_lower" == *"e2b"* ]] || [[ "$model_lower" == *"2b"* ]] || [[ "$model_lower" == *"3b"* ]]; then dynamic_workers=384;
    elif [[ "$model_lower" == *"e4b"* ]] || [[ "$model_lower" == *"4b"* ]] || [[ "$model_lower" == *"8b"* ]] || [[ "$model_lower" == *"9b"* ]]; then dynamic_workers=256;
    elif [[ "$model_lower" == *"14b"* ]] || [[ "$model_lower" == *"35b"* ]]; then dynamic_workers=192; fi

    local final_num_workers="${BASE_NUM_WORKERS:-$dynamic_workers}"
    echo "[INFO] Base Worker Allocation -> Using: $final_num_workers (User Config: ${BASE_NUM_WORKERS:-empty}, H200 Auto: $dynamic_workers)"

    local cmd_args=(
        --model "hosted_vllm/$TARGET_MODEL"
        --tasks "$BENCHMARK"
        --temperature "$TEMPERATURE"
        --n-samples "$N_SAMPLES"
        --num-workers "$final_num_workers"
        --max-completion-tokens "$MAX_COMPLETION_TOKENS"
        --output-dir "$out_dir"
    )

    # ENV Parameters Injection
    [[ -n "${BASE_TOP_P:-}" ]] && cmd_args+=(--top-p "$BASE_TOP_P")
    [[ -n "${BASE_FREQUENCY_PENALTY:-}" ]] && cmd_args+=(--frequency-penalty "$BASE_FREQUENCY_PENALTY")
    [[ -n "${BASE_PRESENCE_PENALTY:-}" ]] && cmd_args+=(--presence-penalty "$BASE_PRESENCE_PENALTY")
    [[ -n "${BASE_TIMEOUT:-}" ]] && cmd_args+=(--timeout "$BASE_TIMEOUT")
    [[ -n "${BASE_STOP:-}" ]] && cmd_args+=(--stop "$BASE_STOP")
    [[ -n "${BASE_SYSTEM_PROMPT:-}" ]] && cmd_args+=(--system-prompt "$BASE_SYSTEM_PROMPT")
    [[ -n "${BASE_OVERRIDE_ARGS:-}" ]] && cmd_args+=(--override-args "$BASE_OVERRIDE_ARGS")
    [[ -n "${BASE_MAX_TURNS:-}" ]] && cmd_args+=(--max-turns "$BASE_MAX_TURNS")
    [[ -n "${BASE_TOOL_CONFIG:-}" ]] && cmd_args+=(--tool-config "$BASE_TOOL_CONFIG")
    [[ -n "${BASE_CALLBACK:-}" ]] && cmd_args+=(--callback "$BASE_CALLBACK")

    [[ "${BASE_ENABLE_MULTITURN:-false}" == "true" ]] && cmd_args+=(--enable-multiturn)
    [[ "${BASE_RESUME:-false}" == "true" ]] && cmd_args+=(--resume)
    
    echo "[INFO] Executing generation phase with full parameters..."
    python -m evalhub.cli gen "${cmd_args[@]}"

    local solutions_file="$out_dir/${BENCHMARK}.jsonl"
    [[ ! -f "$solutions_file" ]] && solutions_file="$out_dir/${BENCHMARK}_raw.jsonl"

    echo "[INFO] Executing evaluation phase..."
    python -m evalhub.cli eval \
        --tasks "$BENCHMARK" \
        --solutions "$solutions_file" \
        --output-dir "$out_dir"
}

run_extract_worker() {
    echo "[INFO] --- Running Pass@K Extraction ---"
    
    local clean_model=$(get_clean_model_name "$TARGET_MODEL")
    local filtered_dir="${FILTERED_BASE_DIR}/${clean_model}"
    mkdir -p "$filtered_dir" "${PROJECT_ROOT}/logs"
    
    local out_dir="${RESULTS_BASE_DIR}/${clean_model}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}/${BENCHMARK}"    
    local results_file="${out_dir}/${BENCHMARK}_results.jsonl"
    local raw_file="${out_dir}/${BENCHMARK}_raw.jsonl"
    local output_filtered="${filtered_dir}/${BENCHMARK}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}_corrects.jsonl"    
    
    if [[ ! -f "$results_file" && ! -f "$raw_file" ]]; then
        echo "[ERROR] Base evaluation dosyaları bulunamadı. Önceki adım patlamış olabilir."
        echo "[INFO] Dependency zincirini kırmamak için boş dosya oluşturulup graceful exit yapılıyor."
        touch "$output_filtered"
        return 0
    fi

    python "${PROJECT_ROOT}/scripts/cot_judge_pipeline/01_extract_corrects.py" \
        --results_file "$results_file" \
        --raw_file "$raw_file" \
        --output_file "$output_filtered" || {
        echo "[WARN] Extraction betiği 0 correct (pass@k=0) nedeniyle hata döndürmüş olabilir."
    }
    
    touch "$output_filtered"
}

run_judge_worker() {
    echo "[INFO] --- Starting CoT Judging for Model: $JUDGE_MODEL ---"
    
    local clean_target=$(get_clean_model_name "$TARGET_MODEL")
    local clean_judge=$(basename "$JUDGE_MODEL")
    local out_dir="${RESULTS_BASE_DIR}/judgments/${clean_target}_evaluated_by_${clean_judge}_${JUDGE_MAX_COMPLETION_TOKENS}/${BENCHMARK}_t${JUDGE_TEMP}"   
    mkdir -p "$out_dir"

    if [[ -z "${FILTERED_FILE:-}" ]]; then
        FILTERED_FILE="${FILTERED_BASE_DIR}/${clean_target}/${BENCHMARK}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}_corrects.jsonl"
    fi

    if [[ ! -s "$FILTERED_FILE" ]]; then
        echo "[INFO] Filtrelenmiş dosya boş veya eksik (pass@k=0). Judge worker atlanıyor."
        touch "$out_dir/math_judge.jsonl"
        return 0
    fi
    
    export THINK_MODE="true"
    local worker_port=$((BASE_PORT + 10000 + RANDOM % 10000))
    export EVALHUB_CACHE_DIR="${PROJECT_ROOT}/data/cache/judge_${RANDOM}"
    mkdir -p "$EVALHUB_CACHE_DIR" "${PROJECT_ROOT}/logs"
    
    start_server_and_wait "$JUDGE_MODEL" "$worker_port" "$JUDGE_PARALLEL_COUNT"

    export HOSTED_VLLM_API_BASE="http://127.0.0.1:$worker_port/v1"
    export HOSTED_VLLM_API_KEY="EMPTY"

    local final_override="{\"file_path\": \"$FILTERED_FILE\"}"
    
    local dynamic_workers=192
    local model_lower=$(echo "$JUDGE_MODEL" | tr '[:upper:]' '[:lower:]')
    
    if [[ "$model_lower" == *"0.8b"* ]]; then dynamic_workers=512;
    elif [[ "$model_lower" == *"e2b"* ]] || [[ "$model_lower" == *"2b"* ]] || [[ "$model_lower" == *"3b"* ]]; then dynamic_workers=384;
    elif [[ "$model_lower" == *"e4b"* ]] || [[ "$model_lower" == *"4b"* ]] || [[ "$model_lower" == *"8b"* ]] || [[ "$model_lower" == *"9b"* ]]; then dynamic_workers=256;
    elif [[ "$model_lower" == *"14b"* ]] || [[ "$model_lower" == *"35b"* ]]; then dynamic_workers=192; fi

    local final_judge_workers="${JUDGE_NUM_WORKERS:-$dynamic_workers}"
    echo "[INFO] Judge Worker Allocation -> Using: $final_judge_workers"

    local cmd_args=(
        --model "hosted_vllm/$JUDGE_MODEL"
        --tasks "math_judge"
        --temperature "$JUDGE_TEMP"
        --n-samples "$JUDGE_N_SAMPLES"
        --num-workers "$final_judge_workers"
        --max-completion-tokens "$JUDGE_MAX_COMPLETION_TOKENS"
        --output-dir "$out_dir"
        --override-args "$final_override"
    )

    # ENV Parameters Injection for Judge
    [[ -n "${JUDGE_TOP_P:-}" ]] && cmd_args+=(--top-p "$JUDGE_TOP_P")
    [[ -n "${JUDGE_FREQUENCY_PENALTY:-}" ]] && cmd_args+=(--frequency-penalty "$JUDGE_FREQUENCY_PENALTY")
    [[ -n "${JUDGE_PRESENCE_PENALTY:-}" ]] && cmd_args+=(--presence-penalty "$JUDGE_PRESENCE_PENALTY")
    [[ -n "${JUDGE_TIMEOUT:-}" ]] && cmd_args+=(--timeout "$JUDGE_TIMEOUT")
    [[ -n "${JUDGE_STOP:-}" ]] && cmd_args+=(--stop "$JUDGE_STOP")
    [[ -n "${JUDGE_SYSTEM_PROMPT:-}" ]] && cmd_args+=(--system-prompt "$JUDGE_SYSTEM_PROMPT")
    [[ -n "${JUDGE_MAX_TURNS:-}" ]] && cmd_args+=(--max-turns "$JUDGE_MAX_TURNS")
    [[ -n "${JUDGE_TOOL_CONFIG:-}" ]] && cmd_args+=(--tool-config "$JUDGE_TOOL_CONFIG")
    [[ -n "${JUDGE_CALLBACK:-}" ]] && cmd_args+=(--callback "$JUDGE_CALLBACK")

    [[ "${JUDGE_ENABLE_MULTITURN:-false}" == "true" ]] && cmd_args+=(--enable-multiturn)
    [[ "${JUDGE_RESUME:-false}" == "true" ]] && cmd_args+=(--resume)

    echo "[INFO] Executing judgement generation phase..."
    python -m evalhub.cli gen "${cmd_args[@]}"

    local judge_solutions_file="$out_dir/math_judge.jsonl"
    [[ ! -f "$judge_solutions_file" ]] && judge_solutions_file="$out_dir/math_judge_raw.jsonl"

    python -m evalhub.cli eval \
        --tasks "math_judge" \
        --solutions "$judge_solutions_file" \
        --output-dir "$out_dir" \
        --override-args "$final_override"

    cleanup_server_and_cache
}

run_post_worker() {
    echo "[INFO] --- Post-Processing Judge Outputs ---"
    
    local clean_target=$(get_clean_model_name "$TARGET_MODEL")
    local clean_judge=$(basename "$JUDGE_MODEL")
    
    local out_dir="${RESULTS_BASE_DIR}/judgments/${clean_target}_evaluated_by_${clean_judge}_${JUDGE_MAX_COMPLETION_TOKENS}/${BENCHMARK}_t${JUDGE_TEMP}"
    rm -f "${out_dir}/math_judge_summary.json"

    local eval_judge_output="${out_dir}/math_judge.jsonl"
    [[ ! -f "$eval_judge_output" ]] && eval_judge_output="${out_dir}/math_judge_raw.jsonl"

    local majority_file="${out_dir}/math_judge_majority.jsonl"
    local base_results_path="${RESULTS_BASE_DIR}/${clean_target}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}/${BENCHMARK}"
    local base_results_file="${base_results_path}/${BENCHMARK}_results.jsonl"

    local judged_results_file="${out_dir}/${BENCHMARK}_results.jsonl"
    local summary_file="${out_dir}/${BENCHMARK}_summary.json"
    local stats_file="${out_dir}/${BENCHMARK}_generation_stats.json"

    if [[ ! -s "$eval_judge_output" ]]; then
        echo "[INFO] Judge output boş (muhtemelen pass@k=0). Post-processing atlanıyor."
        touch "$majority_file" "$judged_results_file"
        echo '{"pass_at_k": 0.0, "note": "Skipped due to 0 base corrects"}' > "$summary_file"
        echo '{"note": "Skipped due to 0 base corrects"}' > "$stats_file"
        return 0
    fi

    python "${PROJECT_ROOT}/scripts/cot_judge_pipeline/04_aggregate_votes.py" --input_file "$eval_judge_output" --output_file "$majority_file"
    python "${PROJECT_ROOT}/scripts/cot_judge_pipeline/05_apply_metrics.py" --base_results_file "$base_results_file" --judge_majority_file "$majority_file" --output_file "$judged_results_file" --summary_file "$summary_file" --stats_file "$stats_file"
}

run_extract_judge_post_worker() {
    echo "[INFO] --- Starting Combined Pipeline (Extract -> Judge -> Post) ---"
    trap cleanup_server_and_cache EXIT

    run_extract_worker

    local clean_target=$(get_clean_model_name "$TARGET_MODEL")
    local filtered_file="${FILTERED_BASE_DIR}/${clean_target}/${BENCHMARK}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}_corrects.jsonl"
    export FILTERED_FILE="$filtered_file"

    if [[ -z "${JUDGE_MODELS:-}" ]]; then
        echo "[INFO] No JUDGE_MODELS defined. Completing job after extraction."
        return 0
    fi

    for CURRENT_JUDGE in $JUDGE_MODELS; do
        for CURRENT_J_TEMP in ${JUDGE_TEMPERATURES:-$JUDGE_TEMP}; do
            echo "[INFO] ========================================================"
            echo "[INFO] Running Judge Pipeline -> Model: $CURRENT_JUDGE | Temp: $CURRENT_J_TEMP"
            echo "[INFO] ========================================================"
            
            export JUDGE_MODEL="$CURRENT_JUDGE"
            export JUDGE_TEMP="$CURRENT_J_TEMP"
            export JUDGE_N_SAMPLES="${JUDGE_N_SAMPLES:-1}"
            export JUDGE_MAX_COMPLETION_TOKENS="${JUDGE_MAX_COMPLETION_TOKENS:-16384}"
            
            run_judge_worker
            run_post_worker
        done
    done
    
    echo "[INFO] Combined Pipeline Completed Successfully."
}

# ------------------------------------------------------------------------------
# Orchestrator Logic
# ------------------------------------------------------------------------------
run_orchestrator() {
    echo "[INFO] Initializing Unified Pipeline Orchestrator (Monolithic Mode)"

    mkdir -p "${PROJECT_ROOT}/logs" "${RESULTS_BASE_DIR}/judgments" "${FILTERED_BASE_DIR}" "${PROJECT_ROOT}/plots" "${PROJECT_ROOT}/data/cache"

    local script_path=$(realpath "$0")
    local global_last_job_id="${WAIT_FOR_JOB_ID:-}"
    
    if [[ -n "$global_last_job_id" ]]; then
        echo "[INFO] External dependency detected! Pipeline will wait for Job ID: $global_last_job_id to finish (afterany)."
    fi

    for MODEL in $BASE_MODELS; do
        local clean_model=$(get_clean_model_name "$MODEL")
        for BENCHMARK in $BENCHMARKS; do
            for B_TEMP in $BASE_TEMPERATURES; do
                
                local run_id="${clean_model}_${BENCHMARK}_t${B_TEMP}_max${BASE_MAX_COMPLETION_TOKENS}"
                echo "[INFO] Submitting DAG for Combination: $run_id"

                local global_dep=""
                [[ -n "$global_last_job_id" ]] && global_dep="--dependency=afterany:${global_last_job_id}"

                local nodelist_param=""
                [[ -n "${BASE_SLURM_NODELIST:-}" ]] && nodelist_param="--nodelist=$BASE_SLURM_NODELIST"

                # 1. Aşama: Base Evaluation (vLLM Server + Generation)
                local base_job=$(sbatch --parsable $global_dep \
                    --job-name="eval_${run_id}" \
                    --output="${PROJECT_ROOT}/logs/eval_${run_id}_%j.out" \
                    --error="${PROJECT_ROOT}/logs/eval_${run_id}_%j.err" \
                    --ntasks=1 --cpus-per-task="${BASE_SLURM_CPUS}" --mem="${BASE_SLURM_MEM}" \
                    --gres="${BASE_SLURM_GRES}" --time="${BASE_SLURM_TIME}" \
                    $nodelist_param \
                    --export=ALL,RUN_MODE="eval_worker",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",N_SAMPLES="$BASE_N_SAMPLES",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS",HF_TOKEN="${HF_TOKEN}",PROJECT_ROOT="${PROJECT_ROOT}",THINK_MODE="${THINK_MODE}" \
                    "$script_path")

                # 2. Aşama: Combined (Extract + Judge + Post) 
                if [[ -z "${JUDGE_MODELS:-}" ]]; then
                    local combined_job=$(sbatch --parsable --dependency=afterany:"${base_job}" \
                        --job-name="extr_${run_id}" \
                        --output="${PROJECT_ROOT}/logs/extr_${run_id}_%j.out" \
                        --ntasks=1 --cpus-per-task="${EXTRACT_SLURM_CPUS}" --mem="${EXTRACT_SLURM_MEM}" \
                        --time="${EXTRACT_SLURM_TIME}" \
                        --export=ALL,RUN_MODE="extract_judge_post_worker",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS",PROJECT_ROOT="${PROJECT_ROOT}",THINK_MODE="${THINK_MODE}" \
                        "$script_path")
                    global_last_job_id="${combined_job}"
                else
                    local judge_nodelist=""
                    [[ -n "${JUDGE_SLURM_NODELIST:-}" ]] && judge_nodelist="--nodelist=$JUDGE_SLURM_NODELIST"

                    local combined_job=$(sbatch --parsable --dependency=afterany:"${base_job}" \
                        --job-name="pipe_${run_id}" \
                        --output="${PROJECT_ROOT}/logs/pipe_${run_id}_%j.out" \
                        --error="${PROJECT_ROOT}/logs/pipe_${run_id}_%j.err" \
                        --ntasks=1 --cpus-per-task="${JUDGE_SLURM_CPUS}" --mem="${JUDGE_SLURM_MEM}" \
                        --gres="${JUDGE_SLURM_GRES}" --time="${JUDGE_SLURM_TIME}" \
                        $judge_nodelist \
                        --export=ALL,RUN_MODE="extract_judge_post_worker",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS",HF_TOKEN="${HF_TOKEN}",PROJECT_ROOT="${PROJECT_ROOT}",THINK_MODE="${THINK_MODE}" \
                        "$script_path")
                    global_last_job_id="${combined_job}"
                fi
                
                echo "[INFO] DAG Submitted. Tail Job ID: $global_last_job_id"
            done
        done
    done
}

case "$RUN_MODE" in
    orchestrator)              run_orchestrator ;;
    eval_worker)               run_eval_worker ;;
    extract_worker)            run_extract_worker ;;
    judge_worker)              run_judge_worker ;;
    post_worker)               run_post_worker ;;
    extract_judge_post_worker) run_extract_judge_post_worker ;;
    *)                         echo "[ERROR] Invalid RUN_MODE: $RUN_MODE" >&2; exit 1 ;;
esac

exit 0