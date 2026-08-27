#!/bin/bash
# ==============================================================================
# EvalHub Master Orchestrator: Monolithic Pipeline Implementation (Gemma 4 / vLLM)
# ==============================================================================
set -euo pipefail

# 1. Path Independence: Establish project root dynamically
if [[ -z "${PROJECT_ROOT:-}" ]]; then
    export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Triton Fallback: Disable custom triton kernels if SparseMatrix issues persist on H200
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

# Ensure HF_TOKEN is exposed to vLLM
export HF_TOKEN="${HF_TOKEN:-}"

RUN_MODE="${RUN_MODE:-orchestrator}"
BASE_PORT="${PORT:-30000}"

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
start_server_and_wait() {
    local model_path="$1"
    local port="$2"
    local p_count="$3"
    
    mkdir -p logs
    local log_file="logs/vllm_server_error_${port}.log"
    touch "$log_file" # Ensure the log file is created
    
    echo "[INFO] Initializing vLLM server for model: $model_path on port $port"
    echo "[INFO] Hardware Strategy: --tensor-parallel-size $p_count"

    # 1. Store basic vLLM arguments
    local vllm_args=(
        --model "$model_path"
        --port "$port"
        --tensor-parallel-size "$p_count"
        --trust-remote-code
    )

    # 2. Hardcoded Template Selection
    local template_dir="${PROJECT_ROOT}/scripts/templates"
    local template_file=""

    # Case-insensitive eşleşme için model adını küçült
    local model_lower=$(echo "$model_path" | tr '[:upper:]' '[:lower:]')
    
    if [[ "$model_lower" == *"gemma-4"* ]]; then
        template_file="${template_dir}/gemma4.jinja"
    elif [[ "$model_lower" == *"ministral"* ]]; then
        template_file="${template_dir}/ministral3.jinja"
    elif [[ "$model_lower" == *"qwen"* ]]; then
        template_file="${template_dir}/qwen3.5.jinja"
    else
        echo "[ERROR] Desteklenmeyen/Bilinmeyen model tespit edildi: $model_path" >&2
        echo "[ERROR] Lütfen '$template_dir' dizinine bu model için uygun bir Jinja template'i ekleyin ve 'scripts/vllm.sh' içindeki if-else bloğunu güncelleyin." >&2
        exit 1
    fi

    # Seçilen template'in diskte gerçekten var olup olmadığını kontrol et
    if [[ ! -f "$template_file" ]]; then
        echo "[ERROR] Template dosyası diskte bulunamadı: $template_file" >&2
        echo "[ERROR] Lütfen ilgili jinja dosyasını belirtilen path'e oluşturduğunuzdan emin olun." >&2
        exit 1
    fi

    echo "[INFO] Model için belirlenen chat template yükleniyor: $template_file"
    vllm_args+=( --chat-template "$template_file" )

    # 3. Pass arguments safely to vLLM
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
            tail -n 100 "$log_file" >&2
            kill "$SERVER_PID" 2>/dev/null || true
            exit 1
        fi
    done
    echo "[INFO] Server is healthy and accepting requests."
}

cleanup_server_and_cache() {
    echo "[INFO] Cleaning up server (PID: ${SERVER_PID:-}) and cache directory..."
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ -n "${EVALHUB_CACHE_DIR:-}" ]]; then
        rm -rf "$EVALHUB_CACHE_DIR"
    fi
}

# ------------------------------------------------------------------------------
# Workers
# ------------------------------------------------------------------------------
run_eval_worker() {
    echo "[INFO] --- Starting Base Evaluation ---"
    
    # 3. Reconciled Port Logic: Based on configuration PORT
    local worker_port=$((BASE_PORT + RANDOM % 10000))
    export EVALHUB_CACHE_DIR="${PROJECT_ROOT}/data/cache/eval_${RANDOM}"
    mkdir -p "$EVALHUB_CACHE_DIR" "${PROJECT_ROOT}/logs"
    
    trap cleanup_server_and_cache EXIT

    start_server_and_wait "$TARGET_MODEL" "$worker_port" "$BASE_PARALLEL_COUNT"
    
    export HOSTED_VLLM_API_BASE="http://127.0.0.1:$worker_port/v1"
    export HOSTED_VLLM_API_KEY="EMPTY"

    local clean_model=$(basename "$TARGET_MODEL")
    local out_dir="${PROJECT_ROOT}/results/${clean_model}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}/${BENCHMARK}"
    mkdir -p "$out_dir"
    
    local cmd_args=(
        --model "hosted_vllm/$TARGET_MODEL"
        --tasks "$BENCHMARK"
        --temperature "$TEMPERATURE"
        --n-samples "$N_SAMPLES"
        --num-workers "$BASE_NUM_WORKERS"
        --max-completion-tokens "$MAX_COMPLETION_TOKENS"
        --output-dir "$out_dir"
    )

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
    local clean_model=$(basename "$TARGET_MODEL")
    local filtered_dir="${PROJECT_ROOT}/data/passatk_filtered/${clean_model}"
    mkdir -p "$filtered_dir" "${PROJECT_ROOT}/logs"
    
    local out_dir="${PROJECT_ROOT}/results/${clean_model}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}/${BENCHMARK}"    
    local results_file="${out_dir}/${BENCHMARK}_results.jsonl"
    local raw_file="${out_dir}/${BENCHMARK}_raw.jsonl"
    local output_filtered="${filtered_dir}/${BENCHMARK}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}_corrects.jsonl"    
    
    # === DEPENDENCY FALLBACK CHECK ===
    if [[ ! -f "$results_file" && ! -f "$raw_file" ]]; then
        echo "[ERROR] Base evaluation dosyaları bulunamadı. Önceki adım (base_job) patlamış olabilir."
        echo "[INFO] Dependency zincirini kırmamak için boş dosya oluşturulup graceful exit yapılıyor."
        touch "$output_filtered"
        exit 0
    fi

    python "${PROJECT_ROOT}/scripts/cot_judge_pipeline/01_extract_corrects.py" \
        --results_file "$results_file" \
        --raw_file "$raw_file" \
        --output_file "$output_filtered" || {
        echo "[WARN] Extraction betiği 0 correct (pass@k=0) nedeniyle hata döndürmüş olabilir. Dependency kırılmaması için devam ediliyor."
    }
    
    touch "$output_filtered"
}

run_judge_worker() {
    echo "[INFO] --- Starting CoT Judging ---"
    
    local clean_target=$(basename "$TARGET_MODEL")
    local clean_judge=$(basename "$JUDGE_MODEL")
    local out_dir="${PROJECT_ROOT}/results/judgments/${clean_target}evaluated_by${clean_judge}_${JUDGE_MAX_COMPLETION_TOKENS}/${BENCHMARK}_t${JUDGE_TEMP}"   
    mkdir -p "$out_dir"

    if [[ -z "${FILTERED_FILE:-}" ]]; then
        FILTERED_FILE="${PROJECT_ROOT}/data/passatk_filtered/${clean_target}/${BENCHMARK}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}_corrects.jsonl"
    fi

    # === PASS@K 0 CHECK ===
    if [[ ! -s "$FILTERED_FILE" ]]; then
        echo "[INFO] Filtrelenmiş dosya boş veya eksik (pass@k=0). Judge worker atlanıyor ve temiz çıkış yapılıyor."
        touch "$out_dir/math_judge.jsonl"
        exit 0
    fi

    local worker_port=$((BASE_PORT + 10000 + RANDOM % 10000))
    export EVALHUB_CACHE_DIR="${PROJECT_ROOT}/data/cache/judge_${RANDOM}"
    mkdir -p "$EVALHUB_CACHE_DIR" "${PROJECT_ROOT}/logs"

    trap cleanup_server_and_cache EXIT

    start_server_and_wait "$JUDGE_MODEL" "$worker_port" "$JUDGE_PARALLEL_COUNT"

    export HOSTED_VLLM_API_BASE="http://127.0.0.1:$worker_port/v1"
    export HOSTED_VLLM_API_KEY="EMPTY"

    local final_override="{\"file_path\": \"$FILTERED_FILE\"}"
    
    local cmd_args=(
        --model "hosted_vllm/$JUDGE_MODEL"
        --tasks "math_judge"
        --temperature "$JUDGE_TEMP"
        --n-samples "$JUDGE_N_SAMPLES"
        --num-workers "$JUDGE_NUM_WORKERS"
        --max-completion-tokens "$JUDGE_MAX_COMPLETION_TOKENS"
        --output-dir "$out_dir"
        --override-args "$final_override"
    )

    echo "[INFO] Executing judgement generation phase..."
    python -m evalhub.cli gen "${cmd_args[@]}"

    local judge_solutions_file="$out_dir/math_judge.jsonl"
    [[ ! -f "$judge_solutions_file" ]] && judge_solutions_file="$out_dir/math_judge_raw.jsonl"

    python -m evalhub.cli eval \
        --tasks "math_judge" \
        --solutions "$judge_solutions_file" \
        --output-dir "$out_dir" \
        --override-args "$final_override"
}

run_post_worker() {
    echo "[INFO] --- Post-Processing Judge Outputs ---"
    local clean_target=$(basename "$TARGET_MODEL")
    local clean_judge=$(basename "$JUDGE_MODEL")
    
    local out_dir="${PROJECT_ROOT}/results/judgments/${clean_target}evaluated_by${clean_judge}_${JUDGE_MAX_COMPLETION_TOKENS}/${BENCHMARK}_t${JUDGE_TEMP}"
    rm -f "${out_dir}/math_judge_summary.json"

    local eval_judge_output="${out_dir}/math_judge.jsonl"
    [[ ! -f "$eval_judge_output" ]] && eval_judge_output="${out_dir}/math_judge_raw.jsonl"

    local majority_file="${out_dir}/math_judge_majority.jsonl"
    local base_results_path="${PROJECT_ROOT}/results/${clean_target}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}/${BENCHMARK}"
    local base_results_file="${base_results_path}/${BENCHMARK}_results.jsonl"

    local judged_results_file="${out_dir}/${BENCHMARK}_results.jsonl"
    local summary_file="${out_dir}/${BENCHMARK}_summary.json"
    local stats_file="${out_dir}/${BENCHMARK}_generation_stats.json"

    # === PASS@K 0 CHECK ===
    if [[ ! -s "$eval_judge_output" ]]; then
        echo "[INFO] Judge output boş (muhtemelen pass@k=0). Post-processing atlanıyor."
        touch "$majority_file" "$judged_results_file"
        echo '{"pass_at_k": 0.0, "note": "Skipped due to 0 base corrects"}' > "$summary_file"
        echo '{"note": "Skipped due to 0 base corrects"}' > "$stats_file"
        exit 0
    fi

    python "${PROJECT_ROOT}/scripts/cot_judge_pipeline/04_aggregate_votes.py" --input_file "$eval_judge_output" --output_file "$majority_file"
    python "${PROJECT_ROOT}/scripts/cot_judge_pipeline/05_apply_metrics.py" --base_results_file "$base_results_file" --judge_majority_file "$majority_file" --output_file "$judged_results_file" --summary_file "$summary_file" --stats_file "$stats_file"
}

# ------------------------------------------------------------------------------
# Orchestrator Logic
# ------------------------------------------------------------------------------
run_orchestrator() {
    echo "[INFO] Initializing Unified Pipeline Orchestrator (Monolithic Mode)"

    # Gerekli dizinlerin oluşturulması
    mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results/judgments" "${PROJECT_ROOT}/data/passatk_filtered" "${PROJECT_ROOT}/plots" "${PROJECT_ROOT}/data/cache"

    local script_path=$(realpath "$0")
    local global_last_job_id=""

    for MODEL in $BASE_MODELS; do
        local clean_model=$(basename "$MODEL")
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
                    --export=ALL,RUN_MODE="eval_worker",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",N_SAMPLES="$BASE_N_SAMPLES",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS",HF_TOKEN="${HF_TOKEN}",PROJECT_ROOT="${PROJECT_ROOT}" \
                    "$script_path")

                # 2. Aşama: Pass@K Extraction
                local extract_job=$(sbatch --parsable --dependency=afterany:"${base_job}" \
                    --job-name="extr_${run_id}" \
                    --output="${PROJECT_ROOT}/logs/extr_${run_id}_%j.out" \
                    --ntasks=1 --cpus-per-task="${EXTRACT_SLURM_CPUS}" --mem="${EXTRACT_SLURM_MEM}" \
                    --time="${EXTRACT_SLURM_TIME}" \
                    --export=ALL,RUN_MODE="extract_worker",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS",PROJECT_ROOT="${PROJECT_ROOT}" \
                    "$script_path")

                local filtered_file="${PROJECT_ROOT}/data/passatk_filtered/${clean_model}/${BENCHMARK}_t${B_TEMP}_max${BASE_MAX_COMPLETION_TOKENS}_corrects.jsonl"

                local last_judge_dep="${extract_job}"

                if [[ -z "${JUDGE_MODELS:-}" ]]; then
                    echo "[INFO] JUDGE_MODELS is empty. Skipping judge and post-processing stages for $run_id."
                    global_last_job_id="${extract_job}"
                else
                    for JUDGE in $JUDGE_MODELS; do
                        for J_TEMP in ${JUDGE_TEMPERATURES:-$JUDGE_TEMP}; do
                            local clean_judge=$(basename "$JUDGE")
                            local judge_run_id="${run_id}_J-${clean_judge}_t${J_TEMP}"

                            local judge_nodelist=""
                            [[ -n "${JUDGE_SLURM_NODELIST:-}" ]] && judge_nodelist="--nodelist=$JUDGE_SLURM_NODELIST"

                            # 3. Aşama: CoT Judging
                            local judge_job=$(sbatch --parsable --dependency=afterany:"${last_judge_dep}" \
                                --job-name="jdg_${judge_run_id}" \
                                --output="${PROJECT_ROOT}/logs/jdg_${judge_run_id}_%j.out" \
                                --error="${PROJECT_ROOT}/logs/jdg_${judge_run_id}_%j.err" \
                                --ntasks=1 --cpus-per-task="${JUDGE_SLURM_CPUS}" --mem="${JUDGE_SLURM_MEM}" \
                                --gres="${JUDGE_SLURM_GRES}" --time="${JUDGE_SLURM_TIME}" \
                                $judge_nodelist \
                                --export=ALL,RUN_MODE="judge_worker",JUDGE_MODEL="$JUDGE",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS",FILTERED_FILE="$filtered_file",JUDGE_TEMP="$J_TEMP",HF_TOKEN="${HF_TOKEN}",PROJECT_ROOT="${PROJECT_ROOT}" \
                                "$script_path")

                            # 4. Aşama: Post-processing
                            local post_job=$(sbatch --parsable --dependency=afterany:"${judge_job}" \
                                --job-name="post_${judge_run_id}" \
                                --output="${PROJECT_ROOT}/logs/post_${judge_run_id}_%j.out" \
                                --ntasks=1 --cpus-per-task="${POST_SLURM_CPUS}" --mem="${POST_SLURM_MEM}" \
                                --time="${POST_SLURM_TIME}" \
                                --export=ALL,RUN_MODE="post_worker",JUDGE_MODEL="$JUDGE",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS",JUDGE_TEMP="$J_TEMP",PROJECT_ROOT="${PROJECT_ROOT}" \
                                "$script_path")
                            
                            last_judge_dep="${post_job}"
                        done
                    done
                    
                    global_last_job_id="${last_judge_dep}"
                fi
                echo "[INFO] DAG Submitted. Tail Job ID: $global_last_job_id"
            done
        done
    done
}
case "$RUN_MODE" in
    orchestrator)   run_orchestrator ;;
    eval_worker)    run_eval_worker ;;
    extract_worker) run_extract_worker ;;
    judge_worker)   run_judge_worker ;;
    post_worker)    run_post_worker ;;
    *)              echo "[ERROR] Invalid RUN_MODE: $RUN_MODE" >&2; exit 1 ;;
esac

exit 0