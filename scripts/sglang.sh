#!/bin/bash
# ==============================================================================
# EvalHub Master Orchestrator: Monolithic Pipeline Implementation
# ==============================================================================
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export SGLANG_DISABLE_CUDNN_CHECK=1

eval "$(conda shell.bash hook)"
conda activate evalhub_env

CONFIG_FILE="scripts/sglang.env"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Configuration file $CONFIG_FILE not found." >&2
    exit 1
fi

source "$CONFIG_FILE"

RUN_MODE="${RUN_MODE:-orchestrator}"

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
start_server_and_wait() {
    local model_path="$1"
    local port="$2"
    local p_type="$3"         # e.g. "dp" or "tp"
    local p_count="$4"        # e.g. "2"
    local mem_frac="$5"
    local cons="$6"
    local chunk_prefill="$7"
    local max_reqs="$8"
    
    echo "[INFO] Initializing sglang server for model: $model_path on port $port"
    echo "[INFO] Hardware Strategy: --$p_type $p_count"
    
    python -m sglang.launch_server --model-path "$model_path" --port "$port" \
        --"$p_type" "$p_count" \
        --mem-fraction-static "$mem_frac" \
        --schedule-conservativeness "$cons" \
        --chunked-prefill-size "$chunk_prefill" \
        --max-running-requests "$max_reqs" \
        --disable-custom-all-reduce &
    
    SERVER_PID=$!
    local retries=0
    local max_retries=360
    
    echo "[INFO] Waiting for server healthcheck..."
    while ! curl -s "http://127.0.0.1:$port/health" > /dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[ERROR] SGLang server process ($SERVER_PID) died unexpectedly during initialization." >&2
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
    
    local worker_port=$((30000 + RANDOM % 10000))
    export EVALHUB_CACHE_DIR="data/cache/eval_${RANDOM}"
    mkdir -p "$EVALHUB_CACHE_DIR"
    
    trap cleanup_server_and_cache EXIT

    start_server_and_wait "$TARGET_MODEL" "$worker_port" \
        "$BASE_PARALLEL_TYPE" "$BASE_PARALLEL_COUNT" \
        "$BASE_SGL_MEM_FRACTION" "$BASE_SGL_CONSERVATIVENESS" \
        "$BASE_SGL_CHUNKED_PREFILL" "$BASE_SGL_MAX_RUNNING_REQS"
    
    export HOSTED_VLLM_API_BASE="http://127.0.0.1:$worker_port/v1"
    export HOSTED_VLLM_API_KEY="EMPTY"

    local clean_model=$(basename "$TARGET_MODEL")
    local out_dir="results/${clean_model}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}/${BENCHMARK}"
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
    [[ -n "${BASE_TOP_P:-}" ]] && cmd_args+=(--top-p "$BASE_TOP_P")
    [[ -n "${BASE_FREQUENCY_PENALTY:-}" ]] && cmd_args+=(--frequency-penalty "$BASE_FREQUENCY_PENALTY")
    [[ -n "${BASE_PRESENCE_PENALTY:-}" ]] && cmd_args+=(--presence-penalty "$BASE_PRESENCE_PENALTY")
    [[ -n "${BASE_STOP:-}" ]] && cmd_args+=(--stop "$BASE_STOP")
    [[ -n "${BASE_SYSTEM_PROMPT:-}" ]] && cmd_args+=(--system-prompt "$BASE_SYSTEM_PROMPT")
    [[ -n "${BASE_OVERRIDE_ARGS:-}" ]] && cmd_args+=(--override-args "$BASE_OVERRIDE_ARGS")

    echo "[INFO] Executing generation phase with full parameters..."
    python -m evalhub.cli gen "${cmd_args[@]}"

    local solutions_file="$out_dir/${BENCHMARK}.jsonl"
    [[ ! -f "$solutions_file" ]] && solutions_file="$out_dir/${BENCHMARK}_raw.jsonl"

    echo "[INFO] Executing evaluation phase..."
    local eval_args=(
        --tasks "$BENCHMARK"
        --solutions "$solutions_file"
        --output-dir "$out_dir"
    )
    [[ -n "${BASE_OVERRIDE_ARGS:-}" ]] && eval_args+=(--override-args "$BASE_OVERRIDE_ARGS")
    
    python -m evalhub.cli eval "${eval_args[@]}"
}

run_extract_worker() {
    echo "[INFO] --- Running Pass@K Extraction ---"
    
    local clean_model=$(basename "$TARGET_MODEL")
    local filtered_dir="data/passatk_filtered/${clean_model}"
    mkdir -p "$filtered_dir"
    
    local out_dir="results/${clean_model}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}/${BENCHMARK}"    
    local results_file="${out_dir}/${BENCHMARK}_results.jsonl"
    local raw_file="${out_dir}/${BENCHMARK}_raw.jsonl"
    local output_filtered="${filtered_dir}/${BENCHMARK}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}_corrects.jsonl"    
    
    python scripts/cot_judge_pipeline/01_extract_corrects.py \
        --results_file "$results_file" \
        --raw_file "$raw_file" \
        --output_file "$output_filtered"
}

run_judge_worker() {
    echo "[INFO] --- Starting CoT Judging ---"
    
    local worker_port=$((40000 + RANDOM % 10000))
    export EVALHUB_CACHE_DIR="data/cache/judge_${RANDOM}"
    mkdir -p "$EVALHUB_CACHE_DIR"

    trap cleanup_server_and_cache EXIT

    start_server_and_wait "$JUDGE_MODEL" "$worker_port" \
        "$JUDGE_PARALLEL_TYPE" "$JUDGE_PARALLEL_COUNT" \
        "$JUDGE_SGL_MEM_FRACTION" "$JUDGE_SGL_CONSERVATIVENESS" \
        "$JUDGE_SGL_CHUNKED_PREFILL" "$JUDGE_SGL_MAX_RUNNING_REQS"

    export HOSTED_VLLM_API_BASE="http://127.0.0.1:$worker_port/v1"
    export HOSTED_VLLM_API_KEY="EMPTY"

    local clean_target=$(basename "$TARGET_MODEL")
    local clean_judge=$(basename "$JUDGE_MODEL")
    local out_dir="results/judgments/${clean_target}evaluated_by${clean_judge}_${JUDGE_MAX_COMPLETION_TOKENS}/${BENCHMARK}_t${JUDGE_TEMP}"   
    mkdir -p "$out_dir"

    if [[ -z "${FILTERED_FILE:-}" ]]; then
        FILTERED_FILE="data/passatk_filtered/${clean_target}/${BENCHMARK}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}_corrects.jsonl"
    fi

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

    [[ "${JUDGE_RESUME:-false}" == "true" ]] && cmd_args+=(--resume)
    [[ -n "${JUDGE_TOP_P:-}" ]] && cmd_args+=(--top-p "$JUDGE_TOP_P")
    [[ -n "${JUDGE_FREQUENCY_PENALTY:-}" ]] && cmd_args+=(--frequency-penalty "$JUDGE_FREQUENCY_PENALTY")
    [[ -n "${JUDGE_PRESENCE_PENALTY:-}" ]] && cmd_args+=(--presence-penalty "$JUDGE_PRESENCE_PENALTY")
    [[ -n "${JUDGE_STOP:-}" ]] && cmd_args+=(--stop "$JUDGE_STOP")
    [[ -n "${JUDGE_SYSTEM_PROMPT:-}" ]] && cmd_args+=(--system-prompt "$JUDGE_SYSTEM_PROMPT")

    echo "[INFO] Executing judgement generation phase with full parameters..."
    python -m evalhub.cli gen "${cmd_args[@]}"

    echo "[INFO] Executing judgement evaluation phase..."
    local judge_solutions_file="$out_dir/math_judge.jsonl"
    [[ ! -f "$judge_solutions_file" ]] && judge_solutions_file="$out_dir/math_judge_raw.jsonl"

    python -m evalhub.cli eval \
        --tasks "math_judge" \
        --solutions "$judge_solutions_file" \
        --output-dir "$out_dir" \
        --override-args "$final_override"
}

run_post_worker() {
    echo "[INFO] --- Post-Processing Judge Outputs & Plotting ---"
    
    local clean_target=$(basename "$TARGET_MODEL")
    local clean_judge=$(basename "$JUDGE_MODEL")
    
    local out_dir="results/judgments/${clean_target}evaluated_by${clean_judge}_${JUDGE_MAX_COMPLETION_TOKENS}/${BENCHMARK}_t${JUDGE_TEMP}"
    rm -f "${out_dir}/math_judge_summary.json"

    local eval_judge_output="${out_dir}/math_judge.jsonl"
    [[ ! -f "$eval_judge_output" ]] && eval_judge_output="${out_dir}/math_judge_raw.jsonl"

    local majority_file="${out_dir}/math_judge_majority.jsonl"
    local base_results_path="results/${clean_target}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}/${BENCHMARK}"
    local base_results_file="${base_results_path}/${BENCHMARK}_results.jsonl"

    local judged_results_file="${out_dir}/${BENCHMARK}_results.jsonl"
    local summary_file="${out_dir}/${BENCHMARK}_summary.json"
    local stats_file="${out_dir}/${BENCHMARK}_generation_stats.json"

    if [[ ! -f "$base_results_file" ]]; then
        echo "[ERROR] Base results file not found: $base_results_file" >&2
        exit 1
    fi

    echo "[INFO] Aggregating votes..."
    python scripts/cot_judge_pipeline/04_aggregate_votes.py \
        --input_file "$eval_judge_output" \
        --output_file "$majority_file"

    echo "[INFO] Applying CoT filters..."
    python scripts/cot_judge_pipeline/05_apply_metrics.py \
        --base_results_file "$base_results_file" \
        --judge_majority_file "$majority_file" \
        --output_file "$judged_results_file" \
        --summary_file "$summary_file" \
        --stats_file "$stats_file"
        
    echo "[INFO] Triggering plotting utility..."
    if [[ -f "scripts/utils/plot_cot_metrics.py" ]]; then
         python scripts/utils/plot_cot_metrics.py \
             --summary_file "$summary_file" \
             --benchmark "$BENCHMARK" \
             --model "$clean_target" \
             --judge "$clean_judge" \
             --out_dir "plots"
    else
         echo "[WARNING] Plotting script not found. Skipping visualization."
    fi
}

# ------------------------------------------------------------------------------
# Orchestrator Logic
# ------------------------------------------------------------------------------
run_orchestrator() {
    echo "[INFO] Initializing Unified Pipeline Orchestrator (Monolithic Mode)"

    mkdir -p logs results/judgments data/passatk_filtered plots data/cache

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

                local base_job=$(sbatch --parsable $global_dep \
                    --job-name="eval_${run_id}" \
                    --output="logs/eval_${run_id}_%j.out" \
                    --error="logs/eval_${run_id}_%j.err" \
                    --ntasks=1 --cpus-per-task="${BASE_SLURM_CPUS}" --mem="${BASE_SLURM_MEM}" \
                    --gres="${BASE_SLURM_GRES}" --time="${BASE_SLURM_TIME}" \
                    $([[ -n "$BASE_SLURM_NODELIST" ]] && echo "--nodelist=$BASE_SLURM_NODELIST") \
                    --export=ALL,RUN_MODE="eval_worker",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",N_SAMPLES="$BASE_N_SAMPLES",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS" \
                    "$script_path")

                local extract_job=$(sbatch --parsable --dependency=afterok:"${base_job}" \
                    --job-name="extr_${run_id}" \
                    --output="logs/extr_${run_id}_%j.out" \
                    --ntasks=1 --cpus-per-task="${EXTRACT_SLURM_CPUS}" --mem="${EXTRACT_SLURM_MEM}" \
                    --time="${EXTRACT_SLURM_TIME}" \
                    --export=ALL,RUN_MODE="extract_worker",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",N_SAMPLES="$BASE_N_SAMPLES",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS" \
                    "$script_path")

                local filtered_file="data/passatk_filtered/${clean_model}/${BENCHMARK}_t${B_TEMP}_max${BASE_MAX_COMPLETION_TOKENS}_corrects.jsonl"

                if [[ -z "${JUDGE_MODELS:-}" ]]; then
                    echo "[INFO] JUDGE_MODELS is empty. Skipping judge and post-processing stages for $run_id."
                    global_last_job_id="${extract_job}"
                else
                    local all_post_jobs=()
                    for JUDGE in $JUDGE_MODELS; do
                        for J_TEMP in $JUDGE_TEMPERATURES; do
                            local clean_judge=$(basename "$JUDGE")
                            local judge_run_id="${run_id}_J-${clean_judge}_t${J_TEMP}"

                            local judge_job=$(sbatch --parsable --dependency=afterok:"${extract_job}" \
                                --job-name="jdg_${judge_run_id}" \
                                --output="logs/jdg_${judge_run_id}_%j.out" \
                                --error="logs/jdg_${judge_run_id}_%j.err" \
                                --ntasks=1 --cpus-per-task="${JUDGE_SLURM_CPUS}" --mem="${JUDGE_SLURM_MEM}" \
                                --gres="${JUDGE_SLURM_GRES}" --time="${JUDGE_SLURM_TIME}" \
                                $([[ -n "$JUDGE_SLURM_NODELIST" ]] && echo "--nodelist=$JUDGE_SLURM_NODELIST") \
                                --export=ALL,RUN_MODE="judge_worker",JUDGE_MODEL="$JUDGE",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS",FILTERED_FILE="$filtered_file",JUDGE_TEMP="$J_TEMP" \
                                "$script_path")

                            local post_job=$(sbatch --parsable --dependency=afterok:"${judge_job}" \
                                --job-name="post_${judge_run_id}" \
                                --output="logs/post_${judge_run_id}_%j.out" \
                                --ntasks=1 --cpus-per-task="${POST_SLURM_CPUS}" --mem="${POST_SLURM_MEM}" \
                                --time="${POST_SLURM_TIME}" \
                                --export=ALL,RUN_MODE="post_worker",JUDGE_MODEL="$JUDGE",TARGET_MODEL="$MODEL",BENCHMARK="$BENCHMARK",TEMPERATURE="$B_TEMP",MAX_COMPLETION_TOKENS="$BASE_MAX_COMPLETION_TOKENS",JUDGE_TEMP="$J_TEMP" \
                                "$script_path")
                            
                            all_post_jobs+=("$post_job")
                        done
                    done
                    
                    global_last_job_id=$(IFS=, ; echo "${all_post_jobs[*]}")
                fi
                echo "[INFO] DAG Submitted. Tail Job IDs: $global_last_job_id"
            done
        done
    done
}

# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
case "$RUN_MODE" in
    orchestrator)   run_orchestrator ;;
    eval_worker)    run_eval_worker ;;
    extract_worker) run_extract_worker ;;
    judge_worker)   run_judge_worker ;;
    post_worker)    run_post_worker ;;
    *)              echo "[ERROR] Invalid RUN_MODE: $RUN_MODE" >&2; exit 1 ;;
esac

exit 0