#!/bin/bash
# ==============================================================================
# EvalHub Judge-Only Orchestrator: Dynamic Path Extraction
# ==============================================================================
set -euo pipefail

# 1. Path Independence
if [[ -z "${PROJECT_ROOT:-}" ]]; then
    export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_TRITON_FLASH_ATTN=0 

# Conda activation
eval "$(conda shell.bash hook 2>/dev/null || echo '')"
conda activate evalhub_env || echo "[WARNING] Conda environment failed to activate, falling back to system Python."

# Yeni konfigürasyon dosyasının gösterilmesi
CONFIG_FILE="${PROJECT_ROOT}/scripts/judge_only.env"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Configuration file $CONFIG_FILE not found." >&2
    exit 1
fi

# 2. Environment Variable Integration
set -a
source "$CONFIG_FILE"
set +a

export HF_TOKEN="${HF_TOKEN:-}"
RUN_MODE="${RUN_MODE:-orchestrator}"
BASE_PORT="${PORT:-30000}"
export WAIT_FOR_JOB_ID="${WAIT_FOR_JOB_ID:-}"

# ------------------------------------------------------------------------------
# 3. Dynamic Path Metadata Extraction & Validation
# ------------------------------------------------------------------------------
if [[ -z "${BASE_GENERATION_PATH:-}" ]]; then
    echo "[ERROR] BASE_GENERATION_PATH is not set in $CONFIG_FILE" >&2
    exit 1
fi

# Trailing slash temizleme (Sondaki fazladan eğik çizgiyi siler)
BASE_GENERATION_PATH="${BASE_GENERATION_PATH%/}"

# Path hiyerarşisini ayrıştırma
BENCHMARK=$(basename "$BASE_GENERATION_PATH")
MODEL_DIR_PATH=$(dirname "$BASE_GENERATION_PATH")
PARENT_DIR=$(basename "$MODEL_DIR_PATH")

# DİNAMİK KATEGORİ ÇIKARIMI: 'base', 'instruct' veya 'reasoning' klasörünü bulur.
# Örnek path: .../results/base/qwen3.5_t0.7_max1024/gsm8k -> CATEGORY_DIR="base" olur.
CATEGORY_DIR=$(basename "$(dirname "$MODEL_DIR_PATH")")

# Kategoriye göre kök dizinlerin dinamik atanması (env'deki hardcoded ayarların yerine geçer)
export RESULTS_BASE_DIR="${PROJECT_ROOT}/results/${CATEGORY_DIR}"
export FILTERED_BASE_DIR="${PROJECT_ROOT}/data/${CATEGORY_DIR}"

# Metadata çıkarımı: Klasör adının formatının <model>_t<temp>_max<tokens> olduğu varsayılıyor
MAX_COMPLETION_TOKENS=$(echo "$PARENT_DIR" | grep -o 'max[0-9]*$' | sed 's/max//')
TEMPERATURE=$(echo "$PARENT_DIR" | sed -n 's/.*_t\([0-9]*\.[0-9]*\)_max.*/\1/p')
CLEAN_TARGET=$(echo "$PARENT_DIR" | sed "s/_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}//")

if [[ -z "$CLEAN_TARGET" || -z "$TEMPERATURE" || -z "$MAX_COMPLETION_TOKENS" ]]; then
    echo "[ERROR] Yetersiz path formatı. Beklenen format: .../<kategori>/<model>_t<temp>_max<tokens>/<benchmark>" >&2
    exit 1
fi

# THINK MODE ÇIKARIMI:
# Kategori "reasoning" ise veya clean target "think" içeriyorsa true, base/instruct gibi durumlar için false.
if [[ "$CATEGORY_DIR" == "reasoning" ]] || [[ "$CLEAN_TARGET" == *"think"* ]]; then
    export THINK_MODE="true"
else
    export THINK_MODE="false"
fi

echo "[INFO] Tespit edilen kategori: $CATEGORY_DIR"
echo "[INFO] Think Mode (Hedef Model İçin): $THINK_MODE"
echo "[INFO] Sonuçlar şu dizine kaydedilecek: $RESULTS_BASE_DIR"

# Model Tipi Kontrolü (Gemma veya Qwen3.5 yakalama)
model_lower=$(echo "$CLEAN_TARGET" | tr '[:upper:]' '[:lower:]')
if [[ "$model_lower" == *"qwen3.5"* ]] || [[ "$model_lower" == *"gemma"* ]]; then
    echo "[INFO] Geçerli hedef model path üzerinden tespit edildi: $CLEAN_TARGET"
else
    echo "[WARN] Path'ten tespit edilen model 'qwen3.5' veya 'gemma' içermiyor olabilir: $CLEAN_TARGET"
fi

# Sistem uyumluluğu için
TARGET_MODEL="$CLEAN_TARGET"

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
start_server_and_wait() {
    local model_path="$1"
    local port="$2"
    local p_count="$3"
    
    mkdir -p logs
    local log_file="logs/vllm_server_error_${port}.log"
    touch "$log_file"
    
    echo "[INFO] Initializing vLLM server for model: $model_path on port $port"
    echo "[INFO] Hardware Strategy: --tensor-parallel-size $p_count"

    local vllm_args=(
        --model "$model_path"
        --port "$port"
        --tensor-parallel-size "$p_count"
        --trust-remote-code
    )

    local template_dir="${PROJECT_ROOT}/scripts/templates"
    local template_file=""
    local cur_model_lower=$(echo "$model_path" | tr '[:upper:]' '[:lower:]')
    local think_mode_lower=$(echo "$THINK_MODE" | tr '[:upper:]' '[:lower:]')
    
    local is_base=false
    if [[ "$cur_model_lower" == *"base"* ]] || [[ "$cur_model_lower" == *"e2b"* ]] || [[ "$cur_model_lower" == *"e4b"* ]]; then
        is_base=true
    fi

    if [[ "$cur_model_lower" == *"qwen"* ]] && [[ "$is_base" == true ]]; then
        template_file="${template_dir}/qwen3.5-base.jinja"
    elif [[ "$cur_model_lower" == *"gemma"* ]] && [[ "$is_base" == true ]]; then
        template_file="${template_dir}/gemma4-base.jinja"
    elif [[ "$cur_model_lower" == *"ministral"* ]] && [[ "$is_base" == true ]]; then
        template_file="${template_dir}/ministral3-base.jinja"
    elif [[ "$cur_model_lower" == *"ministral"* ]] && [[ "$cur_model_lower" == *"instruct"* ]]; then
        template_file="${template_dir}/ministral3-instruct.jinja"
    elif [[ "$cur_model_lower" == *"ministral"* ]] && [[ "$cur_model_lower" == *"reasoning"* ]]; then
        template_file="${template_dir}/ministral3-reasoning.jinja"
    elif [[ "$cur_model_lower" == *"qwen"* ]]; then
        if [[ "$think_mode_lower" == "true" ]]; then
            template_file="${template_dir}/qwen3.5-think.jinja"
        else
            template_file="${template_dir}/qwen3.5-no-think.jinja"
        fi
    elif [[ "$cur_model_lower" == *"gemma"* ]]; then
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
            echo "[ERROR] Server failed health check timeout." >&2
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
        SERVER_PID="" 
    fi
    if [[ -n "${EVALHUB_CACHE_DIR:-}" ]]; then
        rm -rf "$EVALHUB_CACHE_DIR"
    fi
}

# ------------------------------------------------------------------------------
# Workers
# ------------------------------------------------------------------------------
run_extract_worker() {
    echo "[INFO] --- Running Pass@K Extraction for $CLEAN_TARGET ---"
    
    local filtered_dir="${FILTERED_BASE_DIR}/${CLEAN_TARGET}"
    mkdir -p "$filtered_dir" "${PROJECT_ROOT}/logs"
    
    local out_dir="$BASE_GENERATION_PATH"    
    local results_file="${out_dir}/${BENCHMARK}_results.jsonl"
    local raw_file="${out_dir}/${BENCHMARK}_raw.jsonl"
    local output_filtered="${filtered_dir}/${BENCHMARK}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}_corrects.jsonl"    
    
    if [[ ! -f "$results_file" && ! -f "$raw_file" ]]; then
        echo "[ERROR] Base evaluation dosyaları bulunamadı: $out_dir"
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
    
    local clean_judge=$(basename "$JUDGE_MODEL")
    # Sondaki _t$JUDGE_TEMP kaldırıldı
    local out_dir="${RESULTS_BASE_DIR}/judgments/${CLEAN_TARGET}_evaluated_by_${clean_judge}_${JUDGE_MAX_COMPLETION_TOKENS}/${BENCHMARK}"   
    mkdir -p "$out_dir"

    if [[ -z "${FILTERED_FILE:-}" ]]; then
        FILTERED_FILE="${FILTERED_BASE_DIR}/${CLEAN_TARGET}/${BENCHMARK}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}_corrects.jsonl"
    fi

    if [[ ! -s "$FILTERED_FILE" ]]; then
        echo "[INFO] Filtrelenmiş dosya boş veya eksik (pass@k=0). Judge worker atlanıyor."
        touch "$out_dir/math_judge.jsonl"
        return 0
    fi
    
    # Kural: Judge her zaman THINK_MODE="true" olarak çalışır.
    export THINK_MODE="true"
    
    local worker_port=$((BASE_PORT + 10000 + RANDOM % 10000))
    export EVALHUB_CACHE_DIR="${PROJECT_ROOT}/data/cache/judge_${RANDOM}"
    mkdir -p "$EVALHUB_CACHE_DIR" "${PROJECT_ROOT}/logs"
    
    start_server_and_wait "$JUDGE_MODEL" "$worker_port" "$JUDGE_PARALLEL_COUNT"

    export HOSTED_VLLM_API_BASE="http://127.0.0.1:$worker_port/v1"
    export HOSTED_VLLM_API_KEY="EMPTY"

    local final_override="{\"file_path\": \"$FILTERED_FILE\"}"
    
    local dynamic_workers=192
    local cur_model_lower=$(echo "$JUDGE_MODEL" | tr '[:upper:]' '[:lower:]')
    
    if [[ "$cur_model_lower" == *"0.8b"* ]]; then dynamic_workers=512;
    elif [[ "$cur_model_lower" == *"e2b"* ]] || [[ "$cur_model_lower" == *"2b"* ]] || [[ "$cur_model_lower" == *"3b"* ]]; then dynamic_workers=384;
    elif [[ "$cur_model_lower" == *"e4b"* ]] || [[ "$cur_model_lower" == *"4b"* ]] || [[ "$cur_model_lower" == *"8b"* ]] || [[ "$cur_model_lower" == *"9b"* ]]; then dynamic_workers=256;
    elif [[ "$cur_model_lower" == *"14b"* ]] || [[ "$cur_model_lower" == *"35b"* ]]; then dynamic_workers=192; fi

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
    
    local clean_judge=$(basename "$JUDGE_MODEL")
    # Sondaki _t$JUDGE_TEMP kaldırıldı
    local out_dir="${RESULTS_BASE_DIR}/judgments/${CLEAN_TARGET}_evaluated_by_${clean_judge}_${JUDGE_MAX_COMPLETION_TOKENS}/${BENCHMARK}"
    rm -f "${out_dir}/math_judge_summary.json"

    local eval_judge_output="${out_dir}/math_judge.jsonl"
    [[ ! -f "$eval_judge_output" ]] && eval_judge_output="${out_dir}/math_judge_raw.jsonl"

    local majority_file="${out_dir}/math_judge_majority.jsonl"
    local base_results_file="${BASE_GENERATION_PATH}/${BENCHMARK}_results.jsonl"

    local judged_results_file="${out_dir}/${BENCHMARK}_results.jsonl"
    local summary_file="${out_dir}/${BENCHMARK}_summary.json"
    local stats_file="${out_dir}/${BENCHMARK}_generation_stats.json"

    if [[ ! -s "$eval_judge_output" ]]; then
        echo "[INFO] Judge output boş. Post-processing atlanıyor."
        touch "$majority_file" "$judged_results_file"
        echo '{"pass_at_k": 0.0, "note": "Skipped due to 0 base corrects"}' > "$summary_file"
        echo '{"note": "Skipped due to 0 base corrects"}' > "$stats_file"
        return 0
    fi

    python "${PROJECT_ROOT}/scripts/cot_judge_pipeline/04_aggregate_votes.py" --input_file "$eval_judge_output" --output_file "$majority_file"
    python "${PROJECT_ROOT}/scripts/cot_judge_pipeline/05_apply_metrics.py" --base_results_file "$base_results_file" --judge_majority_file "$majority_file" --output_file "$judged_results_file" --summary_file "$summary_file" --stats_file "$stats_file"
}

run_extract_judge_post_worker() {
    echo "[INFO] --- Starting Judge-Only Pipeline (Extract -> Judge -> Post) ---"
    trap cleanup_server_and_cache EXIT

    run_extract_worker

    local filtered_file="${FILTERED_BASE_DIR}/${CLEAN_TARGET}/${BENCHMARK}_t${TEMPERATURE}_max${MAX_COMPLETION_TOKENS}_corrects.jsonl"
    export FILTERED_FILE="$filtered_file"

    if [[ -z "${JUDGE_MODELS:-}" ]]; then
        echo "[INFO] No JUDGE_MODELS defined. Completing job after extraction."
        return 0
    fi

    for CURRENT_JUDGE in $JUDGE_MODELS; do
        for CURRENT_J_TEMP in ${JUDGE_TEMPERATURES:-$JUDGE_TEMP}; do
            echo "[INFO] ========================================================"
            echo "[INFO] Running Judge Pipeline -> Judge Model: $CURRENT_JUDGE | Temp: $CURRENT_J_TEMP"
            echo "[INFO] ========================================================"
            
            export JUDGE_MODEL="$CURRENT_JUDGE"
            export JUDGE_TEMP="$CURRENT_J_TEMP"
            export JUDGE_N_SAMPLES="${JUDGE_N_SAMPLES:-1}"
            export JUDGE_MAX_COMPLETION_TOKENS="${JUDGE_MAX_COMPLETION_TOKENS:-16384}"
            
            run_judge_worker
            run_post_worker
        done
    done
    
    echo "[INFO] Pipeline Completed Successfully."
}

# ------------------------------------------------------------------------------
# Orchestrator Logic
# ------------------------------------------------------------------------------
run_orchestrator() {
    echo "[INFO] Initializing Judge-Only Unified Pipeline"
    
    mkdir -p "${PROJECT_ROOT}/logs" "${RESULTS_BASE_DIR}/judgments" "${FILTERED_BASE_DIR}" "${PROJECT_ROOT}/data/cache"

    local script_path=$(realpath "$0")
    local run_id="judge_${CLEAN_TARGET}_${BENCHMARK}_t${TEMPERATURE}"
    
    echo "[INFO] --------------------------------------------------"
    echo "[INFO] Submitting Slurm Job for Target PATH: $BASE_GENERATION_PATH"
    echo "[INFO] Run ID: $run_id"
    echo "[INFO] --------------------------------------------------"

    local global_dep=""
    if [[ -n "${WAIT_FOR_JOB_ID:-}" ]]; then
        global_dep="--dependency=afterany:${WAIT_FOR_JOB_ID}"
        echo "[INFO] Waiting for Job ID: $WAIT_FOR_JOB_ID"
    fi

    local judge_nodelist=""
    [[ -n "${JUDGE_SLURM_NODELIST:-}" ]] && judge_nodelist="--nodelist=$JUDGE_SLURM_NODELIST"

    # Export listesine kategori dinamiklerini de paslıyoruz. 
    # Slurm alt işlemde bu değerlere ihtiyaç duyacak
    local pipe_job=$(sbatch --parsable $global_dep \
        --job-name="pipe_${run_id}" \
        --output="${PROJECT_ROOT}/logs/pipe_${run_id}_%j.out" \
        --error="${PROJECT_ROOT}/logs/pipe_${run_id}_%j.err" \
        --ntasks=1 --cpus-per-task="${JUDGE_SLURM_CPUS}" --mem="${JUDGE_SLURM_MEM}" \
        --gres="${JUDGE_SLURM_GRES}" --time="${JUDGE_SLURM_TIME}" \
        $judge_nodelist \
        --export=ALL,RUN_MODE="extract_judge_post_worker",CLEAN_TARGET="$CLEAN_TARGET",BENCHMARK="$BENCHMARK",TEMPERATURE="$TEMPERATURE",MAX_COMPLETION_TOKENS="$MAX_COMPLETION_TOKENS",BASE_GENERATION_PATH="$BASE_GENERATION_PATH",HF_TOKEN="${HF_TOKEN}",PROJECT_ROOT="${PROJECT_ROOT}",THINK_MODE="${THINK_MODE}",CATEGORY_DIR="${CATEGORY_DIR}" \
        "$script_path")
        
    echo "[INFO] DAG Submitted successfully. Job ID: $pipe_job"
}

case "$RUN_MODE" in
    orchestrator)              run_orchestrator ;;
    extract_worker)            run_extract_worker ;;
    judge_worker)              run_judge_worker ;;
    post_worker)               run_post_worker ;;
    extract_judge_post_worker) run_extract_judge_post_worker ;;
    *)                         echo "[ERROR] Invalid RUN_MODE: $RUN_MODE" >&2; exit 1 ;;
esac

exit 0