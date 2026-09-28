#!/usr/bin/env bash
# error_study/run_judge_local.sh: judge the error-injection dataset with a
# LOCAL vLLM-served model, writing NEXT TO the DeepSeek-V4 ("v4") run.
#
#   sbatch error_study/run_judge_local.sh <judge_model> <run_tag>
#   e.g.  sbatch error_study/run_judge_local.sh Qwen/Qwen3.6-35B-A3B qwen35b
#         sbatch error_study/run_judge_local.sh deepseek-ai/DeepSeek-R1-0528-Qwen3-8B r1_8b
#
# Inputs  (READ-ONLY, byte-identical to what the v4 run saw):
#   error_study/output/judge_inputs/manifest.json           50 cells, 7115 items
#   error_study/output/judge_inputs/<model>/<bench>/<error_type>.jsonl
# Output:
#   error_study/output/judge_<tag>/<model>/<bench>/<error_type>/<task>_raw.jsonl
#
# Sampling is FROZEN to the v4 values (n=3, t=0.6, top_p=0.95, max_completion=
# 20480, think state) so the judges differ only in the model. The DeepSeek-only
# --reasoning-effort / --extra-body flags are deliberately NOT sent: vLLM would
# reject them, and Qwen3.x think mode is the chat template's default anyway.
#
# Cells run in manifest order (clean FIRST per group) so the corrupted variants
# reuse the clean solution's prefix in the vLLM prefix cache. The v4 script's
# round-major staging + commit-waits are dropped, those existed only to warm
# DeepSeek's API cache; a local server caches for free and synchronously.
#SBATCH --job-name=es_judge
#SBATCH --partition=slurm_queue
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail

JUDGE_MODEL_ARG="${1:?usage: run_judge_local.sh <judge_model> <run_tag>}"
RUN_TAG="${2:?usage: run_judge_local.sh <judge_model> <run_tag>}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${PROJECT_ROOT}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"

# Activate the project environment when the job does not inherit it (see
# scripts/cot_pipeline.env.example for EVALHUB_CONDA_SH / EVALHUB_ENV_BIN).
if [[ -n "${EVALHUB_CONDA_SH:-}" && "${CONDA_DEFAULT_ENV:-}" != "${EVALHUB_CONDA_ENV:-evalhub_env}" ]]; then
    # shellcheck disable=SC1090
    source "${EVALHUB_CONDA_SH}"
    conda activate "${EVALHUB_CONDA_ENV:-evalhub_env}"
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    # conda activate alone may not override ~/.local/bin; put the env first.
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi
if [[ -n "${EVALHUB_ENV_BIN:-}" ]]; then
    export PATH="${EVALHUB_ENV_BIN}:${PATH}"
fi
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
# Some nodes export ROCR_VISIBLE_DEVICES next to CUDA_VISIBLE_DEVICES; vLLM rejects both.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
# Weights and datasets are expected in the local HF cache.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# shellcheck source=../scripts/lib/pipeline_common.sh
source "${SCRIPT_DIR}/lib/pipeline_common.sh"

OUT="${PROJECT_ROOT}/error_study/output"
RUN_ROOT="${OUT}/judge_${RUN_TAG}"
MANIFEST="${OUT}/judge_inputs/manifest.json"
VGC="${SCRIPT_DIR}/verify_gen_counts.py"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

[[ -f "${MANIFEST}" ]] || pipeline_die "manifest missing: ${MANIFEST}"

# FROZEN sampling knobs (identical to the v4 run)
JUDGE_MODEL="${JUDGE_MODEL_ARG}"
JUDGE_STATE="${JUDGE_STATE:-think}"
JUDGE_N_SAMPLES="${JUDGE_N_SAMPLES:-3}"
JUDGE_MAX_COMPLETION_TOKENS="${JUDGE_MAX_COMPLETION_TOKENS:-20480}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.6}"
JUDGE_TOP_P="${JUDGE_TOP_P:-0.95}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-172800}"

# ---- SERVING knobs (free to tune per model; cannot change the output dist) --
JUDGE_BACKEND="${JUDGE_BACKEND:-vllm}"
JUDGE_PARALLEL_COUNT="${JUDGE_PARALLEL_COUNT:-1}"
# MEASURED, not guessed. Tokenising every one of the 7115 judge inputs with the
# Qwen3.6-35B-A3B tokenizer: the longest solution is 62888 tokens (167809 chars,
# Qwen3.5-9B|aime2026|AIME2026/28|g16), so the longest judge prompt is ~64.4k
# once the question and rubric are added. With cap=20480 the window must exceed
# ~84.9k. Earlier guesses of 65536 and 81920 both left items that died with
# ContextWindowExceededError and were then silently never generated.
# 131072 leaves a 110592-token prompt budget -- 72% headroom over the true max --
# and is natively supported by both judges (35B-A3B 262144, R1-8B 131072/YaRN).
# Costs almost nothing: vLLM allocates KV by ACTUAL sequence length, so only the
# handful of genuinely huge prompts occupy more of the cache.
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-131072}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.92}"
JUDGE_ENFORCE_EAGER="${JUDGE_ENFORCE_EAGER:-false}"
JUDGE_NUM_WORKERS="${JUDGE_NUM_WORKERS:-32}"
JUDGE_VLLM_EXTRA_ARGS="${JUDGE_VLLM_EXTRA_ARGS:-}"
JUDGE_PORT="${JUDGE_PORT:-$(( 20000 + (${SLURM_JOB_ID:-1} % 10000) * 2 + 1 ))}"
# R1-distilled models must use their OWN HF chat template; Qwen3.x uses the
# repo's qwen3.5-think.jinja (which prefills "<think>\n" => think mode ON).
if [[ -z "${JUDGE_USE_TOKENIZER_TEMPLATE:-}" ]]; then
    if [[ "${JUDGE_MODEL}" == *"DeepSeek-R1"* ]]; then
        JUDGE_USE_TOKENIZER_TEMPLATE=true
    else
        JUDGE_USE_TOKENIZER_TEMPLATE=false
    fi
fi

# Per-model measured knobs from the J0 calibration job, if it has run.
CALIB="${SCRIPT_DIR}/configs/.calibrated.env"
if [[ -f "${CALIB}" ]]; then
    # shellcheck disable=SC1090
    source "${CALIB}"
    _pfx="$([[ "${JUDGE_MODEL}" == *"DeepSeek-R1"* ]] && echo R1 || echo Q35B)"
    _nw="${_pfx}_JUDGE_NUM_WORKERS";  JUDGE_NUM_WORKERS="${!_nw:-${JUDGE_NUM_WORKERS}}"
    _gm="${_pfx}_JUDGE_GPU_MEMORY_UTILIZATION"; JUDGE_GPU_MEMORY_UTILIZATION="${!_gm:-${JUDGE_GPU_MEMORY_UTILIZATION}}"
    _ea="${_pfx}_JUDGE_VLLM_EXTRA_ARGS"; JUDGE_VLLM_EXTRA_ARGS="${!_ea:-${JUDGE_VLLM_EXTRA_ARGS}}"
    pipeline_log "Applied calibrated knobs (${_pfx}): nw=${JUDGE_NUM_WORKERS} gmu=${JUDGE_GPU_MEMORY_UTILIZATION} extra='${JUDGE_VLLM_EXTRA_ARGS}'"
fi
export JUDGE_PORT JUDGE_MAX_MODEL_LEN JUDGE_GPU_MEMORY_UTILIZATION \
       JUDGE_ENFORCE_EAGER JUDGE_VLLM_EXTRA_ARGS JUDGE_USE_TOKENIZER_TEMPLATE

pipeline_log "=== error_study judge run ==============================================="
pipeline_log "judge model : ${JUDGE_MODEL}   (state=${JUDGE_STATE}, tokenizer_template=${JUDGE_USE_TOKENIZER_TEMPLATE})"
pipeline_log "run root    : ${RUN_ROOT}"
pipeline_log "sampling    : n=${JUDGE_N_SAMPLES} t=${JUDGE_TEMPERATURE} top_p=${JUDGE_TOP_P} max_completion=${JUDGE_MAX_COMPLETION_TOKENS}"
pipeline_log "serving     : TP=${JUDGE_PARALLEL_COUNT} mml=${JUDGE_MAX_MODEL_LEN} gmu=${JUDGE_GPU_MEMORY_UTILIZATION} nw=${JUDGE_NUM_WORKERS} port=${JUDGE_PORT}"

# bring the judge online
pipeline_register_cleanup
judge_backend_up "${LOG_DIR}/vllm_es_judge_${RUN_TAG}_${SLURM_JOB_ID:-local}.log"

# iterate the 50 cells in manifest order (clean first per group)
# ES_BENCHMARKS restricts this job to a subset of benchmarks so the 50-cell walk
# can be split across GPUs. Cells of different benchmarks live in different
# directories, so parallel jobs never touch the same file. Empty = all.
mapfile -t CELLS < <(ES_BENCHMARKS="${ES_BENCHMARKS:-}" python - "${MANIFEST}" <<'PY'
import json, os, sys
m = json.load(open(sys.argv[1]))
keep = set((os.environ.get("ES_BENCHMARKS") or "").split())
for r in sorted(m["files"], key=lambda r: (r["model"], r["benchmark"], r["run_order_index"])):
    if keep and r["benchmark"] not in keep:
        continue
    print("\t".join([r["model"], r["benchmark"], r["error_type"], r["judge_task"], r["file"], str(r["n_rows"])]))
PY
)
[[ -n "${ES_BENCHMARKS:-}" ]] && pipeline_log "ES_BENCHMARKS filter active: ${ES_BENCHMARKS}"
pipeline_log "manifest: ${#CELLS[@]} cells"

FAILED=(); DONE=0; SKIPPED=0
for cell in "${CELLS[@]}"; do
    IFS=$'\t' read -r model bench etype task rel nrows <<<"${cell}"
    input="${OUT}/judge_inputs/${rel}"
    odir="${RUN_ROOT}/${model}/${bench}/${etype}"
    raw="${odir}/${task}_raw.jsonl"
    label="${model}/${bench}/${etype}"
    mkdir -p "${odir}"

    if python "${VGC}" --input "${input}" --raw "${raw}" --n "${JUDGE_N_SAMPLES}" \
            --label "${label}" --check >/dev/null 2>&1; then
        pipeline_log "[skip] ${label}, already exactly ${JUDGE_N_SAMPLES} (${nrows} items)"
        SKIPPED=$((SKIPPED + 1)); continue
    fi

    pipeline_log "[cell] ${label}, ${nrows} items x ${JUDGE_N_SAMPLES} (task=${task})"
    cell_ok=0
    for attempt in 1 2 3 4; do
        gen_args=(
            --model "hosted_vllm/${JUDGE_MODEL}" --tasks "${task}" --model-state "${JUDGE_STATE}"
            --temperature "${JUDGE_TEMPERATURE}" --top-p "${JUDGE_TOP_P}"
            --n-samples "${JUDGE_N_SAMPLES}" --num-workers "${JUDGE_NUM_WORKERS}"
            --max-completion-tokens "${JUDGE_MAX_COMPLETION_TOKENS}" --timeout "${JUDGE_TIMEOUT}"
            --output-dir "${odir}" --override-args "{\"file_path\": \"${input}\"}"
        )
        # Never truncate: resume tops each task up to n and skips the finished ones.
        [[ -s "${raw}" ]] && gen_args+=(--resume)
        evalhub gen "${gen_args[@]}"

        # exactly-N invariant: trim any excess, then re-check; loop resumes shortfalls.
        python "${VGC}" --input "${input}" --raw "${raw}" --n "${JUDGE_N_SAMPLES}" \
            --label "${label} trim${attempt}" --repair-excess || true
        if python "${VGC}" --input "${input}" --raw "${raw}" --n "${JUDGE_N_SAMPLES}" \
                --label "${label} check${attempt}" --check; then
            cell_ok=1; break
        fi
        pipeline_log "  [${label}] not yet exactly ${JUDGE_N_SAMPLES}, retry ${attempt}/4"
    done

    if [[ "${cell_ok}" == "1" ]]; then
        DONE=$((DONE + 1))
        pipeline_log "[ok]   ${label}"
    else
        FAILED+=("${label}")
        pipeline_log "[FAIL] ${label}, still not exactly ${JUDGE_N_SAMPLES} after 4 attempts"
    fi
done

judge_backend_down

pipeline_log "=== summary: ${DONE} done, ${SKIPPED} already complete, ${#FAILED[@]} failed ==="
if (( ${#FAILED[@]} > 0 )); then
    printf '[FAILED CELL] %s\n' "${FAILED[@]}"
    exit 1
fi
pipeline_log "RAW kept at: ${RUN_ROOT}/<model>/<benchmark>/<error_type>/<task>_raw.jsonl"
pipeline_log "next: error_study/run_analysis.sh ${RUN_TAG}"
