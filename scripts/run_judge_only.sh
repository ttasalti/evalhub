#!/usr/bin/env bash
# scripts/run_judge_only.sh
#
# Run the CoT judge stage over an existing base run, skipping base generation.
# Useful when you already have a *_results.jsonl + *_raw.jsonl on disk and
# want to re-judge it (e.g. with a different judge model, language, or temp).
#
# Pipeline:
#   1. evalhub cot extract, pull base-correct generations into a
#                                      judge-input JSONL.
#   2. start vLLM with the judge model.
#   3. evalhub gen + evalhub eval, judge yes/no per generation.
#   4. stop the judge server.
#   5. evalhub cot finalize, majority-vote + CoT-Pass@K summary.
#
# Output layout (matches the canonical scheme):
#   ${OUTPUT_ROOT}/judgments/<target>_judged_by_<judge>_t<T>_max<N>/<benchmark>/
#       <benchmark>_cot_judge_input.jsonl
#       <benchmark>_cot_majority.jsonl
#       <benchmark>_cot_results.jsonl
#       <benchmark>_cot_summary.json
#       <benchmark>_cot_stats.json
#
# Recommended Slurm header:
#   #SBATCH --job-name=evalhub-judge
#   #SBATCH --partition=gpu
#   #SBATCH --gres=gpu:1
#   #SBATCH --cpus-per-task=16
#   #SBATCH --mem=64G
#   #SBATCH --time=08:00:00
#   #SBATCH --output=logs/slurm-%j.out
#
# Usage:
#   scripts/run_judge_only.sh                    # uses $EVALHUB_PIPELINE_ENV
#   scripts/run_judge_only.sh path/to/judge.env  # explicit env file
#   scripts/run_judge_only.sh --help             # print env contract and exit
#
# Required env: JUDGE_MODEL, BENCHMARK, TARGET_MODEL (used only for naming),
#               and exactly one of:
#                 BASE_RESULTS_DIR                # script derives both JSONLs
#                 BASE_RESULTS_FILE + BASE_RAW_FILE
#
# Judge backend: JUDGE_BACKEND=vllm (default) serves JUDGE_MODEL locally on a
#               GPU. JUDGE_BACKEND=api routes the judge to an external
#               OpenAI-compatible endpoint, no GPU needed, and additionally
#               requires JUDGE_API_BASE and JUDGE_API_KEY (export the key; do
#               not commit it). See scripts/configs/judge_api_deepseek.env.
#
# Optional env: TARGET_STATE, TARGET_TEMPERATURE, TARGET_MAX_COMPLETION_TOKENS
#               (used to compose JUDGE_DIR so it lines up with the base run),
#               JUDGE_STATE, JUDGE_TASK (cot_judge | cot_judge_tr | cot_judge_pt),
#               JUDGE_TEMPERATURE, JUDGE_TOP_P, JUDGE_N_SAMPLES,
#               JUDGE_MAX_COMPLETION_TOKENS, JUDGE_NUM_WORKERS, JUDGE_TIMEOUT,
#               JUDGE_FREQUENCY_PENALTY, JUDGE_PRESENCE_PENALTY, JUDGE_STOP,
#               JUDGE_SYSTEM_PROMPT, JUDGE_TOOL_CONFIG, JUDGE_CALLBACK,
#               JUDGE_MAX_TURNS, JUDGE_ENABLE_MULTITURN, JUDGE_RESUME,
#               JUDGE_PARALLEL_COUNT, OUTPUT_ROOT, JUDGE_PORT, HEALTH_TIMEOUT,
#               JUDGE_REASONING_EFFORT (none|low|medium|high|max), JUDGE_EXTRA_BODY
#               (raw JSON sent as extra_body), JUDGE_EXTRA_BODY_TAG (path slug),
#               JUDGE_STAGED_ROUNDS / JUDGE_COMMIT_WAIT / JUDGE_TOPUP_MAX.
#               The reasoning controls are encoded in the JUDGE_DIR path as
#               __re-<re>__eb-<eb> (default none) and, in api mode, sent to the
#               endpoint: driving thinking models such as DeepSeek V4 Flash.
#               JUDGE_STAGED_ROUNDS>1 produces the N judge samples in N --resume
#               rounds (with JUDGE_COMMIT_WAIT pauses) so a cached API only pays
#               ~1x input instead of Nx; a top-up loop (JUDGE_TOPUP_MAX) then
#               guarantees every solution reaches exactly N samples.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,60p' "$0"
    exit 0
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
    SCRIPT_DIR="${PROJECT_ROOT}/scripts"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

# Activate the project environment when the job does not inherit it. Set
# EVALHUB_CONDA_SH to the site's conda.sh and EVALHUB_CONDA_ENV to the
# environment name (default evalhub_env), or point EVALHUB_ENV_BIN at the
# environment's bin directory. Export these in the shell that submits the job.
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

# Some nodes export ROCR_VISIBLE_DEVICES next to CUDA_VISIBLE_DEVICES; vLLM rejects both being set.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

# shellcheck source=lib/pipeline_common.sh
source "${SCRIPT_DIR}/lib/pipeline_common.sh"

pipeline_load_env "${1:-${EVALHUB_PIPELINE_ENV:-}}"

apply_legacy_env_aliases
require_env JUDGE_MODEL BENCHMARK TARGET_MODEL
apply_target_defaults
apply_judge_defaults
apply_common_defaults
pipeline_init_paths

# Locate the base run files. Acceptance order:
#   1. BASE_RESULTS_FILE + BASE_RAW_FILE point at the files directly.
#   2. BASE_RESULTS_DIR points at a benchmark directory containing both JSONLs.
#   3. Neither given -> derive the base dir the SAME way run_eval_only.sh wrote
#      it, via compose_target_dir() (V4, state-aware). This guarantees the judge
#      reads the base for THIS (target, state, benchmark) and never a different
#      state's base by mistake (the cause of an earlier cross-wired cot).
if [[ -z "${BASE_RESULTS_DIR:-}" && ( -z "${BASE_RESULTS_FILE:-}" || -z "${BASE_RAW_FILE:-}" ) ]]; then
    BASE_RESULTS_DIR="$(compose_target_dir "${BENCHMARK}")"
    pipeline_log "Derived BASE_RESULTS_DIR via compose_target_dir: ${BASE_RESULTS_DIR}"
fi
if [[ -n "${BASE_RESULTS_DIR:-}" ]]; then
    BASE_RESULTS_FILE="${BASE_RESULTS_DIR%/}/${BENCHMARK}_results.jsonl"
    BASE_RAW_FILE="${BASE_RESULTS_DIR%/}/${BENCHMARK}_raw.jsonl"
fi
require_env BASE_RESULTS_FILE BASE_RAW_FILE
[[ -f "${BASE_RESULTS_FILE}" ]] || pipeline_die "Base results not found: ${BASE_RESULTS_FILE}"
[[ -f "${BASE_RAW_FILE}" ]]     || pipeline_die "Base raw not found: ${BASE_RAW_FILE}"

JUDGE_DIR="$(compose_judge_dir "${BENCHMARK}")"
mkdir -p "${JUDGE_DIR}"

pipeline_log "==[1/3]== Extract correct base generations =============================="
JUDGE_INPUT="${JUDGE_DIR}/${BENCHMARK}_cot_judge_input.jsonl"
evalhub cot extract \
    --base-results "${BASE_RESULTS_FILE}" \
    --base-raw "${BASE_RAW_FILE}" \
    --output "${JUDGE_INPUT}"

if [[ ! -s "${JUDGE_INPUT}" ]]; then
    pipeline_log "No correct base generations; CoT-Pass@K = 0 by definition. Stopping."
    pipeline_write_empty_cot_summary "${JUDGE_DIR}" "${BENCHMARK}"
    if [[ -z "${EVALHUB_SKIP_REPORT:-}" ]]; then
        pipeline_log "==[report]== Upserting result rows + refreshing plots =================="
        pipeline_run_report_incremental \
            "$(dirname "${BASE_RESULTS_FILE}")/${BENCHMARK}_summary.json" \
            "${JUDGE_DIR}/${BENCHMARK}_cot_summary.json"
    fi
    exit 0
fi

pipeline_register_cleanup

pipeline_log "==[2/3]== Judge generation & evaluation ================================="
judge_backend_up "${LOG_DIR_LOCAL}/vllm_judge_${SLURM_JOB_ID:-local}_${BENCHMARK}.log"

pipeline_run_judge_gen_eval "${JUDGE_DIR}" "${JUDGE_INPUT}" "${BENCHMARK}"
judge_solutions="${JUDGE_SOLUTIONS_OUT}"
judge_backend_down

pipeline_log "==[3/3]== Aggregate majority vote & CoT-Pass@K =========================="
evalhub cot finalize \
    --base-results "${BASE_RESULTS_FILE}" \
    --base-raw "${BASE_RAW_FILE}" \
    --judge-solutions "${judge_solutions}" \
    --output-dir "${JUDGE_DIR}" \
    --benchmark "${BENCHMARK}"

pipeline_log "[OK] CoT summary: ${JUDGE_DIR}/${BENCHMARK}_cot_summary.json"
pipeline_log "[OK] CoT results: ${JUDGE_DIR}/${BENCHMARK}_cot_results.jsonl"

if [[ -z "${EVALHUB_SKIP_REPORT:-}" ]]; then
    pipeline_log "==[report]== Upserting result rows + refreshing plots =================="
    pipeline_run_report_incremental \
        "$(dirname "${BASE_RESULTS_FILE}")/${BENCHMARK}_summary.json" \
        "${JUDGE_DIR}/${BENCHMARK}_cot_summary.json"
fi
