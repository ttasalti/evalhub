#!/usr/bin/env bash
# ============================================================================
# scripts/run_judge_only.sh
#
# Run the CoT judge stage over an existing base run, skipping base generation.
# Useful when you already have a *_results.jsonl + *_raw.jsonl on disk and
# want to re-judge it (e.g. with a different judge model, language, or temp).
#
# Pipeline:
#   1. evalhub cot extract           — pull base-correct generations into a
#                                      judge-input JSONL.
#   2. start vLLM with the judge model.
#   3. evalhub gen + evalhub eval    — judge yes/no per generation.
#   4. stop the judge server.
#   5. evalhub cot finalize          — majority-vote + CoT-Pass@K summary.
#
# Output layout (matches the canonical scheme):
#   ${OUTPUT_ROOT}/judgments/<target>_judged_by_<judge>_t<T>_max<N>/<benchmark>/
#       <benchmark>_cot_judge_input.jsonl
#       <benchmark>_cot_majority.jsonl
#       <benchmark>_cot_results.jsonl
#       <benchmark>_cot_summary.json
#       <benchmark>_cot_stats.json
#
# ----------------------------------------------------------------------------
# Recommended Slurm header for nscluster:
#   #SBATCH --job-name=evalhub-judge
#   #SBATCH --partition=gpu
#   #SBATCH --gres=gpu:1
#   #SBATCH --cpus-per-task=16
#   #SBATCH --mem=64G
#   #SBATCH --time=08:00:00
#   #SBATCH --output=logs/slurm-%j.out
# ----------------------------------------------------------------------------
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
# Optional env: TARGET_STATE, TARGET_TEMPERATURE, TARGET_MAX_COMPLETION_TOKENS
#               (used to compose JUDGE_DIR so it lines up with the base run),
#               JUDGE_STATE, JUDGE_TASK (cot_judge | cot_judge_tr | cot_judge_pt),
#               JUDGE_TEMPERATURE, JUDGE_TOP_P, JUDGE_N_SAMPLES,
#               JUDGE_MAX_COMPLETION_TOKENS, JUDGE_NUM_WORKERS, JUDGE_TIMEOUT,
#               JUDGE_FREQUENCY_PENALTY, JUDGE_PRESENCE_PENALTY, JUDGE_STOP,
#               JUDGE_SYSTEM_PROMPT, JUDGE_TOOL_CONFIG, JUDGE_CALLBACK,
#               JUDGE_MAX_TURNS, JUDGE_ENABLE_MULTITURN, JUDGE_RESUME,
#               JUDGE_PARALLEL_COUNT, OUTPUT_ROOT, JUDGE_PORT, HEALTH_TIMEOUT.
# ============================================================================
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,50p' "$0"
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=lib/pipeline_common.sh
source "${SCRIPT_DIR}/lib/pipeline_common.sh"

pipeline_load_env "${1:-${EVALHUB_PIPELINE_ENV:-}}"

require_env JUDGE_MODEL BENCHMARK TARGET_MODEL
apply_target_defaults
apply_judge_defaults
apply_common_defaults
pipeline_init_paths

# Locate the base run files. Two acceptance modes:
#   1. BASE_RESULTS_DIR points at a benchmark directory containing both JSONLs.
#   2. BASE_RESULTS_FILE + BASE_RAW_FILE point at the files directly.
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
    exit 0
fi

pipeline_register_cleanup

pipeline_log "==[2/3]== Judge generation & evaluation ================================="
start_vllm "${JUDGE_MODEL}" "${JUDGE_PORT}" "${JUDGE_PARALLEL_COUNT}" "${JUDGE_STATE}" \
    "${LOG_DIR_LOCAL}/vllm_judge_${JUDGE_PORT}.log"
export HOSTED_VLLM_API_BASE="http://127.0.0.1:${JUDGE_PORT}/v1"
export HOSTED_VLLM_API_KEY="EMPTY"

judge_solutions="$(pipeline_run_judge_gen_eval "${JUDGE_DIR}" "${JUDGE_INPUT}")"
stop_vllm

pipeline_log "==[3/3]== Aggregate majority vote & CoT-Pass@K =========================="
evalhub cot finalize \
    --base-results "${BASE_RESULTS_FILE}" \
    --base-raw "${BASE_RAW_FILE}" \
    --judge-solutions "${judge_solutions}" \
    --output-dir "${JUDGE_DIR}" \
    --benchmark "${BENCHMARK}"

pipeline_log "[OK] CoT summary: ${JUDGE_DIR}/${BENCHMARK}_cot_summary.json"
pipeline_log "[OK] CoT results: ${JUDGE_DIR}/${BENCHMARK}_cot_results.jsonl"
