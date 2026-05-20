#!/usr/bin/env bash
# ============================================================================
# scripts/run_eval_only.sh
#
# Run base generation + base evaluation for a single (model, benchmark).
# This is the first stage of the CoT-Pass@K pipeline; no judge is invoked.
#
# Output layout (matches the canonical scheme used by run_end_to_end.sh):
#   ${OUTPUT_ROOT}/<model>_state-<state>_t<T>_max<N>/<benchmark>/
#       <benchmark>.jsonl
#       <benchmark>_raw.jsonl
#       <benchmark>_results.jsonl
#       <benchmark>_summary.json
#
# ----------------------------------------------------------------------------
# Recommended Slurm header for nscluster (paste at the top of a wrapper job):
#   #SBATCH --job-name=evalhub-eval
#   #SBATCH --partition=gpu
#   #SBATCH --gres=gpu:1
#   #SBATCH --cpus-per-task=16
#   #SBATCH --mem=64G
#   #SBATCH --time=08:00:00
#   #SBATCH --output=logs/slurm-%j.out
# ----------------------------------------------------------------------------
#
# Usage:
#   scripts/run_eval_only.sh                    # uses $EVALHUB_PIPELINE_ENV
#   scripts/run_eval_only.sh path/to/eval.env   # explicit env file
#   scripts/run_eval_only.sh --help             # print env contract and exit
#
# Required env: TARGET_MODEL, BENCHMARK
# Optional env: TARGET_STATE, TARGET_TEMPERATURE, TARGET_TOP_P, TARGET_N_SAMPLES,
#               TARGET_MAX_COMPLETION_TOKENS, TARGET_NUM_WORKERS, TARGET_TIMEOUT,
#               TARGET_FREQUENCY_PENALTY, TARGET_PRESENCE_PENALTY, TARGET_STOP,
#               TARGET_SYSTEM_PROMPT, TARGET_OVERRIDE_ARGS, TARGET_TOOL_CONFIG,
#               TARGET_CALLBACK, TARGET_MAX_TURNS, TARGET_ENABLE_MULTITURN,
#               TARGET_RESUME, TARGET_PARALLEL_COUNT, OUTPUT_ROOT, TARGET_PORT,
#               HEALTH_TIMEOUT, LOG_DIR.
# ============================================================================
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,40p' "$0"
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=lib/pipeline_common.sh
source "${SCRIPT_DIR}/lib/pipeline_common.sh"

pipeline_load_env "${1:-${EVALHUB_PIPELINE_ENV:-}}"

require_env TARGET_MODEL BENCHMARK
apply_target_defaults
apply_common_defaults
pipeline_init_paths

TARGET_DIR="$(compose_target_dir "${BENCHMARK}")"
mkdir -p "${TARGET_DIR}"

pipeline_register_cleanup

pipeline_log "==[1/1]== Base generation & evaluation =================================="
start_vllm "${TARGET_MODEL}" "${TARGET_PORT}" "${TARGET_PARALLEL_COUNT}" "${TARGET_STATE}" \
    "${LOG_DIR_LOCAL}/vllm_target_${TARGET_PORT}.log"
export HOSTED_VLLM_API_BASE="http://127.0.0.1:${TARGET_PORT}/v1"
export HOSTED_VLLM_API_KEY="EMPTY"

pipeline_run_target_gen_eval "${TARGET_DIR}" "${BENCHMARK}"
stop_vllm

pipeline_log "[OK] Base results: ${TARGET_DIR}/${BENCHMARK}_results.jsonl"
pipeline_log "[OK] Base summary: ${TARGET_DIR}/${BENCHMARK}_summary.json"
