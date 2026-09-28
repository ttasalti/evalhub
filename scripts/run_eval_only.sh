#!/usr/bin/env bash
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
# Recommended Slurm header (paste at the top of a wrapper job):
#   #SBATCH --job-name=evalhub-eval
#   #SBATCH --partition=gpu
#   #SBATCH --gres=gpu:1
#   #SBATCH --cpus-per-task=16
#   #SBATCH --mem=64G
#   #SBATCH --time=08:00:00
#   #SBATCH --output=logs/slurm-%j.out
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
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,40p' "$0"
    exit 0
fi

# Under Slurm, BASH_SOURCE[0] points to the spool copy of the script, not the
# project tree. Use SLURM_SUBMIT_DIR (the directory sbatch was called from) as
# the authoritative project root instead.
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
require_env TARGET_MODEL BENCHMARK
apply_target_defaults
apply_common_defaults
pipeline_init_paths

TARGET_DIR="$(compose_target_dir "${BENCHMARK}")"
mkdir -p "${TARGET_DIR}"

pipeline_register_cleanup

pipeline_log "==[1/1]== Base generation & evaluation =================================="
start_vllm "${TARGET_MODEL}" "${TARGET_PORT}" "${TARGET_PARALLEL_COUNT}" "${TARGET_STATE}" \
    "${LOG_DIR_LOCAL}/vllm_target_${SLURM_JOB_ID:-local}_${BENCHMARK}.log"
export HOSTED_VLLM_API_BASE="http://127.0.0.1:${TARGET_PORT}/v1"
export HOSTED_VLLM_API_KEY="EMPTY"

pipeline_run_target_gen_eval "${TARGET_DIR}" "${BENCHMARK}"
stop_vllm

pipeline_log "[OK] Base results: ${TARGET_DIR}/${BENCHMARK}_results.jsonl"
pipeline_log "[OK] Base summary: ${TARGET_DIR}/${BENCHMARK}_summary.json"

if [[ -z "${EVALHUB_SKIP_REPORT:-}" ]]; then
    pipeline_log "==[report]== Upserting result row + refreshing plots =================="
    pipeline_run_report_incremental "${TARGET_DIR}/${BENCHMARK}_summary.json"
fi
