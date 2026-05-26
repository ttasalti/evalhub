#!/usr/bin/env bash
# ============================================================================
# scripts/run_end_to_end.sh
#
# Full CoT-Pass@K pipeline: base generation -> judge -> CoT finalization.
# This is the canonical replacement for the legacy run_cot_pass_at_k.sh; that
# file is now a thin deprecation shim that delegates here.
#
# Pipeline stages
#   1. start vLLM with the target model + its chat template
#   2. evalhub gen + evalhub eval  (standard Pass@K data)
#   3. shut target server down
#   4. evalhub cot extract         (filter correct generations)
#   5. start vLLM with the judge model
#   6. evalhub gen + evalhub eval  (judge yes/no per generation)
#   7. shut judge server down
#   8. evalhub cot finalize        (majority vote + CoT-Pass@K)
#
# All knobs come from an env file; pass it as $1 or set EVALHUB_PIPELINE_ENV.
# A defaults sample lives at scripts/cot_pipeline.env.example.
#
# ----------------------------------------------------------------------------
# Recommended Slurm header for nscluster (single GPU node):
#   #SBATCH --job-name=evalhub-e2e
#   #SBATCH --partition=gpu
#   #SBATCH --gres=gpu:2          # one for target, one for judge if same node
#   #SBATCH --cpus-per-task=32
#   #SBATCH --mem=128G
#   #SBATCH --time=24:00:00
#   #SBATCH --output=logs/slurm-%j.out
# ----------------------------------------------------------------------------
#
# Usage:
#   scripts/run_end_to_end.sh                    # uses $EVALHUB_PIPELINE_ENV
#   scripts/run_end_to_end.sh path/to/e2e.env    # explicit env file
#   scripts/run_end_to_end.sh --help             # print env contract and exit
#
# Required env: TARGET_MODEL, JUDGE_MODEL, BENCHMARK
# Optional env: every TARGET_*/JUDGE_* knob documented in
#               scripts/cot_pipeline.env.example, plus OUTPUT_ROOT, TARGET_PORT,
#               JUDGE_PORT, HEALTH_TIMEOUT, LOG_DIR.
# ============================================================================
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,45p' "$0"
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=lib/pipeline_common.sh
source "${SCRIPT_DIR}/lib/pipeline_common.sh"

pipeline_load_env "${1:-${EVALHUB_PIPELINE_ENV:-${SCRIPT_DIR}/cot_pipeline.env}}"

apply_legacy_env_aliases
require_env TARGET_MODEL JUDGE_MODEL BENCHMARK
apply_target_defaults
apply_judge_defaults
apply_common_defaults
pipeline_init_paths

TARGET_DIR="$(compose_target_dir "${BENCHMARK}")"
JUDGE_DIR="$(compose_judge_dir "${BENCHMARK}")"
mkdir -p "${TARGET_DIR}" "${JUDGE_DIR}"

pipeline_register_cleanup

# --------------------------------------------------------------------------
# Stage 1 — base generation + evaluation
# --------------------------------------------------------------------------
pipeline_log "==[1/3]== Base generation & evaluation =================================="
start_vllm "${TARGET_MODEL}" "${TARGET_PORT}" "${TARGET_PARALLEL_COUNT}" "${TARGET_STATE}" \
    "${LOG_DIR_LOCAL}/vllm_target_${TARGET_PORT}.log"
export HOSTED_VLLM_API_BASE="http://127.0.0.1:${TARGET_PORT}/v1"
export HOSTED_VLLM_API_KEY="EMPTY"

pipeline_run_target_gen_eval "${TARGET_DIR}" "${BENCHMARK}"
stop_vllm

# --------------------------------------------------------------------------
# Stage 2 — extract correct generations & run the judge
# --------------------------------------------------------------------------
pipeline_log "==[2/3]== Extract correct generations & judge ==========================="
JUDGE_INPUT="${JUDGE_DIR}/${BENCHMARK}_cot_judge_input.jsonl"
evalhub cot extract \
    --base-results "${TARGET_DIR}/${BENCHMARK}_results.jsonl" \
    --base-raw "${TARGET_DIR}/${BENCHMARK}_raw.jsonl" \
    --output "${JUDGE_INPUT}"

if [[ ! -s "${JUDGE_INPUT}" ]]; then
    pipeline_log "No correct base generations; CoT-Pass@K = 0 by definition. Stopping."
    pipeline_write_empty_cot_summary "${JUDGE_DIR}" "${BENCHMARK}"
    exit 0
fi

start_vllm "${JUDGE_MODEL}" "${JUDGE_PORT}" "${JUDGE_PARALLEL_COUNT}" "${JUDGE_STATE}" \
    "${LOG_DIR_LOCAL}/vllm_judge_${JUDGE_PORT}.log"
export HOSTED_VLLM_API_BASE="http://127.0.0.1:${JUDGE_PORT}/v1"
export HOSTED_VLLM_API_KEY="EMPTY"

judge_solutions="$(pipeline_run_judge_gen_eval "${JUDGE_DIR}" "${JUDGE_INPUT}")"
stop_vllm

# --------------------------------------------------------------------------
# Stage 3 — aggregate majority vote, apply CoT veto, produce summary
# --------------------------------------------------------------------------
pipeline_log "==[3/3]== CoT-Pass@K aggregation ========================================"
evalhub cot finalize \
    --base-results "${TARGET_DIR}/${BENCHMARK}_results.jsonl" \
    --base-raw "${TARGET_DIR}/${BENCHMARK}_raw.jsonl" \
    --judge-solutions "${judge_solutions}" \
    --output-dir "${JUDGE_DIR}" \
    --benchmark "${BENCHMARK}"

pipeline_log "[DONE] CoT-Pass@K summary written under ${JUDGE_DIR}"
