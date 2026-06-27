#!/usr/bin/env bash
# ============================================================================
# scripts/run_end_to_end.sh
#
# Full CoT-Pass@K pipeline: base generation -> judge -> CoT finalization
# -> report aggregation + plots (when BENCHMARKS plural triggers the loop).
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
# Slurm (nscluster):
#   sbatch scripts/run_end_to_end.sh scripts/configs/qwen_0.8b_demo.env
#
#SBATCH --job-name=evalhub-e2e
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=12:00:00
# NOTE: no hard --nodelist pin — float across the partition so the job grabs the
# first free GPU on ANY node (nscluster/nsdl2/novasearchdl). A config that needs a
# specific node sets SLURM_NODELIST; submit.sh then passes --nodelist explicitly.
# The ROCR_VISIBLE_DEVICES unset above makes vLLM work on H200 nodes too.
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# ----------------------------------------------------------------------------
#
# Usage:
#   sbatch scripts/run_end_to_end.sh scripts/configs/qwen_0.8b_demo.env
#   bash   scripts/run_end_to_end.sh scripts/configs/qwen_0.8b_demo.env
#   scripts/run_end_to_end.sh --help
#
# Required env: TARGET_MODEL, JUDGE_MODEL, BENCHMARK (or BENCHMARKS)
# Optional env: every TARGET_*/JUDGE_* knob documented in
#               scripts/cot_pipeline.env.example, plus OUTPUT_ROOT, TARGET_PORT,
#               JUDGE_PORT, HEALTH_TIMEOUT, LOG_DIR.
# ============================================================================
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,45p' "$0"
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

# Activate the project conda environment and ensure its bin directory comes
# first in PATH (conda activate alone may not override ~/.local/bin).
if [[ "${CONDA_DEFAULT_ENV:-}" != "evalhub_env" ]]; then
    source /opt/Anaconda-2021.05/etc/profile.d/conda.sh
    conda activate evalhub_env
fi
export PATH="/user/home/t.tuna/.conda/envs/evalhub_env/bin:${PATH}"

# nsdl2 sets ROCR_VISIBLE_DEVICES alongside CUDA_VISIBLE_DEVICES; vLLM rejects both being set
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

# shellcheck source=lib/pipeline_common.sh
source "${SCRIPT_DIR}/lib/pipeline_common.sh"

pipeline_load_env "${1:-${EVALHUB_PIPELINE_ENV:-${SCRIPT_DIR}/configs/qwen_0.8b_demo.env}}"

# If BENCHMARKS (plural) is set and BENCHMARK is not, loop over each entry by
# re-invoking this script once per benchmark so every run gets a clean vLLM lifecycle.
if [[ -n "${BENCHMARKS:-}" && -z "${BENCHMARK:-}" ]]; then
    _env_arg="${1:-${EVALHUB_PIPELINE_ENV:-${SCRIPT_DIR}/configs/qwen_0.8b_demo.env}}"
    for _bm in ${BENCHMARKS}; do
        # Child runs handle their own REPORT stage; that produces a stale
        # OUTPUT_ROOT/report.csv after each one. The final aggregate below
        # overwrites it with the full set so the leftover state is harmless.
        BENCHMARK="${_bm}" EVALHUB_SKIP_REPORT=1 bash "${BASH_SOURCE[0]}" "${_env_arg}"
    done

    # All benchmarks finished — produce master CSV + plots once.
    apply_legacy_env_aliases
    pipeline_init_paths
    pipeline_log "==[REPORT]== Aggregating results + rendering plots/highlights/atlas ===="
    mkdir -p "${OUTPUT_ROOT}/report_plots"
    evalhub report aggregate \
        --results-root "${OUTPUT_ROOT}" \
        --output "${OUTPUT_ROOT}/report.csv"
    evalhub report plot \
        --csv "${OUTPUT_ROOT}/report.csv" \
        --output-dir "${OUTPUT_ROOT}/report_plots"
    evalhub report highlights \
        --csv "${OUTPUT_ROOT}/report.csv" \
        --output "${OUTPUT_ROOT}/report_highlights.pdf"
    evalhub report atlas \
        --plot-dir "${OUTPUT_ROOT}/report_plots" \
        --output "${OUTPUT_ROOT}/report_plots_atlas.pdf"
    pipeline_log "[DONE] Report CSV + plots + highlights + atlas written under ${OUTPUT_ROOT}"
    exit 0
fi

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
    "${LOG_DIR_LOCAL}/vllm_target_${SLURM_JOB_ID:-local}_${BENCHMARK}.log"
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
    pipeline_log "No correct base generations; CoT-Pass@K = 0 by definition."
    pipeline_write_empty_cot_summary "${JUDGE_DIR}" "${BENCHMARK}"
    # Fall through to Stage 4 so the empty-result run still contributes to
    # report.csv + plots; the empty summary's note column flags it.
    EVALHUB_SKIP_JUDGE=1
fi

if [[ -z "${EVALHUB_SKIP_JUDGE:-}" ]]; then
    start_vllm "${JUDGE_MODEL}" "${JUDGE_PORT}" "${JUDGE_PARALLEL_COUNT}" "${JUDGE_STATE}" \
        "${LOG_DIR_LOCAL}/vllm_judge_${SLURM_JOB_ID:-local}_${BENCHMARK}.log"
    export HOSTED_VLLM_API_BASE="http://127.0.0.1:${JUDGE_PORT}/v1"
    export HOSTED_VLLM_API_KEY="EMPTY"

    pipeline_run_judge_gen_eval "${JUDGE_DIR}" "${JUDGE_INPUT}" "${BENCHMARK}"
    judge_solutions="${JUDGE_SOLUTIONS_OUT}"
    stop_vllm

    # ----------------------------------------------------------------------
    # Stage 3 — aggregate majority vote, apply CoT veto, produce summary
    # ----------------------------------------------------------------------
    pipeline_log "==[3/3]== CoT-Pass@K aggregation ========================================"
    evalhub cot finalize \
        --base-results "${TARGET_DIR}/${BENCHMARK}_results.jsonl" \
        --base-raw "${TARGET_DIR}/${BENCHMARK}_raw.jsonl" \
        --judge-solutions "${judge_solutions}" \
        --output-dir "${JUDGE_DIR}" \
        --benchmark "${BENCHMARK}"

    pipeline_log "[DONE] CoT-Pass@K summary written under ${JUDGE_DIR}"
fi

# --------------------------------------------------------------------------
# Stage 4 — Aggregate report + plots so the OUTPUT_ROOT looks end-to-end
# complete after every run (single benchmark or final tail of a sweep).
# Skipped when invoked from the BENCHMARKS plural loop (parent runs its own).
# --------------------------------------------------------------------------
if [[ -z "${EVALHUB_SKIP_REPORT:-}" ]]; then
    pipeline_log "==[4/4]== Aggregating results + rendering plots/highlights/atlas ======"
    mkdir -p "${OUTPUT_ROOT}/report_plots"
    evalhub report aggregate \
        --results-root "${OUTPUT_ROOT}" \
        --output "${OUTPUT_ROOT}/report.csv"
    evalhub report plot \
        --csv "${OUTPUT_ROOT}/report.csv" \
        --output-dir "${OUTPUT_ROOT}/report_plots"
    evalhub report highlights \
        --csv "${OUTPUT_ROOT}/report.csv" \
        --output "${OUTPUT_ROOT}/report_highlights.pdf"
    evalhub report atlas \
        --plot-dir "${OUTPUT_ROOT}/report_plots" \
        --output "${OUTPUT_ROOT}/report_plots_atlas.pdf"
    pipeline_log "[DONE] Report CSV + plots + highlights + atlas written under ${OUTPUT_ROOT}"
fi
