#!/usr/bin/env bash
# scripts/run_report.sh
#
# Standalone report stage: walks OUTPUT_ROOT, builds the master CSV, and
# renders the static plot set. Designed to be submitted as the tail of a
# DAG (afterany:<all_judge_jobs>), but also runnable directly.
#
# Slurm:
#   sbatch scripts/run_report.sh scripts/configs/<config>.env
#   scripts/submit.sh scripts/run_report.sh scripts/configs/<config>.env
#
#SBATCH --job-name=evalhub-report
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail

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

# shellcheck source=lib/pipeline_common.sh
source "${SCRIPT_DIR}/lib/pipeline_common.sh"

pipeline_load_env "${1:-${EVALHUB_PIPELINE_ENV:-${SCRIPT_DIR}/configs/qwen_0.8b_demo.env}}"
apply_legacy_env_aliases
pipeline_init_paths

pipeline_log "==[REPORT]== Aggregating results + rendering plots under ${OUTPUT_ROOT}"
pipeline_run_report
