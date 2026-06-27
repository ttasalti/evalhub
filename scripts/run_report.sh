#!/usr/bin/env bash
# ============================================================================
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
# ============================================================================
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
    SCRIPT_DIR="${PROJECT_ROOT}/scripts"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "evalhub_env" ]]; then
    source /opt/Anaconda-2021.05/etc/profile.d/conda.sh
    conda activate evalhub_env
fi
export PATH="/user/home/t.tuna/.conda/envs/evalhub_env/bin:${PATH}"

# shellcheck source=lib/pipeline_common.sh
source "${SCRIPT_DIR}/lib/pipeline_common.sh"

pipeline_load_env "${1:-${EVALHUB_PIPELINE_ENV:-${SCRIPT_DIR}/configs/qwen_0.8b_demo.env}}"
apply_legacy_env_aliases
pipeline_init_paths

pipeline_log "==[REPORT]== Aggregating results + rendering plots/highlights/atlas under ${OUTPUT_ROOT}"
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
