#!/usr/bin/env bash
# ============================================================================
# scripts/run_qwen_demo.sh
#
# Qwen3.5-0.8B-Base demo — full CoT-Pass@K for three benchmarks sequentially.
#   aime2026  →  aime2026_tr  →  aime2026_pt
#
# Results → results_demo/base/<model>_t0.6_max16384/<benchmark>/
#
# ----------------------------------------------------------------------------
# Slurm:
#   sbatch scripts/run_qwen_demo.sh
#
#SBATCH --job-name=qwen-demo
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --nodelist=nscluster
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# ----------------------------------------------------------------------------
#
# Usage:
#   scripts/run_qwen_demo.sh               # uses scripts/qwen_demo.env
#   scripts/run_qwen_demo.sh path/to.env   # explicit env file
# ============================================================================
set -euo pipefail

# Activate conda environment that provides vllm + evalhub
source /opt/Anaconda-2021.05/etc/profile.d/conda.sh
conda activate evalhub_env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ENV_FILE="${1:-${SCRIPT_DIR}/qwen_demo.env}"
BENCHMARKS=(aime2026 aime2026_tr aime2026_pt)

# Tee stdout+stderr to a timestamped run log (no-op inside Slurm since the
# #SBATCH --output/--error directives already capture everything).
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/qwen_demo_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "============================================================"
echo "  EvalHub — Qwen3.5-0.8B-Base Demo"
echo "  Started   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Env file  : ${ENV_FILE}"
echo "  Run log   : ${RUN_LOG}"
echo "  Benchmarks: ${BENCHMARKS[*]}"
echo "============================================================"

TOTAL=${#BENCHMARKS[@]}
IDX=0

for bm in "${BENCHMARKS[@]}"; do
    IDX=$((IDX + 1))
    echo ""
    echo "------------------------------------------------------------"
    echo "  [${IDX}/${TOTAL}] ${bm} — started $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"

    BENCHMARK="${bm}" bash "${SCRIPT_DIR}/run_end_to_end.sh" "${ENV_FILE}"

    echo "------------------------------------------------------------"
    echo "  [${IDX}/${TOTAL}] ${bm} — finished $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"
done

echo ""
echo "============================================================"
echo "  All ${TOTAL} benchmarks complete."
echo "  Finished  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
