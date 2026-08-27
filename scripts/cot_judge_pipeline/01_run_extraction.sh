#!/bin/bash
#SBATCH --job-name=extract_corrects
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

ENV_FILE="scripts/configs/extractor.env"
if [[ -f "${ENV_FILE}" ]]; then
    source "${ENV_FILE}"
else
    echo "Error: ${ENV_FILE} not found." >&2
    exit 1
fi

# Parse SUFFIXES into an array, defaulting to "BASE" if undefined
IFS=' ' read -r -a SUFFIX_LIST <<< "${SUFFIXES:-BASE}"

for model in ${MODELS}; do
    for benchmark in ${BENCHMARKS}; do
        for suffix in "${SUFFIX_LIST[@]}"; do
            
            # Convert "BASE" keyword to empty string for the original folder
            actual_suffix="${suffix}"
            if [[ "${actual_suffix}" == "BASE" ]]; then
                actual_suffix=""
            fi

            python scripts/cot_judge_pipeline/01_extract_corrects.py \
                --model "${model}" \
                --benchmark "${benchmark}" \
                --suffix "${actual_suffix}"
        done
    done
done