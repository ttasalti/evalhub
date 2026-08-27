#!/bin/bash
#SBATCH --job-name=eval_stats
#SBATCH --output=logs/eval_stats_%j.out
#SBATCH --error=logs/eval_stats_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

BASE_DIR=${1:-"results"}
SCRIPT_PATH="scripts/utils/analyze_all_stats.py"

echo "Current Directory: $(pwd)" >&2
echo "Target Base Dir: $BASE_DIR" >&2
echo "Using Script: $SCRIPT_PATH" >&2

# Python scriptini çalıştırıyoruz, çıktıları ilgili klasörlere dağıtacak
python3 "$SCRIPT_PATH" --base_dir "$BASE_DIR"

echo "Execution Finished. Check individual run directories for *_generation_stats.jsonl files." >&2