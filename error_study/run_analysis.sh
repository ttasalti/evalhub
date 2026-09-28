#!/usr/bin/env bash
# error_study/run_analysis.sh: reproduce the full analysis chain for a
# judge run produced by run_judge_local.sh. CPU only: no GPU is held while
# pandas chews through the parquet.
#
#   sbatch error_study/run_analysis.sh <run_tag>       # e.g. qwen35b | r1_8b
#
# Inputs :  error_study/output/judge_<tag>/**/<task>_raw.jsonl
#           error_study/output/error_dataset.parquet   (metadata join)
# Outputs:  error_study/output/judge_<tag>_output.{parquet,csv}
#           error_study/output/analysis_<tag>/{cell_summary,per_question_summary,
#                                              report_error_tasks,report_error_questions}.csv
#
# --outdir is MANDATORY on analyze/report_error: their default is dirname(input),
# which is output/ and would overwrite the V4-Flash reference CSVs there.
#SBATCH --job-name=es_analysis
#SBATCH --partition=cpu_only
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

RUN_TAG="${1:?usage: run_analysis.sh <run_tag>}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

# Activate the project environment when the job does not inherit it (see
# scripts/cot_pipeline.env.example for EVALHUB_CONDA_SH / EVALHUB_ENV_BIN).
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
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
PY="${PYTHON:-python}"

OUT="error_study/output"
ROOT="${OUT}/judge_${RUN_TAG}"
BASE="${OUT}/judge_${RUN_TAG}_output"
ADIR="${OUT}/analysis_${RUN_TAG}"
mkdir -p "${ADIR}"

[[ -d "${ROOT}" ]] || { echo "ERROR: judge run root missing: ${ROOT}" >&2; exit 1; }

echo "=== [1/4] consolidate raw -> single file ==="
${PY} -m error_study.consolidate --root "${ROOT}" --out "${BASE}"

echo "=== [2/4] analyze (cell_summary + per_question_summary + threshold pivots) ==="
${PY} -m error_study.analyze --input "${BASE}.parquet" --outdir "${ADIR}"

echo "=== [3/4] report_error (task- and question-grain CSVs) ==="
${PY} -m error_study.report_error --input "${BASE}.parquet" --outdir "${ADIR}"

echo "=== [4/4] token statistics ==="
${PY} -m error_study.estimate_cost measure "${ROOT}"/*/*/*/*_raw.jsonl || true

echo
echo "=== shape check against the V4-Flash reference run ==="
${PY} - "${BASE}.parquet" "${ADIR}" <<'PY'
import sys, os, pandas as pd
base, adir = sys.argv[1], sys.argv[2]
df = pd.read_parquet(base, columns=["row_id", "error_type", "judge_verdict", "judge_think"])
n_gen, n_var = len(df), df.groupby(["row_id", "error_type"]).ngroups
think_pct = 100.0 * (df.judge_think.fillna("").str.len() > 0).mean()
inv_pct = 100.0 * (df.judge_verdict == "invalid").mean()
print(f"generations      : {n_gen}      (reference: 21345)")
print(f"solution-variants: {n_var}      (reference: 7115)")
print(f"judge_think set  : {think_pct:.1f}%  (reference: 100.0%)  <- think-mode evidence")
print(f"invalid verdict  : {inv_pct:.2f}%   (reference: 0.31%)")
print("verdict counts   :", df.judge_verdict.value_counts().to_dict())
for f, exp in [("cell_summary.csv", 50), ("report_error_tasks.csv", 7115),
               ("report_error_questions.csv", 2506)]:
    p = os.path.join(adir, f)
    n = len(pd.read_csv(p)) if os.path.exists(p) else -1
    print(f"{f:30s} {n:6d} rows (reference: {exp})", "OK" if n == exp else "<-- DIFFERS")
PY

echo
echo "=== reference files unchanged? ==="
md5sum "${OUT}/judge_output.parquet" "${OUT}/cell_summary.csv" \
       "${OUT}/per_question_summary.csv" "${OUT}/report_error_tasks.csv" \
       "${OUT}/report_error_questions.csv" 2>/dev/null || true

echo
echo "DONE -> ${BASE}.parquet (+ .csv) , ${ADIR}/"
