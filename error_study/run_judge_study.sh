#!/usr/bin/env bash
# Stage 3c, run the DeepSeek v4-flash think-mode judge over the study's
# judge-input files with MAXIMAL prompt-caching.
#
# Caching is maximised on all three axes, per group (model x benchmark):
#   1. cross-variant  : `clean` is generated + committed FIRST, then the 4
#                       corruptions reuse the shared solution prefix (boxed_only
#                       ~99%, truncated ~75%, intermediate ~55%, consistent varies).
#   2. 3-generation   : round-major staging, every variant's gen-1 is committed
#                       before gen-2/gen-3, which re-send the identical prompt (~99%).
#   3. shared prefix  : same benchmark => identical template; same question =>
#                       identical question prefix (cached after the first commit).
# Per group there are only 3 commit-waits (not one per file), and the working set
# stays small (one group's clean solutions) so nothing is evicted before reuse.
#
# The full run SPENDS MONEY (see estimate_cost.py / run_demo.py). The API key is
# read from $JUDGE_API_KEY and never stored.
#
# Usage:
#   JUDGE_API_KEY=sk-... error_study/run_judge_study.sh            # full run
#   JUDGE_API_KEY=sk-... error_study/run_judge_study.sh --pilot    # cheapest group only
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # error_study/
OUT="${HERE}/output"
PY="${PYTHON:-python}"
EVALHUB="${EVALHUB_BIN:-evalhub}"

: "${JUDGE_API_KEY:?set JUDGE_API_KEY (DeepSeek), not stored anywhere}"
export HOSTED_VLLM_API_BASE="https://api.deepseek.com/v1"
export HOSTED_VLLM_API_KEY="${JUDGE_API_KEY}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

JUDGE_MODEL="deepseek-v4-flash"
NSAMPLES="${JUDGE_N_SAMPLES:-3}"
MAXTOK="${JUDGE_MAX_COMPLETION_TOKENS:-20480}"
TEMP="${JUDGE_TEMPERATURE:-0.6}"
TOPP="${JUDGE_TOP_P:-0.95}"
REASONING="${JUDGE_REASONING_EFFORT:-high}"
EXTRA_BODY="${JUDGE_EXTRA_BODY:-}"
[[ -z "${EXTRA_BODY}" ]] && EXTRA_BODY='{"thinking": {"type": "enabled"}}'
NUM_WORKERS="${JUDGE_NUM_WORKERS:-64}"
COMMIT_WAIT="${JUDGE_COMMIT_WAIT:-90}"

# one evalhub gen call: $1=input file  $2=task  $3=output dir  $4=n_samples  $5=resume(0/1)
gen_call() {
  local input="$1" task="$2" odir="$3" n="$4" resume="$5"
  mkdir -p "${odir}"
  # crash-resume across invocations: if the raw file already has data, TOP UP
  # (resume => "ab") instead of truncating it ("wb"); evalhub only generates the
  # shortfall (n_samples - existing per task), so re-running after a crash never
  # re-judges completed samples.
  [[ -s "${odir}/${task}_raw.jsonl" ]] && resume=1
  local extra=(); [[ "${resume}" == "1" ]] && extra=(--resume)
  "${EVALHUB}" gen \
    --model "hosted_vllm/${JUDGE_MODEL}" --tasks "${task}" --model-state think \
    --temperature "${TEMP}" --top-p "${TOPP}" --n-samples "${n}" \
    --num-workers "${NUM_WORKERS}" --max-completion-tokens "${MAXTOK}" \
    --reasoning-effort "${REASONING}" --extra-body "${EXTRA_BODY}" \
    --output-dir "${odir}" --override-args "{\"file_path\": \"${input}\"}" "${extra[@]}"
}

# returns 0 if <rawfile> already holds >= n samples for every task in <inputfile>
cell_done() {
  "${PY}" - "$1" "$2" "$3" <<'PY'
import sys, os, orjson
from collections import Counter
inp, raw, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
if not os.path.exists(raw):
    sys.exit(1)
need = {orjson.loads(l)["task_id"] for l in open(inp, "rb") if l.strip()}
have = Counter()
for l in open(raw, "rb"):
    if l.strip():
        have[orjson.loads(l)["task_id"]] += 1
sys.exit(0 if need and all(have.get(t, 0) >= n for t in need) else 1)
PY
}

# round-major, clean-first staging for ONE group. Args: base_output_dir, then the
# 5 "file|task|error_type" triples (clean must be first).
run_group() {
  local base_out="$1"; shift
  local jobs=("$@")
  # skip the whole group (and its commit-waits) if every cell is already complete
  local all_done=1
  for j in "${jobs[@]}"; do
    IFS='|' read -r rel task et <<<"${j}"
    cell_done "${OUT}/judge_inputs/${rel}" "${base_out}/${et}/${task}_raw.jsonl" "${NSAMPLES}" || { all_done=0; break; }
  done
  if [[ "${all_done}" == "1" ]]; then echo "  [skip: group already complete]"; return 0; fi
  local clean="" corrs=()
  for j in "${jobs[@]}"; do
    IFS='|' read -r rel task et <<<"${j}"
    if [[ "${et}" == "clean" ]]; then clean="${j}"; else corrs+=("${j}"); fi
  done
  # round 1: clean (commit) -> corruptions (reuse clean prefix)
  IFS='|' read -r rel task et <<<"${clean}"
  gen_call "${OUT}/judge_inputs/${rel}" "${task}" "${base_out}/${et}" 1 0
  echo "  [commit-wait ${COMMIT_WAIT}s: clean prefix]"; sleep "${COMMIT_WAIT}"
  for j in "${corrs[@]}"; do
    IFS='|' read -r rel task et <<<"${j}"
    gen_call "${OUT}/judge_inputs/${rel}" "${task}" "${base_out}/${et}" 1 0
  done
  # rounds 2..N: every variant re-sent (identical prompt => ~99% hit)
  local n
  for n in $(seq 2 "${NSAMPLES}"); do
    echo "  [commit-wait ${COMMIT_WAIT}s: round $((n - 1))]"; sleep "${COMMIT_WAIT}"
    for j in "${clean}" "${corrs[@]}"; do
      IFS='|' read -r rel task et <<<"${j}"
      gen_call "${OUT}/judge_inputs/${rel}" "${task}" "${base_out}/${et}" "${n}" 1
    done
  done
}

# emit "model<TAB>benchmark<TAB>rel|task|error_type" lines, clean first per group
mapfile -t LINES < <("${PY}" - "${OUT}/judge_inputs/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for r in sorted(m["files"], key=lambda r: (r["model"], r["benchmark"], r["run_order_index"])):
    print(f'{r["model"]}\t{r["benchmark"]}\t{r["file"]}|{r["judge_task"]}|{r["error_type"]}')
PY
)

PILOT=0; [[ "${1:-}" == "--pilot" ]] && PILOT=1

declare -A GROUPS_SEEN
cur=""; jobs=()
flush() {
  [[ -z "${cur}" ]] && return 0
  IFS='|' read -r model bench <<<"${cur}"
  echo "=== group ${model} / ${bench}  (round-major, clean-first)"
  run_group "${OUT}/judge/${model}/${bench}" "${jobs[@]}"
}
for line in "${LINES[@]}"; do
  IFS=$'\t' read -r model bench triple <<<"${line}"
  key="${model}|${bench}"
  if [[ "${PILOT}" == "1" && "${key}" != "Qwen3.5-4B|pt_exams_math" ]]; then continue; fi
  if [[ "${key}" != "${cur}" ]]; then flush; cur="${key}"; jobs=(); fi
  jobs+=("${triple}")
done
flush

# Auto-consolidate the per-cell raw into ONE file (from the SAME path we wrote to,
# so there is no --root mismatch) and summarise. The per-cell raw is KEPT (not
# --clean'd) so the raw judge responses stay directly viewable.
echo ">>> judging done, consolidating raw -> single file + summary ..."
"${PY}" -m error_study.consolidate --root "${OUT}/judge" --out "${OUT}/judge_output" \
  && "${PY}" -m error_study.analyze --input "${OUT}/judge_output.parquet"
echo ">>>"
echo ">>> RAW responses (kept): ${OUT}/judge/<model>/<benchmark>/<variant>/<task>_raw.jsonl"
echo ">>> SINGLE FILE:          ${OUT}/judge_output.parquet (+ .csv)"
echo ">>>     (metadata + judge_verdict + majority + judge_think [full reasoning] + judge_reasoning [answer] + usage)"
echo ">>> SUMMARY:              ${OUT}/cell_summary.csv , per_question_summary.csv"
echo ">>> COST:                 ${PY} -m error_study.estimate_cost measure ${OUT}/judge/*/*/*/*_raw.jsonl"
