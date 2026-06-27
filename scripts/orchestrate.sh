#!/usr/bin/env bash
# ============================================================================
# scripts/orchestrate.sh
#
# Multi-model × multi-benchmark × multi-temperature DAG submitter. Reads
# whitespace-separated lists from the env file OR from CLI overrides, then
# submits one base+judge pair per combination plus a tail report job.
#
# Usage:
#   scripts/orchestrate.sh <env_file> [DAG_MODE] [overrides...]
#
# DAG_MODE: sequential (default) | parallel
#
# Examples:
#   # Single model, multi-benchmark — env file holds the lists:
#   scripts/orchestrate.sh scripts/configs/qwen_0.8b_demo.env parallel
#
#   # Generic config + CLI sweep:
#   scripts/orchestrate.sh scripts/configs/base.env sequential \
#       --models "Qwen/Qwen3.5-0.8B-Base meta-llama/Llama-3.1-8B" \
#       --benchmarks "aime2026 math500" \
#       --temps "0.6 0.9" \
#       --judge Qwen/Qwen3.5-0.8B
#
# Recognized CLI overrides:
#   --models "A B"          → TARGET_MODELS
#   --benchmarks "X Y"      → BENCHMARKS
#   --temps "0.6 0.9"       → TARGET_TEMPERATURES
#   --judge X               → JUDGE_MODEL
#   --target-state X        → TARGET_STATE
#   --judge-state X         → JUDGE_STATE
#   --output-root DIR       → OUTPUT_ROOT
#   --set KEY=VAL           → KEY=VAL (free-form, repeatable)
#   --extra-dependency DEP  → slurm dependency string (e.g. "afterany:123:456")
#                             ANDed onto every base job's own --dependency in
#                             this DAG, in addition to any intra-DAG chaining.
#                             Use this to hard-chain an entire orchestrate.sh
#                             invocation after another (e.g. one checkpoint
#                             sweep must not start before a prior one finishes).
# ============================================================================
set -euo pipefail

if [[ $# -lt 1 ]]; then
    sed -n '2,35p' "$0" >&2
    exit 1
fi
env_file="$1"
shift

# Optional positional DAG mode arg.
dag_mode="sequential"
if [[ $# -gt 0 && "$1" != --* ]]; then
    dag_mode="$1"
    shift
fi

if [[ ! -f "${env_file}" ]]; then
    echo "[ERROR] env file not found: ${env_file}" >&2
    exit 1
fi
case "${dag_mode}" in
    sequential|parallel) ;;
    *) echo "[ERROR] DAG_MODE must be sequential or parallel, got '${dag_mode}'" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a

# Parse CLI overrides — these win over env values.
declare -A overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)        overrides[TARGET_MODELS]="$2"; shift 2 ;;
        --benchmarks)    overrides[BENCHMARKS]="$2"; shift 2 ;;
        --temps)         overrides[TARGET_TEMPERATURES]="$2"; shift 2 ;;
        --judge)         overrides[JUDGE_MODEL]="$2"; shift 2 ;;
        --judges)        overrides[JUDGE_MODELS]="$2"; shift 2 ;;
        --target-state)  overrides[TARGET_STATE]="$2"; shift 2 ;;
        --target-states) overrides[TARGET_STATES]="$2"; shift 2 ;;
        --judge-state)   overrides[JUDGE_STATE]="$2"; shift 2 ;;
        --output-root)   overrides[OUTPUT_ROOT]="$2"; shift 2 ;;
        --extra-dependency) EXTRA_DEPENDENCY="$2"; shift 2 ;;
        --set)
            kv="$2"
            if [[ "${kv}" != *=* ]]; then
                echo "[ERROR] --set expects KEY=VAL, got '${kv}'" >&2
                exit 1
            fi
            overrides[${kv%%=*}]="${kv#*=}"
            shift 2
            ;;
        --help|-h)
            sed -n '2,35p' "$0"
            exit 0
            ;;
        *)
            echo "[ERROR] unknown override: $1" >&2
            exit 1
            ;;
    esac
done

# Apply overrides locally so the loop below sees them.
for key in "${!overrides[@]}"; do
    declare "${key}=${overrides[$key]}"
done

# Fall back to singular variants if plurals not set.
TARGET_MODELS="${TARGET_MODELS:-${TARGET_MODEL:-}}"
BENCHMARKS="${BENCHMARKS:-${BENCHMARK:-}}"
TARGET_TEMPERATURES="${TARGET_TEMPERATURES:-${TARGET_TEMPERATURE:-0.6}}"
JUDGE_MODELS="${JUDGE_MODELS:-${JUDGE_MODEL:-}}"
TARGET_STATES="${TARGET_STATES:-${TARGET_STATE:-non-think}}"

if [[ -z "${TARGET_MODELS}" || -z "${BENCHMARKS}" || -z "${JUDGE_MODELS}" ]]; then
    echo "[ERROR] Required: TARGET_MODELS (or --models), BENCHMARKS (or --benchmarks), JUDGE_MODELS (or --judges/--judge)" >&2
    exit 1
fi

# Per-model SLURM resource tiering (Madde 6 H200/nsdl2). Empty values fall back
# to SLURM_CPUS_PER_TASK / SLURM_MEM from the env file.
slurm_tier() {
    # Usage: slurm_tier <model_name>; sets globals TIER_CPUS / TIER_MEM.
    # Uniform 40G host RAM (user directive); CPU tiered by model size.
    local m="$1"
    case "${m}" in
        *Qwen3.5-0.8B*|*Qwen3.5-2B*|*gemma-4-E2B*)  TIER_CPUS=8;  TIER_MEM=40G ;;
        *Qwen3.5-4B*|*gemma-4-E4B*)                  TIER_CPUS=8;  TIER_MEM=40G ;;
        *Qwen3.5-9B*)                                 TIER_CPUS=12; TIER_MEM=40G ;;
        *gemma-4-26B*)                                TIER_CPUS=16; TIER_MEM=40G ;;
        *Qwen3.6-35B*)                                TIER_CPUS=16; TIER_MEM=40G ;;
        *)                                            TIER_CPUS="${SLURM_CPUS_PER_TASK:-8}"; TIER_MEM="${SLURM_MEM:-40G}" ;;
    esac
}

# Per-model client-side concurrency tier (--num-workers passed to evalhub gen).
# Smaller models can absorb more parallel requests; larger ones saturate sooner.
num_workers_tier() {
    # Usage: num_workers_tier <model_name>; sets global TIER_NW.
    local m="$1"
    case "${m}" in
        *Qwen3.5-0.8B*)                               TIER_NW=512  ;;
        *Qwen3.5-2B*|*gemma-4-E2B*)                   TIER_NW=512  ;;
        *Qwen3.5-4B*|*gemma-4-E4B*)                   TIER_NW=512  ;;
        *Qwen3.5-9B*)                                 TIER_NW=384  ;;
        *gemma-4-26B*|*Qwen3.6-35B*)                  TIER_NW=256  ;;
        *)                                            TIER_NW="${TARGET_NUM_WORKERS:-256}" ;;
    esac
}

# Build sbatch CLI args from SLURM_* (same shape as submit.sh).
SBATCH_BASE_ARGS=()
[[ -n "${SLURM_JOB_NAME:-}" ]]      && SBATCH_BASE_ARGS+=(--job-name "${SLURM_JOB_NAME}")
[[ -n "${SLURM_GRES:-}" ]]          && SBATCH_BASE_ARGS+=(--gres "${SLURM_GRES}")
[[ -n "${SLURM_CPUS_PER_TASK:-}" ]] && SBATCH_BASE_ARGS+=(--cpus-per-task "${SLURM_CPUS_PER_TASK}")
[[ -n "${SLURM_MEM:-}" ]]           && SBATCH_BASE_ARGS+=(--mem "${SLURM_MEM}")
[[ -n "${SLURM_TIME:-}" ]]          && SBATCH_BASE_ARGS+=(--time "${SLURM_TIME}")
[[ -n "${SLURM_NODELIST:-}" ]]      && SBATCH_BASE_ARGS+=(--nodelist "${SLURM_NODELIST}")
[[ -n "${SLURM_PARTITION:-}" ]]     && SBATCH_BASE_ARGS+=(--partition "${SLURM_PARTITION}")

# Persist non-iterated overrides (JUDGE_MODEL, OUTPUT_ROOT, --set pairs) to
# an override file so child jobs see them via pipeline_load_env. Iterated
# vars (TARGET_MODEL/BENCHMARK/TARGET_TEMPERATURE) go via --export per job
# since they're single values with no whitespace.
override_file=""
has_shared_overrides=false
for key in "${!overrides[@]}"; do
    case "${key}" in
        TARGET_MODELS|BENCHMARKS|TARGET_TEMPERATURES) ;;
        *) has_shared_overrides=true ;;
    esac
done
if [[ "${has_shared_overrides}" == "true" ]]; then
    env_dir="$(cd "$(dirname "${env_file}")" && pwd)"
    override_file="${env_dir}/.overrides_$(date '+%Y%m%d_%H%M%S')_$$.env"
    {
        echo "# Auto-generated by scripts/orchestrate.sh at $(date)"
        echo "# Shared overrides for every job in this DAG. DO NOT COMMIT."
        for key in "${!overrides[@]}"; do
            case "${key}" in
                TARGET_MODELS|BENCHMARKS|TARGET_TEMPERATURES) continue ;;
            esac
            esc="${overrides[$key]//\'/\'\\\'\'}"
            printf "%s='%s'\n" "${key}" "${esc}"
        done
    } > "${override_file}"
    chmod 600 "${override_file}"
fi

echo "[orch] env file        : ${env_file}"
echo "[orch] DAG mode        : ${dag_mode}"
echo "[orch] TARGET_MODELS   : ${TARGET_MODELS}"
echo "[orch] BENCHMARKS      : ${BENCHMARKS}"
echo "[orch] TARGET_TEMPS    : ${TARGET_TEMPERATURES}"
echo "[orch] TARGET_STATES   : ${TARGET_STATES}"
echo "[orch] JUDGE_MODELS    : ${JUDGE_MODELS}"
echo "[orch] sbatch overrides: ${SBATCH_BASE_ARGS[*]:-(none)}"
echo "[orch] extra dependency: ${EXTRA_DEPENDENCY:-(none)}"

prev_judge_chain=""
all_judge_ids=()
all_job_ids=()

# Build SBATCH args minus per-job mem/cpus so we can override them per model tier.
SBATCH_BASE_NO_MEMCPU=()
skip_next=false
for arg in "${SBATCH_BASE_ARGS[@]}"; do
    if $skip_next; then skip_next=false; continue; fi
    case "${arg}" in
        --cpus-per-task|--mem) skip_next=true ;;
        *) SBATCH_BASE_NO_MEMCPU+=("${arg}") ;;
    esac
done

for STATE in ${TARGET_STATES}; do
    echo
    echo "[orch] ###################################"
    echo "[orch] ## TARGET_STATE = ${STATE}"
    echo "[orch] ###################################"
    for MODEL in ${TARGET_MODELS}; do
        for BM in ${BENCHMARKS}; do
            for TEMP in ${TARGET_TEMPERATURES}; do
                run_id="$(basename "${MODEL}")_${BM}_t${TEMP}_${STATE}"
                echo
                echo "[orch] === ${run_id} ==="

                slurm_tier "${MODEL}";        target_cpus="${TIER_CPUS}"; target_mem="${TIER_MEM}"
                num_workers_tier "${MODEL}";  target_nw="${TIER_NW}"

                exports="TARGET_MODEL=${MODEL},BENCHMARK=${BM},TARGET_TEMPERATURE=${TEMP},TARGET_STATE=${STATE},TARGET_NUM_WORKERS=${target_nw}"
                if [[ -n "${override_file}" ]]; then
                    exports+=",EVALHUB_OVERRIDES_FILE=${override_file}"
                fi

                base_dep_args=()
                base_deps=()
                if [[ "${dag_mode}" == "sequential" && -n "${prev_judge_chain}" ]]; then
                    base_deps+=("afterany:${prev_judge_chain}")
                fi
                if [[ -n "${EXTRA_DEPENDENCY:-}" ]]; then
                    base_deps+=("${EXTRA_DEPENDENCY}")
                fi
                if (( ${#base_deps[@]} > 0 )); then
                    base_dep_args=(--dependency="$(IFS=,; echo "${base_deps[*]}")")
                fi
                base_job=$(sbatch --parsable \
                    "${SBATCH_BASE_NO_MEMCPU[@]}" \
                    --cpus-per-task="${target_cpus}" --mem="${target_mem}" \
                    "${base_dep_args[@]}" \
                    --job-name="base_${run_id}" \
                    --export=ALL,${exports} \
                    "${SCRIPT_DIR}/run_eval_only.sh" "${env_file}")
                echo "[orch] base job  (cpu=${target_cpus} mem=${target_mem} nw=${target_nw}) -> ${base_job}"
                all_job_ids+=("${base_job}")

                # NOTE: the base output dir is deliberately NOT pre-computed here.
                # run_judge_only.sh derives it itself via compose_target_dir()
                # (V4, state-aware), so the judge always reads the base for the
                # exact (target, state, benchmark) — never another state's base.
                # (A previous hand-rolled "_think-true/false" path here cross-wired
                # an instruct judge onto the think base.)

                # One judge job per JUDGE_MODELS entry, all depending on the same
                # base job. Next iteration's base depends on ALL judges finishing.
                judge_ids_this_run=()
                for JUDGE_M in ${JUDGE_MODELS}; do
                    slurm_tier "${JUDGE_M}";       judge_cpus="${TIER_CPUS}"; judge_mem="${TIER_MEM}"
                    num_workers_tier "${JUDGE_M}"; judge_nw="${TIER_NW}"
                    judge_run_id="${run_id}__by_$(basename "${JUDGE_M}")"
                    judge_exports="${exports},JUDGE_MODEL=${JUDGE_M},JUDGE_NUM_WORKERS=${judge_nw}"
                    judge_job=$(sbatch --parsable \
                        "${SBATCH_BASE_NO_MEMCPU[@]}" \
                        --cpus-per-task="${judge_cpus}" --mem="${judge_mem}" \
                        --time="${JUDGE_SLURM_TIME:-08:00:00}" \
                        --dependency="afterany:${base_job}" \
                        --job-name="judge_${judge_run_id}" \
                        --export=ALL,${judge_exports} \
                        "${SCRIPT_DIR}/run_judge_only.sh" "${env_file}")
                    echo "[orch] judge job (cpu=${judge_cpus} mem=${judge_mem} nw=${judge_nw} judge=${JUDGE_M}) -> ${judge_job}"
                    judge_ids_this_run+=("${judge_job}")
                    all_judge_ids+=("${judge_job}")
                    all_job_ids+=("${judge_job}")
                done

                # Chain next base on ALL judges from this run.
                prev_judge_chain=$(IFS=:; echo "${judge_ids_this_run[*]}")
            done
        done
    done
done

if (( ${#all_judge_ids[@]} > 0 )); then
    judge_dep=$(IFS=:; echo "${all_judge_ids[*]}")
    report_exports="ALL"
    if [[ -n "${override_file}" ]]; then
        report_exports="ALL,EVALHUB_OVERRIDES_FILE=${override_file}"
    fi
    report_job=$(sbatch --parsable \
        --dependency="afterany:${judge_dep}" \
        --job-name="report_dag" \
        --export="${report_exports}" \
        "${SCRIPT_DIR}/run_report.sh" "${env_file}")
    echo
    echo "[orch] report job -> ${report_job} (afterany:${judge_dep})"
fi

echo
echo "[orch] ALL_JOB_IDS: $(IFS=' '; echo "${all_job_ids[*]}")"
echo "[orch] DAG submitted. Monitor with: squeue -u \$USER"
