#!/usr/bin/env bash
# ============================================================================
# scripts/submit.sh
#
# sbatch wrapper with two jobs:
#   1. Source <env_file> to pick up SLURM_* allocation knobs.
#   2. Parse CLI overrides (--model, --benchmark, --judge, ...) and persist
#      them to a per-job override file. The orchestrator's
#      pipeline_load_env() sources both the main env file and this override
#      file (via EVALHUB_OVERRIDES_FILE), so quoting and whitespace are safe.
#   3. Submit <orchestrator.sh> with sbatch CLI args + --export of the
#      override file path.
#
# Why an override file instead of sbatch --export=KEY=VAL ?
#   Slurm's --export comma-parses on raw commas; values containing whitespace
#   or commas (e.g. BENCHMARKS="aime2026 aime2026_tr") corrupt the export
#   list. A sourced env file dodges all that.
#
# Usage:
#   scripts/submit.sh <orchestrator.sh> <env_file> [overrides...] [-- sbatch_extra...]
#
# Examples:
#   # Use baked env values:
#   scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/qwen_0.8b_demo.env
#
#   # Pick model + benchmark dynamically with a generic config:
#   scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/base.env \
#       --model Qwen/Qwen3.5-0.8B-Base \
#       --judge Qwen/Qwen3.5-0.8B \
#       --benchmarks "aime2026 aime2026_tr aime2026_pt"
#
# Recognized CLI overrides (CLI > env > defaults):
#   --model X                    → TARGET_MODEL=X
#   --judge X                    → JUDGE_MODEL=X
#   --benchmark X                → BENCHMARK=X
#   --benchmarks "X Y Z"         → BENCHMARKS="X Y Z"  (loop in run_end_to_end.sh)
#   --target-state X             → TARGET_STATE=X      (base|non-think|think)
#   --judge-state X              → JUDGE_STATE=X
#   --output-root DIR            → OUTPUT_ROOT=DIR
#   --temperature N              → TARGET_TEMPERATURE=N
#   --judge-temperature N        → JUDGE_TEMPERATURE=N
#   --n-samples N                → TARGET_N_SAMPLES=N
#   --judge-n-samples N          → JUDGE_N_SAMPLES=N
#   --max-completion-tokens N    → TARGET_MAX_COMPLETION_TOKENS=N
#   --set KEY=VAL                → KEY=VAL (free-form passthrough; repeatable)
#
# Everything after `--` is passed verbatim to sbatch.
# ============================================================================
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,55p' "$0" >&2
    exit 1
fi

orchestrator="$1"
env_file="$2"
shift 2

if [[ ! -f "${orchestrator}" ]]; then
    echo "[ERROR] orchestrator not found: ${orchestrator}" >&2
    exit 1
fi
if [[ ! -f "${env_file}" ]]; then
    echo "[ERROR] env file not found: ${env_file}" >&2
    exit 1
fi

# Source env file so SLURM_* are available here. Pipeline knobs go through
# the override file mechanism (see below), not direct export, so the
# orchestrator's own pipeline_load_env stays the source of truth.
set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a

# Parse CLI overrides into an associative array and sbatch extras.
declare -A overrides=()
sbatch_extras=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                  overrides[TARGET_MODEL]="$2"; shift 2 ;;
        --judge)                  overrides[JUDGE_MODEL]="$2"; shift 2 ;;
        --benchmark)              overrides[BENCHMARK]="$2"; shift 2 ;;
        --benchmarks)             overrides[BENCHMARKS]="$2"; shift 2 ;;
        --target-state)           overrides[TARGET_STATE]="$2"; shift 2 ;;
        --judge-state)            overrides[JUDGE_STATE]="$2"; shift 2 ;;
        --output-root)            overrides[OUTPUT_ROOT]="$2"; shift 2 ;;
        --temperature)            overrides[TARGET_TEMPERATURE]="$2"; shift 2 ;;
        --judge-temperature)      overrides[JUDGE_TEMPERATURE]="$2"; shift 2 ;;
        --n-samples)              overrides[TARGET_N_SAMPLES]="$2"; shift 2 ;;
        --judge-n-samples)        overrides[JUDGE_N_SAMPLES]="$2"; shift 2 ;;
        --max-completion-tokens)  overrides[TARGET_MAX_COMPLETION_TOKENS]="$2"; shift 2 ;;
        --set)
            local_kv="$2"
            if [[ "${local_kv}" != *=* ]]; then
                echo "[ERROR] --set expects KEY=VAL, got '${local_kv}'" >&2
                exit 1
            fi
            overrides[${local_kv%%=*}]="${local_kv#*=}"
            shift 2
            ;;
        --)
            shift
            sbatch_extras=("$@")
            break
            ;;
        --help|-h)
            sed -n '2,55p' "$0"
            exit 0
            ;;
        *)
            echo "[ERROR] unknown override: $1" >&2
            echo "       use '--set KEY=VAL' for arbitrary env vars" >&2
            exit 1
            ;;
    esac
done

# Build sbatch CLI args from SLURM_*.
sbatch_args=()
[[ -n "${SLURM_JOB_NAME:-}" ]]      && sbatch_args+=(--job-name "${SLURM_JOB_NAME}")
[[ -n "${SLURM_GRES:-}" ]]          && sbatch_args+=(--gres "${SLURM_GRES}")
[[ -n "${SLURM_CPUS_PER_TASK:-}" ]] && sbatch_args+=(--cpus-per-task "${SLURM_CPUS_PER_TASK}")
[[ -n "${SLURM_MEM:-}" ]]           && sbatch_args+=(--mem "${SLURM_MEM}")
[[ -n "${SLURM_TIME:-}" ]]          && sbatch_args+=(--time "${SLURM_TIME}")
[[ -n "${SLURM_NODELIST:-}" ]]      && sbatch_args+=(--nodelist "${SLURM_NODELIST}")
[[ -n "${SLURM_PARTITION:-}" ]]     && sbatch_args+=(--partition "${SLURM_PARTITION}")

if [[ -n "${SLURM_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    sbatch_args+=( ${SLURM_EXTRA_ARGS} )
fi

# Persist overrides to a stable, gitignored override file. We pick a path
# under the env file's directory so all derived files live together (logs,
# secrets.env, overrides.env). The orchestrator's pipeline_load_env() picks
# up EVALHUB_OVERRIDES_FILE and sources it after the main env file.
override_file=""
if (( ${#overrides[@]} > 0 )); then
    env_dir="$(cd "$(dirname "${env_file}")" && pwd)"
    override_file="${env_dir}/.overrides_$(date '+%Y%m%d_%H%M%S')_$$.env"
    {
        echo "# Auto-generated by scripts/submit.sh at $(date)"
        echo "# DO NOT COMMIT. Reflects CLI overrides for one sbatch invocation."
        for key in "${!overrides[@]}"; do
            # Use single-quotes around the value to preserve whitespace and
            # most special chars. Escape any embedded single quote.
            esc="${overrides[$key]//\'/\'\\\'\'}"
            printf "%s='%s'\n" "${key}" "${esc}"
        done
    } > "${override_file}"
    chmod 600 "${override_file}"
    sbatch_args+=(--export="ALL,EVALHUB_OVERRIDES_FILE=${override_file}")
fi

echo "[submit] orchestrator   : ${orchestrator}"
echo "[submit] env file       : ${env_file}"
echo "[submit] sbatch args    : ${sbatch_args[*]:-(none, header defaults)}"
if (( ${#overrides[@]} > 0 )); then
    echo "[submit] override file  : ${override_file}"
    for key in "${!overrides[@]}"; do
        printf '   %s=%s\n' "${key}" "${overrides[$key]}"
    done
fi
if (( ${#sbatch_extras[@]} > 0 )); then
    echo "[submit] sbatch extras  : ${sbatch_extras[*]}"
fi

exec sbatch "${sbatch_args[@]}" "${sbatch_extras[@]}" "${orchestrator}" "${env_file}"
