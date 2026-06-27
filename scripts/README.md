# `scripts/`

**Modified from the evalhub-original repository.**

The folder was reset to evalhub-original-style minimalism. All legacy Slurm
orchestrators and helper sub-pipelines (`vllm*.sh`, `judge_vllm3.sh`,
`judge_only_all.sh`, `cot_judge_pipeline/`, `pass_k_pipeline/`,
`configs/`, `utils/`) were deleted because their logic now lives inside the
Python package and is exercised via `evalhub` CLI commands.

## Layout

| File | Purpose |
|---|---|
| `lib/pipeline_common.sh` | **Shared bash library** — env loading, template resolution, vLLM start/stop, default population, canonical output-path composition, and the high-level `pipeline_run_*` stage runners. Sourced by every orchestrator. |
| `run_eval_only.sh` | Stage 1 only: base generation + base evaluation. Produces `*_results.jsonl` + `*_summary.json` under the canonical layout. |
| `run_judge_only.sh` | Stages 2+3 over an existing base run: extract correct generations, run the judge LLM, majority-vote, CoT-Pass@K. |
| `run_end_to_end.sh` | All three stages + report — single Slurm job, base + judge + CoT finalize + report. |
| `run_report.sh` | Just the report stage (`evalhub report aggregate + plot`); used as the DAG tail. |
| `submit.sh` | Thin wrapper that reads `SLURM_*` from the env file and accepts CLI overrides (`--model`, `--benchmark`, `--judge`, ...). |
| `orchestrate.sh` | Multi-model × multi-benchmark × multi-temperature DAG submitter with dependency chains. |
| `cot_pipeline.env.example` | Annotated default values grouped by which scripts consume them. Copy to `cot_pipeline.env` and edit. |
| `configs/base.env` | Generic, model/benchmark-agnostic config; designed for CLI overrides. |
| `configs/qwen_0.8b_demo.env` | Concrete demo config (Qwen 0.8B + 3 AIME benchmarks). |
| `secrets.env.example` | Template for HF_TOKEN and similar secrets. Copy to `secrets.env` (gitignored). |
| `templates/` | Jinja chat templates per `(model_family, state)`. Selected by `evalhub.utils.model_state` and passed to `vllm serve --chat-template`. |

## Quick start

```bash
# Concrete demo — Qwen 0.8B on 3 AIME benchmarks (no edits needed):
sbatch scripts/run_end_to_end.sh scripts/configs/qwen_0.8b_demo.env

# Pick model + benchmark dynamically with a generic config + CLI overrides:
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/base.env \
    --model Qwen/Qwen3.5-0.8B-Base \
    --judge Qwen/Qwen3.5-0.8B \
    --benchmarks "aime2026 aime2026_tr aime2026_pt"

# Multi-model sweep:
scripts/orchestrate.sh scripts/configs/base.env sequential \
    --models "A B" --benchmarks "x y" --judge Z
```

Each script also supports `--help` for an inline env-var contract:

```bash
scripts/run_eval_only.sh --help
scripts/run_judge_only.sh --help
scripts/run_end_to_end.sh --help
```

## Required env per script

| Script | Required env | Notable optional env |
|---|---|---|
| `run_eval_only.sh`  | `TARGET_MODEL`, `BENCHMARK` | every `TARGET_*` sampling knob, `OUTPUT_ROOT`, `TARGET_PORT`, `HEALTH_TIMEOUT` |
| `run_judge_only.sh` | `JUDGE_MODEL`, `BENCHMARK`, `TARGET_MODEL`, and **either** `BASE_RESULTS_DIR` **or** `BASE_RESULTS_FILE`+`BASE_RAW_FILE` | every `JUDGE_*` sampling knob, `OUTPUT_ROOT`, `JUDGE_PORT`, `HEALTH_TIMEOUT` |
| `run_end_to_end.sh` | `TARGET_MODEL`, `JUDGE_MODEL`, `BENCHMARK` | every `TARGET_*` / `JUDGE_*` knob, ports, paths |

## HPC / nscluster (Slurm) usage

The scripts are intentionally **environment-agnostic** — they do not embed
`#SBATCH` directives so they remain portable across HPC sites. Wrap them in
a one-line `sbatch` invocation, or copy the recommended header from the
docstring at the top of each script. Example wrapper for nscluster:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=evalhub-e2e
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out

module load cuda/12.4
source ~/venvs/evalhub/bin/activate
scripts/run_end_to_end.sh scripts/cot_pipeline.env
```

For multi-benchmark sweeps, submit one Slurm job per benchmark with an env
override:

```bash
for bench in aime2024 aime2025 aime2026; do
    BENCHMARK="${bench}" sbatch slurm_wrapper.sh
done
```

## After the run — aggregation & plots

Once you have one or more populated `OUTPUT_ROOT` directories, the
`evalhub report` sub-app aggregates every summary file into a master CSV and
renders the static plot suite. See [`docs/reporting.md`](../docs/reporting.md)
for the full walk-through.
