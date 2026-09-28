# `scripts/`

Shell entry points around the `evalhub` CLI. The four scripts inherited from
upstream (`serve.sh`, `router.sh`, `eval_code.sh`, `r1_recipe.sh`) are kept as
they are; see `docs/baseline.md` for the R1 recipe they implement. Everything
else in this folder belongs to the CoT-Pass@K pipeline added by this fork:
env-driven orchestrators that start the vLLM servers, run the base and judge
stages and write the report, plus the configs and chat templates they use.

## Layout

| File | Purpose |
|---|---|
| `lib/pipeline_common.sh` | Shared bash library: env loading, template resolution, vLLM start/stop, default population, canonical output-path composition, and the high-level `pipeline_run_*` stage runners. Sourced by every orchestrator. |
| `run_eval_only.sh` | Stage 1 only: base generation + base evaluation. Produces `*_results.jsonl` + `*_summary.json` under the canonical layout. |
| `run_judge_only.sh` | Stages 2+3 over an existing base run: extract correct generations, run the judge LLM, majority-vote, CoT-Pass@K. |
| `run_end_to_end.sh` | All three stages + report, single Slurm job, base + judge + CoT finalize + report. |
| `run_report.sh` | Just the report stage (`evalhub report aggregate + plot`); used as the DAG tail. |
| `submit.sh` | Thin wrapper that reads `SLURM_*` from the env file and accepts CLI overrides (`--model`, `--benchmark`, `--judge`, ...). |
| `orchestrate.sh` | Multi-model × multi-benchmark × multi-temperature DAG submitter with dependency chains. |
| `cot_pipeline.env.example` | Annotated default values grouped by which scripts consume them. Copy to `cot_pipeline.env` and edit. |
| `configs/base.env` | Generic, model/benchmark-agnostic config; designed for CLI overrides. |
| `configs/qwen_0.8b_demo.env` | Concrete demo config: Qwen3.5-0.8B as solver and judge on the Portuguese exams, non-thinking mode, finishes in minutes. |
| `secrets.env.example` | Template for HF_TOKEN and similar secrets. Copy to `secrets.env` (gitignored). |
| `templates/` | Jinja chat templates per `(model_family, state)`. Selected by `evalhub.utils.model_state` and passed to `vllm serve --chat-template`. |

## Quick start

```bash
# Concrete demo: Qwen3.5-0.8B on the Portuguese exams (no edits needed):
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
| `run_judge_only.sh` | `JUDGE_MODEL`, `BENCHMARK`, `TARGET_MODEL`, and either `BASE_RESULTS_DIR` or `BASE_RESULTS_FILE`+`BASE_RAW_FILE` | every `JUDGE_*` sampling knob, `OUTPUT_ROOT`, `JUDGE_PORT`, `HEALTH_TIMEOUT` |
| `run_end_to_end.sh` | `TARGET_MODEL`, `JUDGE_MODEL`, `BENCHMARK` | every `TARGET_*` / `JUDGE_*` knob, ports, paths |

### Judge backend (`run_judge_only.sh`)

By default the judge model is served locally on a GPU (`JUDGE_BACKEND=vllm`).
With `JUDGE_BACKEND=api` the judge requests go to an external OpenAI-compatible
endpoint instead and no GPU is launched. In that mode also set `JUDGE_API_BASE`
and `JUDGE_API_KEY`; export the key or put it in the gitignored
`scripts/secrets.env`, and do not commit it. Only the transport changes: the
judge prompt, the verdicts and the metrics are identical.
[`configs/judge_api_deepseek.env`](configs/judge_api_deepseek.env) is a ready
single-generation DeepSeek setup.

## HPC (Slurm) usage

The scripts carry no `#SBATCH` directives, so they stay portable across HPC
sites. When a job does not inherit an activated environment, export
`EVALHUB_CONDA_SH` (the site's `conda.sh`) and `EVALHUB_CONDA_ENV` (default
`evalhub_env`), or `EVALHUB_ENV_BIN` (the environment's `bin/` directory)
before submitting. Either wrap them in a one-line `sbatch` invocation or copy
the recommended header from the docstring at the top of each script. An
example wrapper:

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

## After the run: aggregation & plots

Given one or more populated `OUTPUT_ROOT` directories, the `evalhub report`
sub-app aggregates every summary file into a master CSV and renders the static
plot suite. [`docs/reporting.md`](../docs/reporting.md) has the full
walk-through.
