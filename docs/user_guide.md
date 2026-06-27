# User guide — running the CoT-Pass@K pipeline end to end

A practical, copy-paste guide to running, sweeping, debugging, and extending the
pipeline on your own. New here? Start with the **Quick start** below; for the CSV
schema and the plot-reading manual, see [`reporting.md`](reporting.md).

## 0. Quick start (5 minutes)

From a fresh clone to a report on a single GPU, with a small open model and one
benchmark (~5–10 min once the model is cached):

```bash
# 1. Install (core + base eval + reporting; add ,sglang to serve via SGLang)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[base,report]"
huggingface-cli login           # only if your model/dataset is gated
evalhub --help                  # sanity check

# 2. A tiny base run: copy the demo env and shrink it
cp scripts/cot_pipeline.env.example scripts/demo.env
#   set in scripts/demo.env:
#     TARGET_MODEL="Qwen/Qwen2.5-0.5B-Instruct"   # ~6 GB GPU
#     BENCHMARK="gsm8k"
#     TARGET_N_SAMPLES=2
#     TARGET_MAX_COMPLETION_TOKENS=1024
scripts/run_eval_only.sh scripts/demo.env

# 3. Aggregate into the master CSV + plots
evalhub report aggregate --results-root ./results --output ./report.csv
evalhub report plot --csv ./report.csv --output-dir ./report_plots
```

`run_eval_only.sh` starts a vLLM server, resolves the chat template, runs
`evalhub gen` + `evalhub eval`, and tears the server down on exit. To also run the
judge stage, set `JUDGE_MODEL`/`JUDGE_TASK` and use `run_judge_only.sh` (§4) or the
full `run_end_to_end.sh` (§2). If a step fails, see **Debugging tips** (§8).

## 1. One-time setup

```bash
# Activate the project environment (conda or uv — see the README).
conda activate evalhub          # or: source .venv/bin/activate

# HuggingFace token (only needed for gated datasets/models):
# create a "read" token at HF → Settings → Access Tokens, then put
#   HF_TOKEN="hf_..."
# into scripts/secrets.env  (gitignored — never committed).
cp scripts/secrets.env.example scripts/secrets.env
```

## 2. A single run — three ways

```bash
# 2A. Fixed demo (everything in the env file) — ~3 benchmarks, a few minutes.
sbatch scripts/run_end_to_end.sh scripts/configs/qwen_0.8b_demo.env

# 2B. Same, but let submit.sh forward the env's SLURM_* knobs as sbatch flags.
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/qwen_0.8b_demo.env

# 2C. Dynamic — pick model + benchmark on the CLI, reuse one generic env file.
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/base.env \
    --model Qwen/Qwen3.5-0.8B-Base \
    --judge Qwen/Qwen3.5-0.8B \
    --benchmarks "aime2026 aime2026_tr aime2026_pt" \
    --temperature 0.6 --n-samples 64 --max-completion-tokens 20480 \
    --output-root results
```

Recognised CLI flags (precedence: **CLI args > env file > defaults**):

| Flag | Env var |
|---|---|
| `--model X` | `TARGET_MODEL` |
| `--judge X` | `JUDGE_MODEL` |
| `--benchmark X` / `--benchmarks "X Y"` | `BENCHMARK` / `BENCHMARKS` (looped) |
| `--target-state X` / `--judge-state X` | `TARGET_STATE` / `JUDGE_STATE` (`base\|non-think\|think`) |
| `--temperature N` / `--judge-temperature N` | `TARGET_TEMPERATURE` / `JUDGE_TEMPERATURE` |
| `--n-samples N` / `--judge-n-samples N` | `TARGET_N_SAMPLES` / `JUDGE_N_SAMPLES` |
| `--max-completion-tokens N` | `TARGET_MAX_COMPLETION_TOKENS` |
| `--output-root DIR` | `OUTPUT_ROOT` |
| `--set KEY=VAL` | any KEY=VAL (repeatable) |
| `-- ...` | passed straight through to `sbatch` |

`submit.sh` writes the flags into a throwaway `.overrides_<ts>_<pid>.env`
(gitignored) that the orchestrator sources *after* the base env, so values with
spaces/commas round-trip cleanly.

## 3. Monitoring a job

```bash
squeue -u $USER                                              # my queued/running jobs
sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed
tail -f logs/evalhub-e2e-<JOBID>.out                         # live stdout
tail -f logs/vllm_target_<JOBID>_aime2026.log               # live vLLM log (one per benchmark)
```

### Where results land (V5 layout — one folder per model)

The sampling suffix (`__t<T>__max<N>__n<NS>`) lives on the **benchmark leaf**, so
every model has a single folder; same tuple → same path (idempotent re-run),
different tuple → different leaf (no collision).

```
<OUTPUT_ROOT>/
├── report.csv                                  # aggregated long-form CSV
├── report_plots/                               # Pass@K / base-vs-CoT / heatmap / veto
└── base/                                        # <state>: base | non-think | think
    └── Qwen3.5-0.8B-Base/                       # one folder per model
        ├── aime2026__t0.6__max20480__n64/
        │   ├── aime2026.jsonl                   # generations
        │   ├── aime2026_raw.jsonl               # raw LLM responses
        │   ├── aime2026_results.jsonl           # per-task correct[] + counts
        │   └── aime2026_summary.json            # Pass@K, Cons@K, G-Pass@k, counts
        └── judged_by/Qwen3.5-0.8B__state-think__t0.6__max20480/
            └── aime2026__t0.6__max20480__n64/
                └── aime2026_cot_summary.json    # CoT-Pass@K after the judge veto
```

## 4. Turning results into a report

```bash
evalhub report aggregate --results-root ./results --output ./report.csv
evalhub report plot --csv ./report.csv --output-dir ./report_plots --format both
```

`report.csv` is one long-form row per `(model, mode, benchmark, judge)`. See
[`reporting.md`](reporting.md) for the schema and how to read each figure.

## 5. Multi-model / multi-benchmark / multi-temperature sweeps

```bash
# 5A. Lists in an env file (static sweep definition):
#   TARGET_MODELS="Qwen/Qwen3.5-0.8B-Base meta-llama/Llama-3.1-8B"
#   BENCHMARKS="aime2026 aime2026_tr math500"
#   TARGET_TEMPERATURES="0.6 0.9"
#   JUDGE_MODEL="Qwen/Qwen3.5-0.8B"
scripts/orchestrate.sh scripts/configs/my_sweep.env sequential

# 5B. Or the same lists on the CLI (the env file stays untouched):
scripts/orchestrate.sh scripts/configs/base.env sequential \
    --models "Qwen/Qwen3.5-0.8B-Base meta-llama/Llama-3.1-8B" \
    --benchmarks "aime2026 math500" \
    --judge Qwen/Qwen3.5-0.8B
```

`orchestrate.sh` submits one job per `(model, benchmark, temperature)` cell with
the right Slurm dependencies; `sequential` chains them so they don't contend for
the GPU.

## 6. Adding a model

```bash
# 1) New env config.
cp scripts/configs/qwen_0.8b_demo.env scripts/configs/<new>.env
#    edit TARGET_MODEL, JUDGE_MODEL, TARGET_STATE (base|non-think|think), OUTPUT_ROOT, SLURM_*

# 2) Check for a chat template.
ls scripts/templates/      # qwen3.5-*.jinja, gemma4-*.jinja, ministral3-*.jinja
#    If the model family is new, add a <family>-<state>.jinja and register the
#    family in evalhub/utils/model_state.py (MODEL_FAMILIES).

# 3) Submit.
sbatch scripts/run_end_to_end.sh scripts/configs/<new>.env
```

## 7. Adding a benchmark

Use an existing benchmark as a template (`evalhub/benchmarks/math/aime2026/`):

```python
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

NAME = "<benchmark_name>"
HUB = "owner/dataset"          # HF Hub ID, or None + a local CSV/parquet path

@register_dataset((NAME, HUB, True))   # True = evaluable
class MyDataset(MathDataset):
    def load_tasks(self) -> None:
        # self.add_task(Task(task_id=..., prompt=...))
        # self.add_groundtruth(GroundTruth(task_id=..., answer=...))
        ...
```

Then make the module importable and confirm it registers:

```bash
# add `from .<benchmark_name> import *` to evalhub/benchmarks/math/__init__.py
evalhub tasks | grep <benchmark_name>      # should appear in the list
```

Working examples: `aime2026_tr/`, `aime2026_pt/`, `math500/`, `gsm8k/`.

### Local CSV benchmark — the `tubitak_math2026` example

`evalhub/benchmarks/math/tubitak_math2026/` reads a local
`tubitak_math2026.csv` (no HF Hub). After editing the CSV, invalidate the cache
with `rm -rf ~/.cache/evalhub/`.

```bash
# Pass@K
evalhub gen --model hosted_vllm/Qwen/Qwen3.5-0.8B-Base --tasks tubitak_math2026 \
    --temperature 0.6 --n-samples 8 --output-dir results/tubitak/
evalhub eval --tasks tubitak_math2026 \
    --solutions results/tubitak/tubitak_math2026.jsonl --output-dir results/tubitak/

# CoT-Pass@K (Turkish benchmark → Turkish judge prompt: JUDGE_TASK=cot_judge_tr)
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/tubitak_math2026.env \
    --model Qwen/Qwen3.5-0.8B-Base --judge Qwen/Qwen3.5-0.8B
```

## 8. Debugging tips

- **Job FAILED, `.err` empty:** re-run the script directly with `bash -x` to get
  a shell trace, or read the per-benchmark vLLM log under `logs/`.
- **vLLM health timeout:** the model is still downloading or hit OOM. Raise
  `HEALTH_TIMEOUT`, lower `TARGET_PARALLEL_COUNT`, or use a smaller model.
- **`Missing required env vars`:** the env file wasn't loaded — pass it as `$1`
  or export `EVALHUB_PIPELINE_ENV=path/to/env`.
- **`evalhub gen` AuthenticationError:** export `HOSTED_VLLM_API_BASE` /
  `HOSTED_VLLM_API_KEY` (the orchestrators do this for you once vLLM is up).
- **Empty CoT summary:** the base run produced no `correct=True` generations —
  increase `TARGET_N_SAMPLES` or pick an easier benchmark.
- **`address already in use`:** another process holds `TARGET_PORT` — set
  `TARGET_PORT=30010` (or any free port) in the env file.
- **`report aggregate` returns 0 rows:** no directory under `--results-root`
  matched a recognised layout — confirm the dir names use the V5 layout (or a
  legacy fallback).
- **`ModuleNotFoundError: matplotlib` (on `report plot`):** install the report
  extra: `pip install -e ".[report]"`.

## 9. What each piece does (bird's-eye view)

| Path | Role |
|---|---|
| `scripts/run_eval_only.sh` | base generation + evaluation only |
| `scripts/run_judge_only.sh` | judge an existing base run, then finalize |
| `scripts/run_end_to_end.sh` | the full target → judge → CoT finalize → report job |
| `scripts/orchestrate.sh` | multi-cell DAG sweep submitter |
| `scripts/submit.sh` | single-job submitter with CLI overrides |
| `scripts/lib/pipeline_common.sh` | shared bash helpers (env loading, path composition, vLLM lifecycle) |
| `scripts/configs/*.env` | per-run knobs; `*.env.example` document them |
| `scripts/templates/*.jinja` | per-family chat templates |
| `evalhub/cot/` | CoT-Pass@K post-processing (extract → aggregate → metrics → finalize) |
| `evalhub/report/` | `evalhub report` aggregation + plots |
| `examples/scripts/` | reusable result-management tools (audit, migrate, tables) |
