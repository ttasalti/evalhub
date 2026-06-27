# Onboarding — From Zero to Report

Target audience: a developer who has just cloned the repository and wants a
working end-to-end demo before reading anything else.

The plan: install EvalHub, run base evaluation on a small open model + a
single benchmark, then aggregate the results into the master CSV and plots.
The whole walk-through targets a single GPU and finishes in ~5–10 minutes once
the model is cached locally.

## 1. Install

```bash
git clone <this-repo> evalhub
cd evalhub
uv venv --python 3.12
source .venv/bin/activate

# Core + base + reporting. Add `,sglang` if you intend to serve via SGLang.
uv pip install -e ".[base,report]"

# (optional) huggingface login if your target model is gated
huggingface-cli login
```

Sanity-check the CLI:

```bash
evalhub --help
evalhub report --help
```

## 2. Configure a tiny run

Copy the env template:

```bash
cp scripts/cot_pipeline.env.example scripts/demo.env
```

Edit `scripts/demo.env` and set just three values for the demo:

```bash
TARGET_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
BENCHMARK="gsm8k"
TARGET_N_SAMPLES=2                 # tiny — only for the demo
TARGET_MAX_COMPLETION_TOKENS=1024
```

Leave the rest at their defaults. (`Qwen/Qwen2.5-0.5B-Instruct` runs on a
single ~6 GB GPU; swap for `Qwen/Qwen3-7B-Instruct` if you have a 24 GB+
card.)

## 3. Run base evaluation only

```bash
scripts/run_eval_only.sh scripts/demo.env
```

What this does:

1. Starts a vLLM OpenAI server on `TARGET_PORT` (default `30000`).
2. Resolves a chat template via `python -m evalhub.utils.model_state`.
3. Polls `/health` until the server is ready (`HEALTH_TIMEOUT` seconds).
4. Calls `evalhub gen` to produce two solutions per task.
5. Calls `evalhub eval` to write `gsm8k_results.jsonl` + `gsm8k_summary.json`.
6. Tears the server down via a `trap` on script exit.

Output lands under
`./results/Qwen2.5-0.5B-Instruct_state-non-think_t0.6_max1024/gsm8k/`.

Quick spot-check:

```bash
evalhub view --results ./results/Qwen2.5-0.5B-Instruct_state-non-think_t0.6_max1024/gsm8k/gsm8k_results.jsonl --max-display 5
```

## 4. (Optional) Run the judge stage

If you also have a judge model handy (anything ≥7B works well):

```bash
cat >> scripts/demo.env <<'EOF'
JUDGE_MODEL="Qwen/Qwen3-7B-Instruct"
JUDGE_STATE="think"
JUDGE_TASK="cot_judge"
JUDGE_N_SAMPLES=3
EOF

# Hand the base run dir to run_judge_only.sh:
BASE_RESULTS_DIR="$PWD/results/Qwen2.5-0.5B-Instruct_state-non-think_t0.6_max1024/gsm8k" \
    scripts/run_judge_only.sh scripts/demo.env
```

The judge stage writes its outputs under `./results/judgments/...`.

## 5. Aggregate into the master CSV and plots

```bash
evalhub report aggregate --results-root ./results --output ./report.csv
evalhub report plot --csv ./report.csv --output-dir ./report_plots --format png
```

`report.csv` is one long-form row per `(model, mode, benchmark, judge)`;
`report_plots/` holds the Pass@K vs CoT-Pass@K figures. See
[`reporting.md`](reporting.md) for the full schema and
[`report_plots_guide.md`](report_plots_guide.md) for how to read every figure.

## 6. Submitting to nscluster (Slurm)

The orchestrator scripts have **no** embedded `#SBATCH` directives so they
remain portable. Wrap them in a tiny submission script:

```bash
cat > slurm_demo.sh <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=evalhub-demo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out

module load cuda/12.4
source ~/venvs/evalhub/bin/activate
scripts/run_eval_only.sh scripts/demo.env
EOF
sbatch slurm_demo.sh
```

Multi-benchmark sweep:

```bash
for bench in aime2024 aime2025 aime2026; do
    BENCHMARK="${bench}" sbatch slurm_demo.sh
done
```

## Troubleshooting

| Symptom | Most-likely cause / fix |
|---|---|
| `vLLM health timeout after 1800s on port 30000` | Model still downloading or OOM. Raise `HEALTH_TIMEOUT` or use a smaller model / lower `TARGET_PARALLEL_COUNT`. |
| `address already in use` | Another process is on `TARGET_PORT`. Set `TARGET_PORT=30010` (or any free port) in the env file. |
| `Missing required env vars: TARGET_MODEL BENCHMARK` | The env file wasn't loaded. Pass it as `$1` or export `EVALHUB_PIPELINE_ENV=path/to/env`. |
| `evalhub report aggregate` returns 0 rows | None of the directories under `--results-root` matched the canonical regex. Confirm directory names use `<model>_state-<state>_t<T>_max<N>` or the legacy fallback `<model>_t<T>_max<N>`. |
| `ModuleNotFoundError: matplotlib` (on `report plot`) | Install the report extra: `uv pip install -e ".[report]"`. |
| `JUDGE_INPUT` is empty → empty CoT summary | The base run produced no `correct=True` generations. Increase `TARGET_N_SAMPLES` or pick an easier benchmark. |

## What to read next

* [`docs/reporting.md`](reporting.md) — full CSV schema.
* [`scripts/README.md`](../scripts/README.md) — per-script env contract.
* [`docs/cmds.md`](cmds.md) — every `evalhub` CLI command, grouped by task.
* [`docs/tutorial.md`](tutorial.md) — adding a custom benchmark.
