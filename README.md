# 🔮 EvalHub

<p align="center">
    <a href="https://github.com/ttasalti/evalhub"><img src="https://img.shields.io/badge/Eval-Hub-blue.svg"></a>
    <a href="https://github.com/ttasalti/evalhub/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
    <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

<p align="center">
    <a href="#-about">📖 About</a> •
    <a href="#-paper">📝 Paper</a> •
    <a href="#-features">✨ Features</a> •
    <a href="#-installation">📦 Installation</a> •
    <a href="#-quick-start">🚀 Quick Start</a> •
    <a href="#-cot-passk-pipeline">🧠 CoT-Pass@K Pipeline</a> •
    <a href="#-reporting">📊 Reporting</a> •
    <a href="#-development">🛠 Development</a> •
    <a href="#-roadmap">🛣 Roadmap</a> •
    <a href="#-status">🚧 Status</a>
</p>

## 📖 About

All-in-one benchmarking platform for evaluating Large Language Models (LLMs) with comprehensive metrics and standardized testing frameworks.

> [!Note]
> This repository is a fork of [`ysy-phoenix/evalhub`](https://github.com/ysy-phoenix/evalhub).
> The upstream harness measures Pass@K only: whether a model reaches a correct
> answer, never how it got there. This fork adds the judged variant on top of it.
> A CoT-Pass@K pipeline re-assesses every base-correct generation with a stronger
> judge LLM and majority-votes the verdicts to keep or veto the original label.
> The judge can run locally with vLLM (SGLang is also supported) for long-context
> judging, or through a cached API path for hosted models. Five mathematics
> benchmarks are added (`aime2026`, its translations `aime2026_tr` and
> `aime2026_pt`, `tubitak_math2026` and `pt_exams_math`), and an `evalhub report`
> sub-app aggregates an entire campaign into a master CSV and renders the tables
> and plots.

The judging stage itself is the subject of the paper described in the
[Paper](#-paper) section below.

> [!Warning]
> The pipeline layer is frozen at the configuration the paper reports. The API
> of the Python package may still change.

## 📝 Paper

*Does CoT-Pass@k Really Check the CoT? A Multilingual Mathematical Audit.*
Tarık Tuna Taşaltı, Burcu Hüdaverdi, David Semedo. Accepted at the 6th Workshop
on Multilingual Representation Learning (MRL) at EMNLP 2026.

Correct solutions are corrupted with deterministic edits that damage the
reasoning chain and the final answer separately, so the correct verdict is
known by construction. Across five mathematical benchmarks in English, Turkish
and Portuguese, the judges accept corrupted chains almost as often as clean
ones and reject them mainly when the final answer is wrong; on current solvers
CoT-Pass@k then collapses onto Pass@k. The main tables are in
[`docs/results.md`](docs/results.md).

| Where | What |
|---|---|
| [`analysis/`](analysis/) | the scripts, data tables and figures behind every number in the paper; `analysis/README.md` maps each figure, table and quoted statistic to its script |
| [`analysis/data/`](analysis/data/) | the published data layer: run-level metrics, per-question counts and per-solution judge verdicts (identifiers and counts only, no text) |
| [`error_study/`](error_study/) | the deterministic error-injection study and its judge runners |
| [`docs/results.md`](docs/results.md) | the main tables and figures of the paper, regenerated from `analysis/data` |
| [`docs/benchmarks.md`](docs/benchmarks.md) | the five benchmarks, their sources, licences and translation protocol |
| [`docs/adding_a_benchmark.md`](docs/adding_a_benchmark.md) | how to add a benchmark that works in both the Pass@K and the CoT-Pass@K stage |

The benchmark suite and where each set comes from:

| Benchmark | Language | Problems | Source | In this repository |
|---|---|---|---|---|
| `aime2026` | English | 30 | [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026) on Hugging Face (CC BY-NC-SA 4.0; problems by the [MAA](https://maa.org/maa-invitational-competitions/)) | downloaded at run time |
| `aime2026_tr`, `aime2026_pt` | Turkish, Portuguese | 30 each | our translations of the set above, audited by native speakers (CC BY-NC-SA 4.0); on Hugging Face as [tariktuna/aime2026-tr-pt](https://huggingface.co/datasets/tariktuna/aime2026-tr-pt) | [`evalhub/benchmarks/math/aime2026_tr/`](evalhub/benchmarks/math/aime2026_tr/), [`aime2026_pt/`](evalhub/benchmarks/math/aime2026_pt/) |
| `tubitak_math2026` | Turkish | 32 | [TÜBİTAK National Mathematics Olympiad 2026, first stage](https://bilimolimpiyatlari.tubitak.gov.tr/tr/gecmis-sinav-sorulari) (copyright TÜBİTAK; research use, removed on request) | [`evalhub/benchmarks/math/tubitak_math2026/`](evalhub/benchmarks/math/tubitak_math2026/) |
| `pt_exams_math` | Portuguese | 166 | the mathematics questions of PHEB (Tavares et al., LREC 2026; [AMALIA-LLM/pheb](https://github.com/AMALIA-LLM/pheb)), Portuguese national exams 2006 to 2023, filtered by hand from the multiple-choice set and converted to open answer with their ground truths; the full benchmark lives in the PHEB repository and on Hugging Face as [amalia-llm/pt_exams](https://huggingface.co/datasets/amalia-llm/pt_exams); our open-answer subset is [tariktuna/pt-exams-math-open](https://huggingface.co/datasets/tariktuna/pt-exams-math-open) | [`evalhub/benchmarks/math/pt_exams_math/`](evalhub/benchmarks/math/pt_exams_math/) |

The model generations and the judges' outputs are not released. Everything the
paper reports can be recomputed from the tables in `analysis/data/`; the raw
outputs can be regenerated with the pipeline in this repository.

If you use this code or the benchmark suite, please cite the paper (the
entry will be updated with the ACL Anthology record when it appears; GitHub's
"Cite this repository" button reads `CITATION.cff`):

```bibtex
@inproceedings{tasalti2026cotpassk,
  title     = {Does {CoT}-{Pass}@k Really Check the {CoT}? A Multilingual Mathematical Audit},
  author    = {Ta{\c{s}}alt{\i}, Tar{\i}k Tuna and H{\"u}daverdi, Burcu and Semedo, David},
  booktitle = {Proceedings of the 6th Workshop on Multilingual Representation Learning (MRL)},
  year      = {2026},
  note      = {To appear}
}
```

## ✨ Features

- 🔄 **OpenAI API Compatible** - Seamless integration with existing workflows
- 💻 **Command Line Interface** - Easy benchmarking through intuitive commands
- 🧩 **Extensible Framework** - Add custom tasks and evaluation metrics

> [!Important]
> This project is **not** for production-grade applications requiring high robustness and generalizability (like [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)).
>
> Design principles:
>
> - Separation between generation and evaluation processes
> - Minimal viable code implementation
> - Prioritize simplicity and modularity over comprehensive feature sets
> - Easy to expose evaluation details(prompts, answer extraction, etc.)

## 📦 Installation

The instructions use [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver.

```bash
uv venv --python 3.12
source .venv/bin/activate

# Install the package
uv pip install -e ".[all]" # other options: [dev], [base], [report], [sglang]

# Recommend cleaning up cache after pulling the latest changes
rm -rf ~/.cache/evalhub/
```

With conda, the provided `environment.yml` covers core evaluation and reporting:

```bash
conda env create -f environment.yml
conda activate evalhub
```

Versions used for the experiments in the paper (Python 3.12 on Linux, NVIDIA
A100 and H200 GPUs):

| Package | Version |
|---|---|
| vllm | 0.19.0 |
| torch | 2.10.0 |
| transformers | 5.6.0 (development build, commit `947eff6e`) |
| litellm | 1.80.0 |
| pandas | 3.0.2 |
| numpy | 2.2.6 |
| scipy | 1.17.1 |
| matplotlib | 3.10.8 (paper figures rendered with 3.7.0) |

> [!Note]
> Python 3.12 is the tested and recommended version. The model-serving stack
> (vLLM or SGLang) is sensitive to the GPU and CUDA version and is installed
> separately from the core package; see [Quick Start](#-quick-start) below, or
> install the SGLang extra with `pip install -e ".[sglang]"`.

## 🚀 Quick Start

### First run

Three entry points, depending on what you want to measure. All three use
Qwen3.5-0.8B in non-thinking mode on the Portuguese exams, so they fit on one
GPU and finish in minutes; the served-model stack is vLLM
(`pip install vllm`). Fetch the checkpoint first if the machine has no Hugging Face access at run
time:

```bash
hf download Qwen/Qwen3.5-0.8B
```

**Pass@K only.** Generation and evaluation of one benchmark, no judge. Start
a server, point the CLI at it, sample, grade:

```bash
vllm serve Qwen/Qwen3.5-0.8B --port 30000 --chat-template scripts/templates/qwen3.5-no-think.jinja &
export HOSTED_VLLM_API_BASE="http://0.0.0.0:30000/v1" HOSTED_VLLM_API_KEY="EMPTY"
evalhub gen  --model hosted_vllm/Qwen/Qwen3.5-0.8B --tasks pt_exams_math \
    --temperature 0.6 --top-p 0.95 --n-samples 4 --max-completion-tokens 8192 --output-dir out/
evalhub eval --tasks pt_exams_math --solutions out/pt_exams_math.jsonl --output-dir out/
```

`out/pt_exams_math_summary.json` holds Pass@k, G-Pass@k and mG-Pass@k;
`out/pt_exams_math_results.jsonl` the per-generation correctness. The same two
stages under Slurm, with the server managed for you:
`scripts/submit.sh scripts/run_eval_only.sh scripts/configs/qwen_0.8b_demo.env`.

**CoT-Pass@K on an existing run.** Judge the answer-correct generations of a
base run you already have (the `out/` above), then apply the veto:

```bash
evalhub cot extract --base-results out/pt_exams_math_results.jsonl --base-raw out/pt_exams_math_raw.jsonl \
    --output out/judge/cot_judge_pt_input.jsonl
evalhub gen --model hosted_vllm/Qwen/Qwen3.5-0.8B \
    --tasks cot_judge_pt --temperature 0.6 --top-p 0.95 --n-samples 3 --max-completion-tokens 4096 \
    --output-dir out/judge/ --override-args '{"file_path": "out/judge/cot_judge_pt_input.jsonl"}'
evalhub cot finalize --base-results out/pt_exams_math_results.jsonl --base-raw out/pt_exams_math_raw.jsonl \
    --judge-solutions out/judge/cot_judge_pt_raw.jsonl --output-dir out/judge/ --benchmark pt_exams_math
```

`out/judge/pt_exams_math_cot_summary.json` holds CoT-Pass@k. The same server
serves as judge here; the paper uses a separate, stronger thinking-mode judge. Pick the judge task
whose language matches the benchmark (`cot_judge`, `cot_judge_tr`,
`cot_judge_pt`). Under Slurm the same stage is `run_judge_only.sh` with
`BASE_RESULTS_DIR` pointing at the base run:

```bash
scripts/submit.sh scripts/run_judge_only.sh scripts/configs/qwen_0.8b_demo.env \
    --set BASE_RESULTS_DIR=results_demo/non-think/Qwen3.5-0.8B/pt_exams_math__t0.6__max8192__n4
```

**Both, end to end.** The demo config runs generation, evaluation, extraction,
three judge verdicts per generation, the vote and veto, and the report in one
job. Without Slurm, plain bash starts and stops the servers itself; with Slurm,
`submit.sh` turns the config's `SLURM_*` knobs into sbatch flags:

```bash
bash scripts/run_end_to_end.sh scripts/configs/qwen_0.8b_demo.env
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/qwen_0.8b_demo.env
```

The config samples 4 generations per problem on the Portuguese exams and
judges with the Portuguese prompt. To run another benchmark with its matching
judge prompt, pass overrides through `submit.sh`, or through an overrides file
for plain bash; values there win over the config:

```bash
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/qwen_0.8b_demo.env \
    --benchmarks aime2026_tr --set JUDGE_TASK=cot_judge_tr
# or
printf 'BENCHMARKS="aime2026_tr"\nJUDGE_TASK="cot_judge_tr"\n' > my_overrides.env
EVALHUB_OVERRIDES_FILE=my_overrides.env bash scripts/run_end_to_end.sh scripts/configs/qwen_0.8b_demo.env
```

Where the results land (`OUTPUT_ROOT=results_demo` in the config), from a run
of this config on one A100 (23 minutes):

```
results_demo/non-think/Qwen3.5-0.8B/
  pt_exams_math__t0.6__max8192__n4/
    pt_exams_math_raw.jsonl              every generation (664)
    pt_exams_math_results.jsonl          per-generation correctness
    pt_exams_math_summary.json           Pass@k: 29.7 / 43.2 / 56.6 at k = 1, 2, 4
    pt_exams_math_per_task.csv           per-problem counts
  judged_by/Qwen3.5-0.8B__state-non-think__t0.6__max4096__basemax8192/
    pt_exams_math__t0.6__max8192__n4/
      pt_exams_math_cot_judge_input.jsonl  the 197 answer-correct generations handed to the judge
      cot_judge_pt_raw.jsonl               three judge verdicts per correct generation
      pt_exams_math_cot_majority.jsonl     the majority vote per generation
      pt_exams_math_cot_summary.json       CoT-Pass@k: 7.1 / 12.8 / 21.7 (150 of 197 vetoed)
results_demo/report.csv, report_tasks.csv, report_plots/
```

The numbers say something about the demo, not about the metric: a 0.8B model
judging its own chains in non-thinking mode vetoes most of them. The paper's
setting, a stronger thinking-mode judge with a 16,384-token budget, is what the
`analysis/` package documents.

Then `evalhub report aggregate --results-root results_demo/ --output results_demo/report.csv`
and `evalhub report plot --csv results_demo/report.csv --output-dir results_demo/report_plots/`
rebuild the tables and plots over everything under the root (the end-to-end
job already does this).

`docs/user_guide.md` explains every knob, the result layout and the Slurm
wrappers; `docs/adding_a_benchmark.md` shows how to plug in a new benchmark so
that both stages work.

### Environment Variables

EvalHub uses [litellm](https://www.litellm.ai/) to access models, so the API key and base URL have to be set for the provider in use.
For a local model served with vLLM or SGLang:

```bash
export HOSTED_VLLM_API_BASE="http://0.0.0.0:30000/v1"
export HOSTED_VLLM_API_KEY="your_api_key"

hf download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir $HOME/models/Qwen/Qwen3-30B-A3B-Instruct-2507
python -m sglang.launch_server \
  --model $HOME/models/Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --context-length 32768 \
  --tp-size 4 \
  --ep-size 4 \
  --host 0.0.0.0 \
  --port 30000
```

Logging goes through loguru and is configured with `LOG_LEVEL` and `LOG_DIR`.

```bash
export LOG_LEVEL="INFO" # default is "INFO"
export LOG_DIR="./logs" # default is None
```

### Commands

```bash
evalhub --help

# aime2025
evalhub gen --model hosted_vllm/Qwen/Qwen3-30B-A3B-Instruct-2507 --tasks aime2025 --temperature 0.7 --top-p 0.8 --n-samples 64 --max-completion-tokens 28272 --num-workers 256 --output-dir $HOME/metrics/Qwen/Qwen3-30B-A3B-Instruct-2507/
evalhub eval --tasks aime2025 --solutions $HOME/metrics/Qwen/Qwen3-30B-A3B-Instruct-2507/aime2025.jsonl --output-dir $HOME/metrics/Qwen/Qwen3-30B-A3B-Instruct-2507/
evalhub view --results $HOME/metrics/Qwen/Qwen3-30B-A3B-Instruct-2507/aime2025_results.jsonl --max-display 20

# livecodebench
evalhub gen --model hosted_vllm/Qwen/Qwen3-30B-A3B-Instruct-2507 --tasks livecodebench --temperature 0.7 --top-p 0.8 --n-samples 64 --max-completion-tokens 28272 --num-workers 256 --output-dir $HOME/metrics/Qwen/Qwen3-30B-A3B-Instruct-2507/ --override-args '{"release_version": "v6"}'
evalhub eval --tasks livecodebench --solutions $HOME/metrics/Qwen/Qwen3-30B-A3B-Instruct-2507/livecodebench_v6.jsonl --output-dir $HOME/metrics/Qwen/Qwen3-30B-A3B-Instruct-2507/ --override-args '{"release_version": "v6"}'
evalhub view --results $HOME/metrics/Qwen/Qwen3-30B-A3B-Instruct-2507/livecodebench_results.json --max-display 20
```

More commands are listed in [docs/cmds.md](docs/cmds.md).

> [!Note]
> `view` is supported for math and livecodebench tasks only now!

## 🧠 CoT-Pass@K Pipeline

Every base-correct generation is checked by a stronger judge LLM, and the
per-generation verdicts are majority-voted to either keep or veto the original
"correct" label.

The orchestrators live under [`scripts/`](scripts/README.md) and share a common
bash library. `scripts/run_eval_only.sh` runs base generation and base
evaluation only; `scripts/run_judge_only.sh` runs the judge stage over an
existing base run, useful for re-judging with a different judge;
`scripts/run_end_to_end.sh` is the single-job orchestrator that runs the
target, the judge, the CoT finalize step and the report in sequence. All three
are driven by environment variables (`scripts/cot_pipeline.env.example`
documents every knob) and can be wrapped in a Slurm job. The quick start in
[`docs/user_guide.md`](docs/user_guide.md) is a five-minute walk-through on a
small model, and the same guide covers running, debugging and extending the
pipeline; [`scripts/README.md`](scripts/README.md) describes the shell scripts
and their env contracts.

A single run with the model and benchmark chosen on the command line, without
editing an env file:

```bash
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/base.env \
    --model Qwen/Qwen3.5-0.8B-Base \
    --judge Qwen/Qwen3.5-0.8B \
    --benchmarks "aime2026 aime2026_tr aime2026_pt"
```

Sweeps over several models, benchmarks or temperatures go through the DAG
submitter:

```bash
scripts/orchestrate.sh scripts/configs/base.env sequential \
    --models "Qwen/Qwen3.5-0.8B-Base meta-llama/Llama-3.1-8B" \
    --benchmarks "aime2026 math500" \
    --judge Qwen/Qwen3.5-0.8B
```

`scripts/submit.sh` and `scripts/orchestrate.sh` accept the same CLI
override flags (`--model`, `--benchmark[s]`, `--judge`, ...), so one env
file serves many runs.

## 📊 Reporting

The `evalhub report` sub-app aggregates every summary file under an
`OUTPUT_ROOT` into a single long-form CSV and renders the plot suite:

```bash
# 1. Aggregate every {benchmark}_summary.json / *_cot_summary.json into one CSV
evalhub report aggregate --results-root ./results --output ./report.csv

# 2. Render PNG + PDF plots (Pass@K curves, base-vs-CoT bars, heatmaps, veto rate)
evalhub report plot --csv ./report.csv --output-dir ./report_plots --format both
```

The optional dependency group has to be installed first: `uv pip install -e ".[report]"`.
[`docs/reporting.md`](docs/reporting.md) has the full walk-through and the CSV schema.

## 🛠 Development

### New Dataset

See [docs/adding_a_benchmark.md](docs/adding_a_benchmark.md) for a benchmark that
works in both the Pass@K and the CoT-Pass@K stage, and [docs/tutorial.md](docs/tutorial.md)
for the generic upstream walkthrough.

### Code Quality Tools

[Ruff](https://github.com/astral-sh/ruff) is the Python linter and formatter.

```bash
# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Pre-commit Hooks

Pre-commit hooks check the code before each commit.

```bash
# 📦 Installation
pre-commit install

# Run all checks manually
pre-commit run --all-files
```

### Test

```bash
# Run all tests
pytest -W ignore::Warning
```

## 🛣 Roadmap

[docs/history.md](docs/history.md) has the details.

## 🚧 Status

Frozen at the configuration reported in the paper. The pipeline scripts were
written to drive a multi-model campaign on a SLURM cluster; site-specific
settings are read from environment variables and the config files under
`scripts/configs/`, see [scripts/README.md](scripts/README.md).

## 🌐 Acknowledgements

The audit was supported by the AMALIA project under Measure RE-C05-i08 of the
Portuguese national Programa de Recuperação e Resiliência, and by the NOVA
LINCS project (UID/04516/2025). The upstream project acknowledges:

- [EvalPlus](https://github.com/evalplus/evalplus)
- [deepscaler](https://github.com/agentica-project/deepscaler)
- [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness)
- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
- [verl](https://github.com/volcengine/verl)

## 📄 License

This project is licensed under the terms of the MIT license. Third-party code
and data keep their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
