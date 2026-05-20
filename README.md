# 🔮 EvalHub

<p align="center">
    <a href="https://github.com/yourusername/evalhub"><img src="https://img.shields.io/badge/Eval-Hub-blue.svg"></a>
    <a href="https://github.com/yourusername/evalhub/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
    <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

<p align="center">
    <a href="#-about">📖 About</a> •
    <a href="#-features">✨ Features</a> •
    <a href="#-installation">📦 Installation</a> •
    <a href="#-quick-start">🚀 Quick Start</a> •
    <a href="#-cot-passk-pipeline">🧠 CoT-Pass@K Pipeline</a> •
    <a href="#-reporting--dashboard">📊 Reporting & Dashboard</a> •
    <a href="#-development">🛠 Development</a> •
    <a href="#-roadmap">🛣 Roadmap</a>
</p>

## 📖 About

All-in-one benchmarking platform for evaluating Large Language Models (LLMs) with comprehensive metrics and standardized testing frameworks.

> [!Warning]
> This project is under active development and the API is not stable yet.

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

> [!Note]
> [uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
uv venv --python 3.12
source .venv/bin/activate

# Install the package
uv pip install -e ".[all]" # other options: [dev], [base], [sglang]

# Recommend cleaning up cache after pulling the latest changes
rm -rf ~/.cache/evalhub/
```

## 🚀 Quick Start

### Environment Variables

Evalhub uses [litellm](https://www.litellm.ai/) to access model. Please first set the api_key and base_url according to the model provider you are using.
For example, to use a local model served via vllm/sglang, you can set:

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

Additionally, evalhub uses loguru's logger, configured via `LOG_LEVEL` and `LOG_DIR`.

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

For more commands, please refer to [docs/cmds.md](docs/cmds.md).

> [!Note]
> `view` is supported for math and livecodebench tasks only now!

## 🧠 CoT-Pass@K Pipeline

EvalHub ships a chain-of-thought-aware Pass@K extension: every base-correct
generation is checked by a stronger judge LLM, and the per-generation verdicts
are majority-voted to either keep or veto the original "correct" label.

End-to-end orchestrators live under [`scripts/`](scripts/README.md) and share a
common bash library:

| Script | What it runs |
|---|---|
| `scripts/run_eval_only.sh` | Base generation + base evaluation only. |
| `scripts/run_judge_only.sh` | Judge stage over an existing base run (handy for re-judging with a different judge). |
| `scripts/run_end_to_end.sh` | Full target → judge → CoT finalize. Canonical replacement for the legacy `run_cot_pass_at_k.sh`. |

All three are env-driven (`scripts/cot_pipeline.env.example` documents every knob)
and Slurm-wrappable. See [`docs/onboarding.md`](docs/onboarding.md) for a
five-minute walk-through on a small model.

## 📊 Reporting & Dashboard

The `evalhub report` sub-app aggregates every summary file under an
`OUTPUT_ROOT` into a single long-form CSV, renders publication-ready plots,
and launches a Streamlit + Plotly dashboard for interactive analysis:

```bash
# 1. Aggregate every {benchmark}_summary.json / *_cot_summary.json into one CSV
evalhub report aggregate --results-root ./results --output ./report.csv

# 2. Render PNG + PDF plots (Pass@K curves, base-vs-CoT bars, heatmaps, veto rate)
evalhub report plot --csv ./report.csv --output-dir ./report_plots --format both

# 3. Launch the interactive dashboard
evalhub report dashboard --csv ./report.csv --results-root ./results --port 8501
```

Install the optional dependency group first: `uv pip install -e ".[report]"`.
Full walk-through and CSV schema in [`docs/reporting.md`](docs/reporting.md).

## 🛠 Development

### New Dataset

See [docs/tutorial.md](docs/tutorial.md) for more details.

### Code Quality Tools

> [!Note]
> We use [Ruff](https://github.com/astral-sh/ruff) as our Python linter and formatter.

```bash
# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Pre-commit Hooks

> [!Note]
> Pre-commit hooks automatically check your code before committing.

```bash
# Installation
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

See [docs/history.md](docs/history.md) for more details.

## 🌐 Acknowledgements

- [EvalPlus](https://github.com/evalplus/evalplus)
- [deepscaler](https://github.com/agentica-project/deepscaler)
- [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness)
- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
- [verl](https://github.com/volcengine/verl)

> [!Important]


## 📄 License

This project is licensed under the terms of the MIT license.
