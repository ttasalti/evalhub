## Commands

### serve model

```bash
# serve model via vllm or sglang
vllm serve "$HOME/models/Qwen2.5-3B-Instruct" --port 30000
python -m sglang.launch_server --model-path "$HOME/models/Qwen2.5-3B-Instruct"
```

### Math

```bash
# gsm8k
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks gsm8k --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks gsm8k --solutions $HOME/metrics/Qwen2.5-3B-Instruct/gsm8k.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/gsm8k_results.jsonl --max-display 20

# hendrycks_math
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks hendrycks_math --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks hendrycks_math --solutions $HOME/metrics/Qwen2.5-3B-Instruct/hendrycks_math.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/hendrycks_math_results.jsonl --max-display 20

# math500
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks math500 --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks math500 --solutions $HOME/metrics/Qwen2.5-3B-Instruct/math500.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/math500_results.jsonl --max-display 20

# aime2024
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks aime2024 --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks aime2024 --solutions $HOME/metrics/Qwen2.5-3B-Instruct/aime2024.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/aime2024_results.jsonl --max-display 20

# gpqa
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks gpqa --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks gpqa --solutions $HOME/metrics/Qwen2.5-3B-Instruct/gpqa.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/gpqa_results.jsonl --max-display 20
```

### Code

```bash
# humaneval && mbpp
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks humaneval --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/ -p temperature=0.2 -p top_p=0.95 # -p key=value to override default config
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks mbpp --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalplus.evaluate --dataset humaneval --samples $HOME/metrics/Qwen2.5-3B-Instruct/humaneval.jsonl
evalplus.evaluate --dataset mbpp --samples $HOME/metrics/Qwen2.5-3B-Instruct/mbpp.jsonl

# livecodebench
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks livecodebench --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub eval --tasks livecodebench --solutions $HOME/metrics/Qwen2.5-3B-Instruct/livecodebench.jsonl --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
evalhub view --results $HOME/metrics/Qwen2.5-3B-Instruct/livecodebench_results.json --max-display 20

# bigcodebench
evalhub gen --model "$HOME/models/Qwen2.5-3B-Instruct" --tasks bigcodebench --output-dir $HOME/metrics/Qwen2.5-3B-Instruct/
docker pull bigcodebench/bigcodebench-evaluate
docker run -it \
  --name bcb-eval \
  -v $HOME/metrics/:/app/metrics \
  bigcodebench/bigcodebench-evaluate \
  --execution local \
  --split instruct \
  --subset full \
  --samples /app/metrics/Qwen2.5-3B-Instruct/bigcodebench.jsonl

# run eval next time
docker start bcb-eval
docker exec -it bcb-eval bash
python3 -m bigcodebench.evaluate -execution local --split instruct --subset full --samples /app/data/bigcodebench.jsonl
```

### multi-turn & tool call or callback

#### gsm8k with tool call
```bash
temperature=0.6
top_p=0.95
max_tokens=4096
tool_config_path="$HOME/projects/evalhub/evalhub/tools/config/gsm8k_tool_config.yaml"
system_prompt="You are a math expert. You are given a question and you need to solve it step by step. Reasoning step by step before any tool call. You should use the \`calc_gsm8k_reward\` tool after step by step solving the question, before generate final answer at least once and refine your answer if necessary."

evalhub gen --model "$HOME/models/Qwen2.5-7B-Instruct" --tasks gsm8k --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/ --max-tokens $max_tokens --temperature $temperature --top-p $top_p --tool-config-path $tool_config_path --enable-multiturn --system-prompt "$system_prompt"
```

#### livecodebench with callback
```bash
temperature=0.6
top_p=0.95
max_tokens=4096
system_prompt="You are an expert Python programmer. \
You will be given a question (problem specification) and \
will generate a correct Python program that matches the specification and passes all tests. \
We will provide you with feedback of public test cases results to help you improve your code."

evalhub gen --model "$HOME/models/Qwen2.5-7B-Instruct" --tasks livecodebench --output-dir $HOME/metrics/Qwen2.5-7B-Instruct/ --max-tokens $max_tokens --temperature $temperature --top-p $top_p  --enable-multiturn --system-prompt "$system_prompt" --callback "evalhub.callback.code_callback.CodeCallback"
```

### CoT-Pass@K orchestrators

The orchestrator scripts under `scripts/` chain `evalhub gen`, `evalhub eval`,
and the `evalhub cot ...` post-processing stages together. All three are
env-driven; see `scripts/cot_pipeline.env.example` for every knob.

```bash
# Stage 1 only — base generation + base evaluation.
scripts/run_eval_only.sh scripts/cot_pipeline.env

# Stages 2+3 only — judge an existing base run, then finalize.
BASE_RESULTS_DIR="$HOME/metrics/Qwen2.5-7B-Instruct/aime2025" \
    scripts/run_judge_only.sh scripts/cot_pipeline.env

# Full pipeline.
scripts/run_end_to_end.sh scripts/cot_pipeline.env
```

Or run directly with the model, judge, benchmark, temperature, and sampling
given on the command line — no env editing. `submit.sh` writes the flags into a
throwaway overrides file that the orchestrator sources after the base env
(precedence: **CLI args > env file > defaults**):

```bash
# One model × several benchmarks, explicit sampling.
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/base.env \
    --model Qwen/Qwen3.5-0.8B-Base \
    --judge Qwen/Qwen3.5-0.8B \
    --benchmarks "aime2026 aime2026_tr aime2026_pt" \
    --temperature 0.6 \
    --n-samples 64 \
    --max-completion-tokens 20480 \
    --output-root results
```

Common flags: `--model` (`TARGET_MODEL`), `--judge` (`JUDGE_MODEL`),
`--benchmark` / `--benchmarks` (single / looped), `--target-state`
(`base|non-think|think`), `--temperature`, `--judge-temperature`, `--n-samples`,
`--judge-n-samples`, `--max-completion-tokens`, `--output-root`, and
`--set KEY=VAL` for any other knob. The same flags work with `orchestrate.sh`
for multi-model / multi-temperature DAG sweeps.

### Report aggregation

```bash
# 1. Sweep every {benchmark}_summary.json / *_cot_summary.json into one CSV.
evalhub report aggregate --results-root ./results --output ./report.csv

# 2. Render publication-ready static plots.
evalhub report plot --csv ./report.csv --output-dir ./report_plots --format both
```

See [reporting.md](reporting.md) for the CSV schema, or
[user_guide.md](user_guide.md) for a five-minute quick start and full guide.
