# Reporting & Dashboard

This document covers the `evalhub report` sub-app, which turns the per-run
summary files written by `evalhub eval` and `evalhub cot finalize` into a
queryable master CSV, publication-ready static plots, and an interactive
Streamlit + Plotly dashboard.

## Install

The aggregation pipeline ships under an opt-in extra so the core install
stays lean:

```bash
uv pip install -e ".[report]"
```

This pulls in `pandas`, `matplotlib`, `seaborn`, `streamlit`, and `plotly`.

## The three commands

```bash
evalhub report aggregate \
    --results-root ./results \
    --output ./report.csv

evalhub report plot \
    --csv ./report.csv \
    --output-dir ./report_plots \
    --format both                       # png | pdf | both

evalhub report dashboard \
    --csv ./report.csv \
    --results-root ./results \
    --port 8501
```

The dashboard launches Streamlit in the foreground; visit
`http://localhost:8501` to use it. Passing `--results-root` enables the
"Drill-down" tab, which loads the raw JSONL records for a selected run.

## What the scanner sees

`evalhub report aggregate` walks `--results-root` with `rglob("*_summary.json")`
and classifies each summary file by its filename suffix:

* `<benchmark>_summary.json`     → base evaluation
* `<benchmark>_cot_summary.json` → CoT-vetted evaluation

The parent directory name is parsed by regex to extract the model, state,
temperature, and max-token settings. Two layouts are recognised:

```
# Canonical (produced by run_end_to_end.sh, run_eval_only.sh, run_judge_only.sh)
<model>_state-<base|non-think|think>_t<T>_max<N>/<benchmark>/<benchmark>_summary.json
judgments/<target>_state-<state>_judged_by_<judge>_state-<state>_t<T>_max<N>/<benchmark>/<benchmark>_cot_summary.json

# Legacy fallback (pre-`_state-` directories)
<model>_t<T>_max<N>/<benchmark>/<benchmark>_summary.json   # state="unknown"
```

Anything that doesn't match either pattern is skipped with a debug log line.

## CSV schema (long format)

One row per `(run, K)`. Stats columns are populated only for `eval_type =
"cot_eval"` rows (they come from the sibling `*_cot_stats.json`).

| Column | Type | Notes |
|---|---|---|
| `model` | string | Target model basename (e.g. `Qwen3-7B-Instruct`). |
| `state` | string | `base` / `non-think` / `think` / `unknown`. |
| `temperature` | float | Sampling temperature. |
| `max_tokens` | int | `max_completion_tokens` used for this run. |
| `benchmark` | string | Benchmark short name (`aime2025`, `gsm8k`, ...). |
| `eval_type` | string | `base_eval` or `cot_eval`. |
| `judge_model` | string | Only set for `cot_eval` rows. |
| `judge_state` | string | Only set for `cot_eval` rows. |
| `judge_temperature` | float | Only set for `cot_eval` rows. |
| `judge_max_tokens` | int | Only set for `cot_eval` rows. |
| `k` | int | The K-axis. NaN only for empty `pass_at_k` stubs. |
| `pass_at_k` | float | Pass@K for this row. |
| `cons_at_k` | float | Cons@K — repeated on every K of the same run. |
| `total_tasks` | int | From `*_cot_stats.json`. |
| `total_generations` | int | From `*_cot_stats.json`. |
| `true_count` | int | Generations the base evaluator marked correct. |
| `false_count` | int | Generations marked incorrect. |
| `cot_false_count` | int | Generations downgraded by the judge's CoT veto. |
| `invalid_count` | int | Generations the parser couldn't grade. |
| `run_dir` | string | Absolute path to the run's output directory. |

## Plots written by `evalhub report plot`

| Name | What it shows |
|---|---|
| `pass_at_k__<model>__<benchmark>.{png,pdf}` | Pass@K vs K on a log-2 axis, one line per `eval_type` (base vs CoT). One file per (model, benchmark) pair. |
| `base_vs_cot_pass_at_1.{png,pdf}` | Horizontal grouped bars: Pass@1 per benchmark, hue=model, comparing base vs CoT-vetted evaluation. |
| `pass_at_1_heatmap.{png,pdf}` | Pass@1 heatmap with rows=model, cols=benchmark, values annotated. |
| `cot_veto_rate.{png,pdf}` | CoT veto rate (`cot_false / total_generations`) per (model, benchmark). |

Filenames are sanitised so the output directory is safe to commit.

## Dashboard tabs

| Tab | What it shows |
|---|---|
| **Overview**   | KPI cards (#runs, #models, #benchmarks, #CoT runs) and a sortable, filterable table of the long-form CSV. |
| **Pass@K**     | Plotly line plot, faceted by benchmark, with model/state/eval-type encoded by colour / dash / symbol. |
| **Heatmap**    | Plotly imshow of Pass@K for a selectable K. |
| **CoT veto**   | Horizontal bar of `cot_false / total_generations` per (model, benchmark). |
| **Drill-down** | Pick a run, pick one of its JSONL files, load up to N records, view in a paginated table. Requires `--results-root`. |

The sidebar exposes multi-select filters for `eval_type`, `model`, `state`,
and `benchmark`. All tabs respect the active filter.

## End-to-end demo

Assuming you already ran the pipeline against a couple of models/benchmarks:

```bash
# (one-time) install report deps
uv pip install -e ".[report]"

# (one-time) sanity-check that the scanner sees your results
evalhub report aggregate --results-root ./results --output ./report.csv

# render the static plot set
evalhub report plot --csv ./report.csv --output-dir ./report_plots --format both

# launch the dashboard
evalhub report dashboard --csv ./report.csv --results-root ./results
```

## Programmatic use

Every CLI command has a public Python entry point so the same logic can drive
notebooks or CI jobs:

```python
from pathlib import Path
from evalhub.report import aggregate_results, build_dataframe, scan_results
from evalhub.report.plots import render_all

records = scan_results(Path("./results"))
df = build_dataframe(records)               # long-form pandas DataFrame
render_all(df, Path("./report_plots"), formats=("png",))
```
