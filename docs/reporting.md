# Reporting

`evalhub report` turns the per-run summary files written by `evalhub eval` and
`evalhub cot finalize` into a single **wide master CSV** — one row per evaluated
`(model, mode, benchmark, judge)`, with every metric at every K and τ in that
row — and then renders a **Pass@K vs CoT-Pass@K** visualisation suite from it.
The CSV is the contract; the plots read only the CSV.

The CSV lives **inside the results tree** by convention: `results/report.csv`
(with plots under `results/report_plots/`). Both are the command defaults.

## Install

```bash
uv pip install -e ".[report]"     # pulls in pandas
```

## The five commands

```bash
# Full rebuild: scan a results tree -> write the whole CSV (default: results/report.csv)
evalhub report aggregate --results-root ./results

# Incremental: add/replace ONE row for a single summary (idempotent)
evalhub report upsert --summary ./results/.../aime2026_summary.json

# Render the Pass@K vs CoT-Pass@K plot suite from the CSV (default out: results/report_plots/)
evalhub report plot
```

The pipeline scripts (`scripts/run_report.sh`, `scripts/run_end_to_end.sh`) run
`aggregate → plot` as the report stage, writing everything under `OUTPUT_ROOT`
(i.e. `results/`).

A companion guide makes the output usable without re-deriving anything:

* **[`docs/report_plots_guide.md`](report_plots_guide.md)** — the prose
  interpretation manual for every `results/report_plots/<family>` folder and PNG:
  what each shows, how to read the axes (note y = 0→cell-max ⇒ compare gaps
  *within* a cell, not heights across cells), and a "which folder answers which
  question" recipe.

`upsert` is keyed by `(model, state, benchmark, judge_model, judge_state)`:
re-running it for the same key **replaces** that row, a new key **appends**, and
an unseen K **grows** the schema (older rows get blanks for the new columns).
Drop it into a pipeline right after each evaluation to grow the CSV one result at
a time; `aggregate` is the from-scratch equivalent.

## The No-Judge vs cot convention

The **`judge_model` column is the discriminator** — there are no separate `cot_*`
columns:

| `judge_model` | `judged` | What the metric columns mean |
|---|---|---|
| empty | `False` | **No-Judge**: `pass@k`, `g-pass@k`, `mg-pass@k` (base eval) |
| set   | `True`  | **cot**: `cot-pass@k`, `cot-g-pass@k`, `cot-mg-pass@k` (judged) |

A judge only vetoes (relabels answer-correct-but-CoT-wrong generations), so a cot
value is always ≤ its No-Judge counterpart; the gap is the veto effect.

## What the scanner sees

`aggregate` / `upsert` parse `*_summary.json` (No-Judge) and `*_cot_summary.json`
(judged) and read model/state/benchmark/judge from the surrounding directory
names. The recognised layouts (V3 hoisted-state, V2 nested, V1/V0 flat) are
unchanged from before; unrecognised directories are skipped with a debug log.
A single file can be parsed directly via
`evalhub.report.scan.record_from_summary(path, source_root)`.

## CSV schema (wide)

One row per `(model, state, benchmark, judge_model, judge_state)`.

**Identity / metadata**

| Column | Notes |
|---|---|
| `model` / `model_short` | Full name + short label (`Qwen3.5-9B-Base` → `Q-9B·Base`). |
| `model_family` / `model_size_b` / `is_base` | `Qwen`/`gemma`/…, params in B, pretrained flag. |
| `state` / `mode` | `base`/`non-think`/`think` + human label. |
| `benchmark` / `language` | Benchmark + `EN`/`PT`/`TR`/`TR-OL`. |
| `judged` | `False` = No-Judge row, `True` = a judge graded it. |
| `series` | `No-Judge` or `cot:<short_judge>·<jstate>` — one-column grouping key. |
| `judge_model` / `judge_state` | Empty on No-Judge rows. |
| `n_samples`, `temperature`, `max_tokens`, `judge_*` | Sampling knobs. |
| `cons_at_k` | Cons@K. |
| `total_tasks`, `total_generations`, `true_count`, `false_count`, `cot_false_count`, `invalid_count` | Run totals (cot rows carry the veto counts). |
| `run_dir`, `summary_path` | Provenance. |

**Metric columns** (judge empty → pass family; judge set → cot family), grouped
per K (`pass`, then the four τ, then `mg`):

| Pattern | Example | Meaning |
|---|---|---|
| `pass@{k}` | `pass@64` | Pass@k |
| `gpass@{k}_t{τ}` | `gpass@64_t0.5` | G-Pass@k at threshold τ (≥⌈k·τ⌉ of k correct) |
| `mgpass@{k}` | `mgpass@64` | mG-Pass@k (integrated G-Pass over τ∈(0.5, 1]) |

τ ∈ {0.25, 0.5, 0.75, 1.0}. The K axis is the union of all K seen in the data
(canonical fallback `[1,2,4,8,16,32,64,128]` for an empty CSV). Cells for a K a
run didn't produce are left blank.

### Companion per-task CSV (V2+)

Alongside each summary, the evaluator also writes `<benchmark>_per_task.csv`
(and `<benchmark>_cot_per_task.csv`) with one row per question — counts
(`true`/`false`/`cot_false`/`invalid_format`), per-K pass rates, ground truth,
majority vote, and `is_correct_majority`. Open this in Excel for a per-question
drill-down.

## The plot suite (`report plot`)

`evalhub report plot` reads the wide CSV and renders a comparison-first suite into
`results/report_plots/<family>/`. Two papers frame it: **2504.13837** (Pass@K
curves over K, base vs RL → here **No-Judge vs Judge**) and **2506.14245** (the
**CoT-Pass@K** metric — answer *and* reasoning correct → our judged rows). The
single through-line of every figure is **how much the CoT veto moves the metric**,
per model, per benchmark, per language — never averaged away.

Two conventions hold everywhere:

* **Y axis runs 0 → the cell's own max** (not a fixed 0–100 %), so small
  No-Judge↔cot gaps stay legible; the 0 baseline keeps absolute rates readable.
* **X axis is K on a log₂ scale.** The judge is always `think` (non-think judge
  labels are migration mislabels and never appear).

Each of the six metric *lenses* — `pass`, `gpass_t{0.25,0.5,0.75,1.0}`, `mgpass`
— gets its **own PNG** (so each G-Pass τ is separate), generally split per mode
(`base` / `non-think` / `think`).

| Folder | What one file shows |
|---|---|
| `judge_effect/{metric}__{state}.png` | **Core.** Grid rows=model × cols=benchmark; each cell = No-Judge solid (`pass@k`) + every judge dashed (`cot-pass@k`). |
| `bench_compare/{metric}__{state}__{nojudge,cot}.png` | Language transfer: 4 language curves per cell (`__cot` adds a column per judge). |
| `size_compare/{metric}__{state}__{nojudge,cot__judge}.png` | Size scaling: one curve per model, coloured by params; cot variant overlays the judge dashed. |
| `veto_curve/{metric}__{state}.png` | Δ(k) = No-Judge − cot per judge — how the veto grows with K even when pass@k saturates. |
| `mode_compare/{metric}__{family}.png` | Pretrained vs Instruct-Non-Think vs Instruct-Think overlay (rows=size × cols=benchmark). |
| `per_model/{model}__{state}.png` | One model's fingerprint: rows=metric (6) × cols=benchmark. |
| `tables/{benchmark}__k{1,64}__nojudge.png` | All `(model·mode)` × 6 metrics ×100, sequential colour. |
| `tables/{benchmark}__k{1,64}__cotdelta__{judge}.png` | Veto Δ = (No-Judge − cot) ×100, diverging colour. |
| `tables/headline__{judge}__k{1,64}.png` | `pass / cot-pass` per benchmark cell, coloured by Δ. |
| `comparisons/veto__{metric}__k{1,64}__{judge}.png` | **Multilingual veto heatmap** — rows=`(model·mode)` × cols=language, value = No-Judge − cot. |
| `comparisons/pass_vs_cot_k{1,64}.csv` | Long companion: every `(model, mode, language, judge, metric)` with `nojudge` / `cot` / `delta`. |

The whole suite is driven by `evalhub.report.plots.render_all(df, out_dir)`; each
family is isolated, so a failure in one logs and is skipped rather than aborting
the rest.

## Programmatic use

```python
from pathlib import Path
import pandas as pd
from evalhub.report import (
    aggregate_results, build_wide_dataframe, scan_results,
    upsert_summary, record_from_summary,
)
from evalhub.report.plots import render_all

# full rebuild (writes results/report.csv)
df = aggregate_results(Path("./results"), Path("./results/report.csv"))

# or in-memory
df = build_wide_dataframe(scan_results(Path("./results")))

# incremental, one summary at a time
upsert_summary(Path("./results/.../aime2026_summary.json"),
               Path("./results/report.csv"), Path("./results"))

# render the plot suite from the CSV
render_all(pd.read_csv("./results/report.csv"), Path("./results/report_plots"))
```
