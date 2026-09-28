# `evalhub/report/`

The `evalhub report` sub-app. It walks a results tree, turns every summary
file into one row of a wide CSV, and renders the Pass@K versus CoT-Pass@K plot
suite from that CSV. The full reference is `docs/reporting.md`.

| Module | Role |
|---|---|
| `scan.py` | finds `*_summary.json` and `*_cot_summary.json` files under a results root and parses the directory layout (solver, mode, benchmark, sampling, judge) into a `RunRecord` |
| `aggregate.py` | writes `report.csv`, one row per (solver, mode, benchmark, solver budget, judge); adds cap-rate and veto columns, integrity checks, and the `upsert` path used after each pipeline run |
| `tasks.py` | writes `report_tasks.csv`, one row per (run, problem), from the per-task CSVs the evaluator leaves next to each run |
| `backfill.py` | adds the any/all approval-rule metrics to judged runs that were finalised before those rules existed |
| `labels.py` | short names, language codes and mode labels used in tables and plots |
| `plots.py` | the plot families (judge effect, benchmark comparison, size and mode comparison, veto curves, per-model fingerprints, tables) |
| `_cli.py` | the Typer commands `aggregate`, `upsert`, `backfill-anyall`, `plot` |

```bash
evalhub report aggregate --results-root results/ --output results/report.csv
evalhub report plot --csv results/report.csv --output-dir results/report_plots
```
