# Examples

Auxiliary, reusable tooling that operates on a results tree produced by the
CoT-Pass@K pipeline (`scripts/run_*.sh`). These are **not** part of the core
package; run them from the repository root, e.g.:

```bash
python examples/scripts/audit_integrity.py --results-root ./results
```

## `examples/scripts/`

| Tool | What it does |
|---|---|
| `migrate_results_layout.py` | Migrate a results tree to the current **V5** layout (one folder per model, sampling suffix on the benchmark leaf). Safe by default: dry-run unless `--execute`, takes a tar backup, writes a manifest, and supports `--revert`. |
| `migrate_legacy_paths.py` | Older legacy path migration (earlier layout phases). Kept for trees that predate V5. |
| `audit_integrity.py` | Audit a results tree for integrity — per-cell judge-vote counts, duplicated generations, and missing base/judge files. Read-only. |
| `finalize_missing_cot.py` | Finalize CoT-Pass@K summaries for `judged_by/…` cells where the judge ran but the `*_cot_summary.json` is missing (majority vote + metrics). |
| `trim_judge_votes.py` | Trim over-voted judge cells back to the expected number of votes (dry-run first; `--execute` to apply), then recompute the CoT summary. |
| `make_passk_tables.py` | Render custom Pass@K / CoT-Pass@K comparison tables from the aggregated `report.csv`. |

All tools take their results root via a flag (default `results/`) and never
modify generation/judge **prompts** — they only read or reshape result files.
