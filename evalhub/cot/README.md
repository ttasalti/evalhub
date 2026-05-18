# `evalhub/cot/` — CoT-Pass@K post-processing

**Newly created for this project.**

This package implements the three local stages that turn standard Pass@K
artifacts into CoT-Pass@K metrics:

| File | Responsibility |
|---|---|
| `ids.py` | Encode/decode the join key `<original_task_id>_gen_<idx>`. The only contract between the base evaluation, the judge generation, and the metric step. |
| `extract.py` | Read `*_results.jsonl` + `*_raw.jsonl`, keep only generations marked correct, write one record per surviving generation. |
| `aggregate.py` | Group judge yes/no outputs by generation id and apply strict majority voting (`#yes > #no`). Invalid extractions abstain. |
| `metrics.py` | Downgrade base-correct generations whose judge majority is "no" to `cot_false`, then recompute Pass@K and Cons@K over the full Ks ladder up to `n_samples`. |
| `pipeline.py` | `finalize_cot_pipeline(...)` composes the three steps when both gen passes have already produced local files. |

All four are pure-Python, side-effect-free except for the explicit output files
they're asked to write. They reuse `evalhub.utils.metrics.compute_pass_at_k` so
the Pass@K formula has exactly one implementation in the repository.

Exposed via the CLI as:

```
evalhub cot extract    --base-results F --base-raw F --output F
evalhub cot aggregate  --judge-solutions F --output F
evalhub cot metrics    --base-results F --majority F --output F --summary F [--stats F]
evalhub cot finalize   --base-results F --base-raw F --judge-solutions F --output-dir D --benchmark NAME
```
