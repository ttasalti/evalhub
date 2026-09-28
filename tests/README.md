# `tests/`

`pytest tests -q` runs the package tests (about six minutes, most of it the
plot suite). `pytest error_study/tests -q` covers the error-injection study.

| Folder | Coverage |
|---|---|
| `tests/math/` | the answer verifier (upstream), the per-task CSV writer, the `tubitak_math2026` loader |
| `tests/cot/` | generation-id round trips, correct-only extraction, majority voting and ties, the veto and metric recomputation, the end-to-end `finalize` pipeline, model-state resolution, and a `CliRunner` pass through `evalhub cot` |
| `tests/report/` | directory-layout scanning, wide-CSV aggregation and upsert, integrity checks, the any/all threshold columns, per-problem `report_tasks.csv`, and the plot suite on a synthetic results tree (`conftest.py` builds it) |
| `tests/lib/` | the output-path composition rules of `scripts/lib/pipeline_common.sh` |
| `tests/inference/` | the request payload of the generator (token cap sent under both field names) |
| `tests/analysis/` | every analysis script that runs from `analysis/data/` executes and reproduces the shipped tables byte for byte; figures are regenerated and checked for presence |
| `error_study/tests/` | the number helpers, the wrong-number menu, the four injection primitives and the analysis tables of the error-injection study |
