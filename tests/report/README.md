# `tests/report/` — Aggregation & Report Tests

This directory exercises the `evalhub.report` subpackage end-to-end.

## Fixtures

`conftest.py` builds on-the-fly results trees inside `tmp_path`, one per layout
the scanner must understand — so realistic inputs are exercised without
committing any binary artefacts:

* `v5_results_root` — the **current** V5 layout (one folder per model; the
  sampling suffix lives on the benchmark leaf, on both the base side and under
  `judged_by/`). Mirrors what `scripts/run_end_to_end.sh` produces today.
* `v3_results_root` / `v2_results_root` — earlier nested layouts (state hoisted
  to a parent dir; or `__state-…__` embedded in the leaf).
* `fake_results_root` / `legacy_results_root` — the older flat `_state-…` and
  pre-`state-` namings. Confirm the scanner's backward-compatibility fallbacks.

Nothing is written to the repository; everything lives under pytest's per-test
temporary directory.

## Files

* `test_scan.py`      — regex parsing + classification of base vs CoT records across layouts.
* `test_aggregate.py` — wide DataFrame explosion, CSV round-trip, upsert schema growth, empty stubs.
* `test_upsert.py`    — append-or-replace row identity for `report upsert`.
* `test_integrity.py` — cross-checks (CoT ≤ No-Judge), model exclusion, K cap.
* `test_plots.py`     — the plot suite renders without error (requires matplotlib + seaborn).
