# `tests/report/` — Aggregation & Dashboard Tests

This directory exercises the `evalhub.report` subpackage end-to-end.

## Fixtures

`conftest.py` builds two on-the-fly results trees inside `tmp_path`:

* `fake_results_root` — one base eval + one CoT-vetted eval over `gsm8k`
  using the canonical `_state-…` naming. Mirrors the layout produced by
  `scripts/run_end_to_end.sh`.
* `legacy_results_root` — a single base eval using the older naming
  (`<model>_t<T>_max<N>`, no `_state-` suffix). Confirms the scanner's
  backward-compatibility fallback.

Neither fixture writes to the repository; everything lives under pytest's
per-test temporary directory.

## Files

* `test_scan.py`     — regex parsing + classification of base vs CoT records.
* `test_aggregate.py` — DataFrame explosion, CSV round-trip, empty stubs.

Plot and dashboard modules are smoke-tested manually because they require
GUI/Streamlit backends that aren't installed by default.
