# `tests/`

**Modified from the evalhub-original repository.**

Existing upstream subfolder `tests/math/` (with `test_verifier.py`) is unchanged.

Newly created for this project:

| Path | Coverage |
|---|---|
| `tests/cot/test_ids.py` | Generation-id encode/decode round-trips, including edge cases. |
| `tests/cot/test_extract.py` | Correct-only filtering, order preservation, interleaved task ids, empty result. |
| `tests/cot/test_aggregate.py` | Strict-majority voting, tie handling, list-form solutions, case/whitespace normalisation. |
| `tests/cot/test_metrics.py` | CoT veto downgrade to `cot_false`, Pass@K and Cons@K recomputation, all-vetoed case, multi-task aggregation, stats file. |
| `tests/cot/test_pipeline.py` | End-to-end `finalize_cot_pipeline` against synthetic JSONLs. |
| `tests/cot/test_model_state.py` | State normalisation, model-name state inference, template path resolution, unknown-family fallback. |
| `tests/cot/test_cli_integration.py` | `typer.testing.CliRunner` round-trip through `evalhub cot {extract, aggregate, metrics, finalize}`. |

Run all CoT tests:

```bash
pytest tests/cot/ -q
```
