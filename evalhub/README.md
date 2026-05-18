# `evalhub/` (Python package)

**Modified from the evalhub-original repository.**

Modules unchanged from upstream: `inference/`, `callback/`, `tools/`, `view.py`,
`benchmarks/code/`, `benchmarks/general/`, `benchmarks/multilingual/`,
`benchmarks/registry.py`, `benchmarks/base.py`.

Modified for CoT-Pass@K support:

| Path | Change |
|---|---|
| `cli.py` | Added the `cot` Typer sub-app with `extract`, `aggregate`, `metrics`, `finalize` commands. |
| `gen.py` | Logs the resolved chat template and `model_state` at run start. |
| `inference/schemas.py` | Added the `model_state` field on `GenerationConfig` (`base` / `non-think` / `think`). |
| `benchmarks/__init__.py` | Re-exports the new `cot` sub-package. |
| `benchmarks/alignment/__init__.py` | Dropped the `math_judge*` re-exports (the classes are gone). |
| `benchmarks/math/__init__.py` | Re-exports the AIME-2026 variants and the new local datasets (unchanged from your repo). |

Newly created for this project:

| Path | Purpose |
|---|---|
| [`cot/`](./cot/README.md) | Pure-Python CoT-Pass@K post-processing pipeline. |
| [`benchmarks/cot/`](./benchmarks/cot/README.md) | Parameterised LLM-as-a-Judge dataset. |
| `utils/model_state.py` | Model + state → chat template registry. |
