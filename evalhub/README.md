# `evalhub/` (Python package)

The upstream package with the CoT-Pass@K layer added. Upstream code is kept
in place; the fork adds three sub-packages and a handful of hooks.

New in the fork:

| Path | Purpose |
|---|---|
| [`cot/`](./cot/README.md) | CoT-Pass@K post-processing: extract the answer-correct generations, majority-vote the judge verdicts, apply the veto and recompute the metrics (`evalhub cot extract / aggregate / metrics / finalize`) |
| [`benchmarks/cot/`](./benchmarks/cot/README.md) | the judge tasks `cot_judge`, `cot_judge_tr`, `cot_judge_pt`: one prompt template per language over a math run's correct generations |
| [`report/`](./report/README.md) | `evalhub report`: one wide CSV per results tree, per-problem CSV, integrity checks and the plot suite |
| `benchmarks/math/{aime2026,aime2026_tr,aime2026_pt,tubitak_math2026,pt_exams_math}/` | the five benchmarks of the paper ([`../docs/benchmarks.md`](../docs/benchmarks.md)) |
| `utils/model_state.py` | model and mode (`base`, `non-think`, `think`) to chat-template registry |
| `utils/metrics.py` | G-Pass@k and mG-Pass@k next to the upstream Pass@k and majority vote |

Changed upstream files:

| Path | Change |
|---|---|
| `cli.py` | adds the `cot` and `report` sub-apps |
| `gen.py` | logs the resolved chat template and model state; writes the extended summary |
| `inference/generator.py`, `inference/schemas.py` | `model_state`, `reasoning_effort` and `extra_body` on the generation config; the requested token cap is sent as both `max_completion_tokens` and `max_tokens` so API endpoints that ignore one still honour it; requests are issued in task order so generations of one problem share the prompt prefix |
| `benchmarks/base.py` | output files are opened in truncate mode for a fresh run and appended only under `--resume`, so a re-run cannot duplicate records |
| `benchmarks/math/base.py` | per-task CSV next to the results file (`<bench>_per_task.csv`) |
| `benchmarks/math/verifier/rllm.py` | treats a comma between digits as a decimal mark (Turkish answers such as `0,5`) |
| `benchmarks/__init__.py`, `benchmarks/math/__init__.py` | register the new packages |

Everything else (`callback/`, `tools/`, `view.py`, the other benchmark groups)
is upstream code.
