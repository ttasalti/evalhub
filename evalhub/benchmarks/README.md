# `evalhub/benchmarks/`

One package per benchmark, grouped by kind. A loader registers itself with
`@register_dataset` and provides `load_tasks()`, `format_prompt()` and the
answer extraction and grading hooks of its base class; `registry.py` maps the
task name given to `evalhub gen --tasks` to the class.

| Group | Content |
|---|---|
| `math/` | the mathematics benchmarks; the five used in the paper (`aime2026`, `aime2026_tr`, `aime2026_pt`, `tubitak_math2026`, `pt_exams_math`) plus the upstream sets, and the shared answer verifier in `math/verifier/` |
| `cot/` | the CoT judge tasks (`cot_judge`, `cot_judge_tr`, `cot_judge_pt`): one prompt template per language, applied to the answer-correct generations of a math run |
| `multilingual/`, `code/`, `general/`, `alignment/` | upstream benchmark loaders, unchanged |

`docs/benchmarks.md` lists the sources, sizes and licences of the mathematics
sets; `docs/adding_a_benchmark.md` shows how to add one so that both Pass@K and
CoT-Pass@K work.
