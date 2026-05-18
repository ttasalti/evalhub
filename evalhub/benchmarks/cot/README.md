# `evalhub/benchmarks/cot/` — LLM-as-a-Judge dataset for CoT-Pass@K

**Newly created for this project.** Replaces the three near-duplicate
`math_judge`, `math_judge_tr`, `math_judge_pt` packages previously kept under
`evalhub/benchmarks/alignment/`.

| File | Responsibility |
|---|---|
| `prompts.py` | One `{question, solution}` prompt template per language. Adding a new language is a single dictionary entry. |
| `judge.py`   | `CoTJudgeDataset` — parameterised by `language`. Reads a JSONL of correct base generations (produced by `evalhub cot extract`), resolves the original question by looking up the source dataset in `DATASET_MAP`, and emits one judge task per generation. Registers three names (`cot_judge`, `cot_judge_tr`, `cot_judge_pt`). |

The judge:

- `extract_solution(response)` returns `extract_boxed_answer(...).lower()` so
  `\boxed{yes}` / `\boxed{no}` becomes `"yes"` / `"no"`.
- `check_correct(answer, _, _)` returns `answer == "yes"`, so the existing
  `evalhub eval` produces a per-generation `correct: bool` column that the
  `evalhub cot aggregate` step then majority-votes.

Add a new judge language by:

1. Adding the prompt to `prompts.py::JUDGE_PROMPTS`.
2. Adding the new short name to `judge.py::COT_JUDGE_VARIANTS`.

No new subclass is required.
