# Adding a benchmark for Pass@K and CoT-Pass@K

How to add a mathematics benchmark so that it works in both stages of the
pipeline: plain Pass@K (`evalhub gen` followed by `evalhub eval`) and the
judged CoT-Pass@K (`evalhub cot extract`, the judge run, `evalhub cot finalize`).
The `tubitak_math2026` loader serves as the template; it is a local CSV with
mixed integer and LaTeX answers in a non-English language, which covers every
step below.

## 1. Put the data in place

Create `evalhub/benchmarks/math/<name>/` and place the data file next to the
loader (`<name>.csv` or `<name>.parquet`). Files with these extensions are
packaged automatically (`pyproject.toml`, `package-data`). The file needs one
row per problem with at least a problem text and a reference answer; keep
LaTeX inline with `$...$`. If the source is a public dataset, the loader can
instead download it with `datasets.load_dataset` (see `aime2026/__init__.py`).

Record the source, its licence and any translation or conversion protocol in
`docs/benchmarks.md`, and add a short `README.md` in the loader directory when
the terms need explaining (see `tubitak_math2026/README.md`).

## 2. Write the loader

`evalhub/benchmarks/math/<name>/__init__.py`:

```python
import os
from typing import Any

import pandas as pd

from evalhub.benchmarks.base import GroundTruth, Task
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

MY_BENCH = "my_bench"

@register_dataset((MY_BENCH, None, True))
class MyBenchDataset(MathDataset):
    """One-line description: language, source, answer types."""

    def __init__(self, name: str = MY_BENCH, **kwargs):
        super().__init__(name, **kwargs)

    def load_tasks(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{MY_BENCH}.csv")
        df = pd.read_csv(path, encoding="utf-8")
        for _, row in df.iterrows():
            task_id = f"{MY_BENCH}/{row['id']}"
            self.add_task(Task(task_id=task_id, prompt=self.format_prompt(row.to_dict())))
            self.add_groundtruth(GroundTruth(task_id=task_id, answer=str(row["answer"]).strip()))

    def format_prompt(self, item: dict[str, Any]) -> str:
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        return item["problem"].strip() + " " + instruction
```

Three rules matter for the judged stage.

Task ids start with the registered name followed by `/` (`my_bench/17`). The
judge derives the source benchmark from the id
(`task_id.split("/")[0].lower().replace("-", "_")`) to look up the original
question, so the prefix has to equal the `register_dataset` name.

The instruction sentence ends with a boxed answer request in the problem's
language; the extractor reads the last `\boxed{}`. Use the existing sentences
for English, Turkish and Portuguese (`docs/benchmarks.md`, "Prompts") so that
results stay comparable.

Answers go through the shared verifier (`MathDataset.check_correct`,
`evalhub/benchmarks/math/verifier/`). Override `patch()` only for a
benchmark-specific tolerance, as `pt_exams_math` does for decimals and
percentages, and strip wrappers like `$...$` in the loader.

Import the loader in `evalhub/benchmarks/math/__init__.py` so the registration
runs when the package is imported.

## 3. Pick or add the judge prompt

The judge reads the original question and the solver's chain and answers
`\boxed{yes}` or `\boxed{no}`. Three judge tasks exist, one prompt template per
language: `cot_judge` (English), `cot_judge_tr` (Turkish), `cot_judge_pt`
(Portuguese). Use the one whose language matches the benchmark by setting
`JUDGE_TASK` in the run config.

For a new language, add a template to `evalhub/benchmarks/cot/prompts.py`
(`JUDGE_PROMPTS`) and a name to `COT_JUDGE_VARIANTS` in
`evalhub/benchmarks/cot/judge.py`. Do not edit the existing templates; they
are the prompts the paper reports and are excluded from the formatter for that
reason (`pyproject.toml`, `extend-exclude`).

## 4. Run Pass@K

```bash
evalhub gen --model hosted_vllm/Qwen/Qwen3.5-0.8B-Base --tasks my_bench \
    --temperature 0.6 --n-samples 8 --output-dir results/my_bench/
evalhub eval --tasks my_bench \
    --solutions results/my_bench/my_bench.jsonl --output-dir results/my_bench/
```

`evalhub eval` writes `my_bench_results.jsonl` (per generation), `my_bench_summary.json`
(Pass@k, G-Pass@k, mG-Pass@k) and `my_bench_per_task.csv`. Clear
`~/.cache/evalhub/` after editing the data file; loaded tasks are cached.

## 5. Run CoT-Pass@K

Copy a config and set the judge task:

```bash
cp scripts/configs/tubitak_math2026.env scripts/configs/my_bench.env
# edit: BENCHMARK="my_bench", JUDGE_TASK="cot_judge"   (or _tr / _pt)
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/my_bench.env \
    --model Qwen/Qwen3.5-4B --judge Qwen/Qwen3.6-35B-A3B
```

`run_end_to_end.sh` runs generation, evaluation, `evalhub cot extract` (the
answer-correct generations), the judge with `JUDGE_N_SAMPLES` verdicts per
generation, `evalhub cot finalize` (majority vote and veto) and the report
upsert. The judged results land under
`<OUTPUT_ROOT>/<state>/<solver>/judged_by/<judge>__.../my_bench__t0.6__max16384__n8/`
with `my_bench_cot_summary.json` holding CoT-Pass@k. To judge an existing base
run, use `run_judge_only.sh` with `BASE_RESULTS_DIR`.

## 6. Make the report know the benchmark

`evalhub report` works without changes, but two optional tables give the
benchmark a proper label and language: `LANG` in `evalhub/report/labels.py`
and `BENCH_ORDER` in `evalhub/report/plots.py`
for the plot order. Without them the benchmark appears under its raw name.

## 7. Test

Copy `tests/math/test_tubitak_math2026.py` and adapt it: registry entry, task
count, the task-id prefix rule of step 2, answer normalisation, and one
`grade_answer` check per answer type in the data. Run `pytest tests/math -q`.
