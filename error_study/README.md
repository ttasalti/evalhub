# error_study: controlled error injection for CoT judges

The error-injection study reported in the paper. Correct, long thinking-mode
solutions from Qwen3.5-4B and Qwen3.5-9B are sampled across five benchmarks,
four controlled error types are injected (next to the unmodified control), and
every variant is judged with a CoT judge, which shows how reliably the judge
catches each error type and where in the reasoning it fails.

The package is read-only against `results/`; everything it produces lands in
`error_study/output/`. Run it from the repository root in the project
environment (needs `aiofiles`, `sympy`, `pandas` and `pyarrow`).

```bash
export PYTHONPATH=$PWD
python -m error_study.sample             # -> output/base_sample.parquet   (1500 rows)
python -m error_study.corrupt            # -> output/error_dataset.parquet (7500 rows) + logs
python -m error_study.build_judge_inputs # -> output/judge_inputs/*        (7115 items)
python -m error_study.verify --determinism   # assertions + rebuild-compare
python -m error_study.estimate_cost analytic # pre-estimate for the API judge (no API calls)
python -m pytest error_study/tests -q        # 61 unit tests
```

Both `base_sample` and `error_dataset` are written as `.parquet` and `.csv`
(the CSV quotes the multi-line LaTeX solution fields; round-trip verified).

## What is published

The generations and the judge outputs are not released. `output/` in this
repository holds only two aggregate tables, both for the DeepSeek V4-Flash judge
run: `output/cell_summary.csv` gives, per (model, benchmark, error type), the
number of solution-variants, the judge yes/no counts, and the acceptance and
veto rates under the any/majority/all rules; `output/per_question_summary.csv`
gives the same counts per (benchmark, question, error type). The three-judge
panel the paper reports is in `analysis/data/report_error_tasks.csv`.

The remaining outputs (`base_sample`, `error_dataset`, `judge_inputs/`,
`judge_*` raw files, `judge_output.*`) are regenerated locally from a
`results/` tree with the commands above.

## The grid (2 models x 5 benchmarks = 10 groups)

Sampled only from the highest max-token variant of each benchmark (no pooling:
lower caps mostly truncate thinking-mode solutions).

| benchmark | variant | answer type |
|---|---|---|
| aime2026 | max65536 | integer 0-999 |
| aime2026_pt | max32768 | integer 0-999 (Portuguese) |
| aime2026_tr | max32768 | integer 0-999 (Turkish) |
| tubitak_math2026 | max32768 | about 69% integer, rest LaTeX |
| pt_exams_math | max16384 | about 39% integer, rest LaTeX/decimal/percent |

## Stage 1: sampling (`sample.py`)

Per group: keep a generation iff it is correct (boxed answer re-graded against
the authoritative `_results.jsonl` ground truth with the evalhub grader),
`finish_reason == "stop"`, and `completion_tokens > 600`. Then round-robin
sample 150 rows with at most 6 solutions per question (at least 1 each),
maximising question diversity. Output: `output/base_sample.parquet` (1500
rows) with `model`, `state` (think), `task_id`, `question_text`,
`ground_truth`, `answer_type`, `solution_text` and `solution_len_tokens`
(`usage.completion_tokens`).

## Stage 2: error injection (`corrupt.py`)

Five rows per base solution (`clean` + four corruptions) give 7500 rows. One
wrong number is drawn per base solution and shared by `boxed_only` and
`consistent_error`; the two differ only in where the text changes.

- `boxed_only`: change only the number inside the final `\boxed{}`.
- `intermediate_error`: in the 40-70% region of the solution, change an existing
  number that is not in the question, not the answer, and not 0 or 1.
- `consistent_error`: change every occurrence of the answer (boxed and body) to
  the same wrong number as `boxed_only`. Skipped when the answer sits inside a
  larger number, equals a number in the question, or is not found literally.
- `truncated`: drop the last 25% of the text but keep the final `\boxed{}` and
  its number.
- `clean`: the unmodified control.

Wrong-number menu: target shares of 40% (plus or minus 1 or 2), 30% digit
transposition and 30% single-digit change, with post-checks (fits the answer
type, differs from the real answer and from every number in the question, and
for the intermediate edit is not 0 or 1) and a recorded fallback when a
category is infeasible. Non-integer answers are corrupted type-awarely (a digit
token inside the expression, e.g. `\frac{52}{5}` to `\frac{53}{5}`); purely
symbolic and multiple-choice answers skip the two answer edits with a note.

Outputs: `error_dataset.parquet` (7500 rows), `corruption_log.csv` (6000
corruption rows: task, group, type, old and new number, change location, menu
category, applied or skipped with reason) and `skipped_solutions.csv`.

## Stage 3: judge inputs and judging

`build_judge_inputs.py` writes 50 judge-input JSONL files in the evalhub
six-field schema (corrupted solution at
`raw_response.choices[0].message.content`) for the 7115 judged rows, plus a
`manifest.json` with a cache-friendly run order.

Two runners share the same inputs and sampling (n=3, temperature 0.6, top-p
0.95, 20480 completion tokens, thinking mode):

- `run_judge_study.sh`: the DeepSeek V4-Flash API judge, with staged prompt
  caching (clean variant first per group, then corruptions in decreasing-overlap
  order, three rounds with commit waits so later generations hit the cache).
  Reads the key from `JUDGE_API_KEY`; `--pilot` runs the cheapest group only.
- `run_judge_local.sh <judge_model> <run_tag>`: any vLLM-served judge (used for
  Qwen3.6-35B-A3B and DeepSeek-R1-0528-Qwen3-8B), writing next to the API run.

`estimate_cost.py` gives an analytic cost estimate for the API judge before
spending anything and measures realised token counts afterwards.

## Stage 4: analysis

`consolidate.py` merges the raw judge files into one table, `analyze.py`
produces `cell_summary.csv` and `per_question_summary.csv`, and
`report_error.py` writes the task- and question-grain CSVs the paper's analysis
scripts read. `run_analysis.sh <run_tag>` chains the three for a local judge
run and checks the shapes against the API reference run.

## Tests

`error_study/tests/` holds 61 pytest unit tests over the boxed-answer and
number helpers, the wrong-number menu, and all four injection primitives,
including regression tests for defects found in review (negative-answer skip,
change location pointing at a pre-existing number, truncation landing inside
the boxed answer, partial capture of a boundary-straddling number, grading
through the authoritative `check_correct`, and two verification checks).

## Notes

`digit_swap` uses a generalised N-digit transposition
(`config.DIGIT_SWAP_STRICT_2DIGIT=False`). This keeps the 30% target on this
three-digit-heavy corpus; the strict two-digit reading would make the category
fall back for most answers. The realised menu distribution still deviates from
40/30/30, because transposition is infeasible for many numbers (single-digit
intermediates, repeated digits, zero padding); the fallback fires and is
recorded (about 14.7%). `consistent_error` has a real skip rate (about 24%) by
the rules above.

Everything is reproducible from the fixed `SEED=42`; rebuilding yields
byte-identical outputs (`verify.py --determinism`).
