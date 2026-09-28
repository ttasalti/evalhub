# Benchmark suite

The fork adds five mathematics benchmarks in three languages. Two are native
sets (a Turkish olympiad and Portuguese national exams), three are the same
AIME 2026 problems in English and in Turkish and Portuguese translation. All
Portuguese material is European Portuguese.

| Name | Language | Problems | Generations per problem | Origin | Where the data lives | Licence and terms |
|---|---|---|---|---|---|---|
| `aime2026` | EN | 30 | 64 | AIME 2026 I and II (MAA), as transcribed by MathArena | downloaded at run time from `MathArena/aime_2026` on Hugging Face | CC BY-NC-SA 4.0 (MathArena); problems copyright MAA |
| `aime2026_tr` | TR | 30 | 64 | Turkish translation of `aime2026` | `evalhub/benchmarks/math/aime2026_tr/aime2026_tr.parquet` | CC BY-NC-SA 4.0, derived from the MathArena set; credit MAA and MathArena |
| `aime2026_pt` | PT | 30 | 64 | Portuguese translation of `aime2026` | `evalhub/benchmarks/math/aime2026_pt/aime2026_pt.parquet` | CC BY-NC-SA 4.0, derived from the MathArena set; credit MAA and MathArena |
| `tubitak_math2026` | TR | 32 | 64 | 2026 TÜBİTAK National Mathematics Olympiad, first stage | `evalhub/benchmarks/math/tubitak_math2026/tubitak_math2026.csv` | copyright TÜBİTAK, all rights reserved; reproduced with attribution for non-commercial research, removed on request |
| `pt_exams_math` | PT | 166 | 16 | mathematics questions of PHEB (Tavares et al., LREC 2026), the Portuguese national exams 2006 to 2023, converted from multiple choice to open answer | `evalhub/benchmarks/math/pt_exams_math/pt_exams_math.csv` | derived from PHEB; see the PHEB repository for its terms |

## Translation protocol

The Turkish and Portuguese AIME sets were first machine-translated and then
audited by native speakers of each language: six for Portuguese and two for
Turkish, one of them an author. The audit corrected wording and mathematical
notation so that every translated problem states the same problem as the
English original; the reference answers are the MAA answers.

The Portuguese exam questions keep the original wording of PHEB. Their
multiple-choice format was converted to open-answer form by removing the
options and keeping the ground-truth value; two further graduate students,
native speakers of Portuguese and independent of the translation audit, checked
the converted questions and answers.

## Prompts

Every problem is followed by the same instruction in the problem's language:

| Language | Solver instruction |
|---|---|
| EN | Let's think step by step and output the final answer within \boxed{}. |
| TR | Adım adım düşün ve nihai cevabı \boxed{} içerisinde ver. |
| PT | Vamos pensar passo a passo e apresentar a resposta final dentro de \boxed{}. |

The judge prompt of Wen et al. (2025) is used unchanged for English and in a
Turkish and a Portuguese variant for the other benchmarks (`cot_judge`,
`cot_judge_tr`, `cot_judge_pt` in `evalhub/benchmarks/cot/`). The prompts are
frozen; see `pyproject.toml` (`extend-exclude`) for the files the formatter
leaves untouched for that reason.

## Grading

All five benchmarks are graded by the shared math verifier
(`evalhub/benchmarks/math/verifier/`), which compares the boxed answer with the
reference through sympy, mathd and dapo checkers. The Portuguese exams add a
tolerance for decimal and percentage answers in their loader.

## Contamination note

The TÜBİTAK olympiad was held on 16 May 2026 and AIME 2026 in February 2026,
both after the release of every model evaluated in the paper. The Portuguese
exams were written between 2006 and 2023 and are the most exposed of the five
sets.

## Adding or updating a benchmark

Loaders live under `evalhub/benchmarks/math/<name>/__init__.py` and register
themselves with `@register_dataset`. evalhub caches loaded tasks under
`~/.cache/evalhub/`; clear it after editing a data file.

## Inherited from upstream

The loaders below come from the upstream project and are kept so that the fork
stays a superset of it. They were not used in the paper and have not been
exercised beyond the upstream tests; treat them as upstream code.

| Group | Loaders |
|---|---|
| math | `aime2024`, `aime2025`, `autologi`, `gsm8k`, `hendrycks_math`, `math500`, `zebralogic` |
| multilingual | `include`, `mlogiqa`, `mmmlu`, `mt_aime2024`, `polymath` |
| code | `bigcodebench`, `humaneval`, `livecodebench` |
| general | `ceval`, `gpqa`, `mmlu_redux` |
| alignment | `ifeval`, `writingbench` |

The CoT-Pass@K judging stage (`cot` group) works with any of the math loaders
that produce a boxed answer.
