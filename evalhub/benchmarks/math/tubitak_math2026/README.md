# `tubitak_math2026`

The 32 first-stage problems of the 2026 TÜBİTAK National Mathematics Olympiad
(Türkiye), as a Turkish open-answer benchmark.

## Source and terms

The problems are published by TÜBİTAK on its past-exam page,
<https://bilimolimpiyatlari.tubitak.gov.tr/tr/gecmis-sinav-sorulari>, and were
transcribed from the official PDF for this benchmark. The material belongs to
TÜBİTAK (all rights reserved). It is reproduced here, with attribution, for
non-commercial research use only and will be removed on request from the rights
holder. Answers were transcribed from the official key.

## Data format

`tubitak_math2026.csv` (UTF-8, three columns):

| Column | Content |
|---|---|
| `Question_Number` | 1 to 32 |
| `Question_Text` | Turkish problem statement; may contain inline LaTeX (`$...$`) |
| `Answer` | reference answer; an optional `$...$` wrapper is stripped by the loader |

Answers mix integers and LaTeX (`$2$`, `$\frac{52}{5}$`, `$105^\circ$`,
`$3\sqrt{10}$`, `$8^5$`, `$-\frac{49}{8}$`). Grading goes through the shared
math verifier in [`evalhub/benchmarks/math/verifier/`](../verifier/) (sympy,
mathd and dapo checkers), which handles all of these forms.

The solver instruction appended to every problem is the Turkish counterpart of
the English one: "Adım adım düşün ve nihai cevabı \boxed{} içerisinde ver."

## Updating the CSV

evalhub caches loaded tasks. After editing the CSV, clear the cache so the next
`evalhub gen` or `evalhub eval` re-reads it:

```bash
rm -rf ~/.cache/evalhub/
```

## Pass@K

```bash
evalhub gen --model hosted_vllm/Qwen/Qwen3.5-0.8B-Base --tasks tubitak_math2026 \
    --temperature 0.6 --n-samples 8 --output-dir results/tubitak/
evalhub eval --tasks tubitak_math2026 \
    --solutions results/tubitak/tubitak_math2026.jsonl --output-dir results/tubitak/
```

## CoT-Pass@K

The Turkish judge prompt (`cot_judge_tr`) is required:

```bash
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/tubitak_math2026.env \
    --model X --judge Y
# JUDGE_TASK=cot_judge_tr is set in that config file
```

or with the generic config:

```bash
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/base.env \
    --model X --judge Y --benchmark tubitak_math2026 \
    --set JUDGE_TASK=cot_judge_tr
```
