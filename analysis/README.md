# Analysis

The scripts, data tables and figures behind every number in
*Does CoT-Pass@k Really Check the CoT? A Multilingual Mathematical Audit*
(MRL Workshop, EMNLP 2026).

```
analysis/
  data/      published data layer (identifiers, counts, verdict tallies, metrics; no text)
  src/       one script per figure, table or quoted statistic
  figures/   the twelve figures of the paper, as shipped
  tables/    LaTeX tables and text summaries the paper quotes
```

## Running

```bash
pip install -e ".[report]" scipy statsmodels
export PYTHONPATH=analysis/src
python analysis/src/fig_error_by_benchmark.py      # one artefact
bash analysis/run_all.sh                           # everything that runs from data/
```

Every script reads from `analysis/data/` and writes to `analysis/figures/` or
`analysis/tables/`. Paths are resolved in `src/common.py`; nothing depends on
the working directory. Scripts marked *needs raw outputs* below read the model
generations or judge outputs, which are not released; they stop with a message
naming the missing input. Set `EVALHUB_RESULTS` and `EVALHUB_ERROR_STUDY_OUTPUT`
to point them at a local reproduction.

## From the paper to the code

| Paper | Script | Inputs | Output |
|---|---|---|---|
| Acceptance under injected errors per benchmark and judge (Section 5.1) | `fig_error_by_benchmark.py` | `error_injection_panel_3judges.csv` | `fig_error_by_benchmark.png` |
| Pass@64 and CoT-Pass@64 per solver and benchmark (Section 5.2) | `fig_gap_endpoints.py` | `gap_curve.csv` | `fig_gap_endpoints.png` |
| Acceptance under the three verification strategies (Appendix B) | `tab_error_rules.py`, needs the error-study judge outputs | `report_error_tasks.csv`, R1-distill judge cache | `tab_error_rules.tex` |
| Figure, acceptance by kind and size of the edit | `fig_edit_size.py`, `edit_size.py` | `corruption_log.csv`, `error_injection_panel_3judges.csv` | `fig_edit_size.png`, `tab_edit_size.tex`, `edit_size.txt` |
| Figure, language of the error-injection study | `fig_language_by_experiment.py` | `code_switching_panel.csv`, `judge_language*.csv` | `fig_language_exp1.png` |
| Tables, judge matrix (earlier and current generation) | `tab_judge_matrix.py` | `gap_curve.csv` | `tab_judge_matrix.tex` |
| Figure, the difference against k | `fig_k_ladder.py` | `gap_curve.csv` | `fig_k_ladder.png` |
| Figure, question-level bootstrap intervals | `bootstrap_questions.py`, `fig_bootstrap.py` | `verdicts.csv`, `gap_curve.csv` | `diff_bootstrap_ci.csv`, `fig_bootstrap.png` |
| Figure, acceptance rate of correct generations | `fig_acceptance.py` | `report_tasks.csv` | `fig_acceptance.png` |
| Figure, saturation and conditioning on c | `fig_saturation.py` | `report_tasks.csv` | `fig_saturation.png` |
| Figure and table, language of the chains | `fig_code_switching.py`, `code_switching_tables.py` | `code_switching_*.csv` | `fig_code_switching.png`, `tab_code_switching.tex`, `code_switching.txt` |
| Figure, language of the judge-matrix chains and judgments | `fig_language_by_experiment.py` | `code_switching_main.csv`, `judge_language.csv` | `fig_language_exp2.png` |
| Table, stops at the max-token limit | `tab_conditions.py` | `report.csv`, `gap_curve.csv` | `tab_conditions.tex` |
| Figure, judge capping | `fig_judge_capping.py` | `report.csv`, `verdicts.csv`, `gap_curve.csv` | `fig_judge_capping.png` |
| Table, the budget ladder | `tab_ladder.py` | `report.csv`, `gap_curve.csv` | `tab_ladder.tex` |
| Table, the same AIME problems in three languages | `tab_language.py` | `report.csv` | `tab_language.tex` |
| Figure, language under generation mode and budget | `fig_language_by_experiment.py` | `code_switching_budget.csv` | `fig_language_exp3.png` |
| Exact McNemar tests (Section 5.1) | `mcnemar_exact.py` | `error_injection_panel_3judges.csv` | `mcnemar_exact.txt` |
| Panel bootstrap intervals (Section 5.1) | `bootstrap_panel.py` | `error_injection_panel_3judges.csv` | `bootstrap_paired.txt` |
| Language versus benchmark test (Section 5.2) | `language_vs_benchmark.py` | `report_tasks.csv` | `language_vs_benchmark.txt` |
| Fair-set selection rule (22 cells) | `selection_rule.py` | `report_tasks.csv` | `selection_rule.txt` |
| Judging cost (Appendix A) | `judge_cost.py`, needs raw outputs | `results/` | `judge_cost.csv`, `judge_cost.txt` |
| Language identification of the chains | `code_switching.py`, needs raw outputs and the GlotLID model | `results/`, `verdicts.csv` | `code_switching_*.csv` |
| Language of the judges' output | `judge_language.py`, `judge_language_v4flash.py`, needs raw outputs | `results/`, `error_study/output/` | `judge_language*.csv` |
| Three-judge panel table | `panel_three_judges.py`, needs the error-study judge outputs | `report_error_tasks.csv`, judge caches | `error_injection_panel_3judges.csv` |
| The data layer itself | `build_data.py`, needs raw outputs | `results/`, `error_study/output/` | `data/*.csv` |

`palette.py` holds the colours shared by the figures; `vendor/` holds MaskLID
(MIT) and the FLORES-200 label list used by the language identification.

## Figures and tables

`figures/` holds the twelve PNGs exactly as they appear in the paper. They were
rendered with Matplotlib 3.7.0; regenerating them with another Matplotlib
version reproduces the same content with small antialiasing differences, so the
tests check that the scripts run and write the files rather than comparing
pixels. `tables/` holds the LaTeX tables and text summaries, which the tests
compare byte for byte with what the scripts produce (`pytest tests/analysis`).

## Data

See [`data/README.md`](data/README.md) for the column dictionary and the
statement of what is and is not released.
