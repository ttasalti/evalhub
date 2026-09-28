# Data layer

These tables are the published inputs of every number, table and figure in the
paper. They hold identifiers, counts, verdict tallies and metrics only. No
generation text, no judge text and no problem text is included; the raw
experiment outputs under `results/` and `error_study/output/` are not released.

`build_data.py` produces the first block from `results/` and the error-study
outputs. The second block is derived from the first by the scripts named in
each row.

## Built from the experiment outputs

| File | Grain | Notes |
|---|---|---|
| `report.csv` | one row per run (solver, mode, benchmark, solver budget, judge) | Pass@k, G-Pass@k and mG-Pass@k as wide columns, with `__any` / `__all` suffixes for the alternative approval rules; counts (`n_tasks`, `n_generations`, `true_count`, `n_veto`); solver and judge cap rates; `generation` (previous = Qwen2.5, current = the rest), `judge_short`, `exp`. A run enters only when every task has the full `n_samples` generations (64; 16 on the Portuguese exams). |
| `report_tasks.csv` | one row per (run, task) | `true_count`, `false_count` and `n_veto` per question; the source of the acceptance and saturation figures and of the language-versus-benchmark test. |
| `verdicts.csv` | one row per (answer-correct solution, judge) | `solution_id` = `solver|state|benchmark|solver_max_tokens|task_id|g<idx>`, shared across judges. `n_yes`, `n_no`, `n_missing` tally the three judge generations; `n_missing_capped` counts the missing verdicts caused by the judge hitting its own token budget. `approved_maj` is the stored majority decision that produced the reported metrics; `approved_any` and `approved_all` are the alternative rules. |
| `report_error_tasks.csv` | one row per (corrupted solution variant, judge) | The error-injection study: `condition` in {clean, boxed_only, intermediate_error, consistent_error, truncated}; `n_yes`, `n_no`, `n_missing`, `correct_*` and `veto_*` under the three approval rules; `answer_correct` says whether the variant's final answer is still right; `panel` marks the balanced 720-solution panel (72 per cell, seed 0) used in the paper. |
| `corruption_log.csv` | one row per corruption attempt | Edit design without text: `old_number`, `new_number`, `wrong_number_menu`, `edit_position_pct`, `solution_length` in tokens, `corruption_applied`, `skip_reason`. |

## Derived

| File | Produced by | Content |
|---|---|---|
| `gap_curve.csv` | `build_data.py derived` | one row per (run, judge, k): `pass_at_k`, `cot_pass_at_k`, `cot_gap`; `pass_at_k` does not depend on the judge and repeats across judge rows. |
| `judge_paired.csv` | `build_data.py derived` | one row per experiment: veto rates and judge cap rates for the two judges that cover the full grid; experiments with no correct answer carry NA. |
| `diff_bootstrap_ci.csv` | `bootstrap_questions.py` | question-level bootstrap intervals (10,000 resamples) of Pass@64 minus CoT-Pass@64 per generation, solver and cell. |
| `error_injection_panel_3judges.csv` | `panel_three_judges.py` | the 720-solution panel judged by V4-Flash, Qwen3.6 and R1-distill: `correct_maj` per (judge, solution, condition). |
| `code_switching_main.csv` | `code_switching.py` | language of every answer-correct chain of the ten judge-matrix solvers at the 16k budget: letter shares per language from the segment layer, MaskLID labels from the chain layer, and the two judges' decisions. |
| `code_switching_panel.csv` | `code_switching.py` | the same for the 720 panel solutions, with the clean and final-answer decisions of the three judges. |
| `code_switching_ptexams.csv`, `code_switching_budget.csv` | `code_switching.py --corpus ptexams`, `--corpus budget` | the same for the Portuguese exams and for the budget-ladder cells. |
| `judge_language.csv`, `judge_language_v4flash.csv` | `judge_language.py`, `judge_language_v4flash.py` | language of the judges' own output on sampled judgments (segment layer). |

## Conventions

`truncated` names an error-injection condition; running out of token budget is
called capping everywhere (`solver_cap_rate`, `judge_cap_rate`,
`n_missing_capped`). The short judge names stand for `Qwen3.6` =
Qwen3.6-35B-A3B, `Gemma4` = Gemma-4-26B-A4B-it, `V4-Flash` = DeepSeek V4-Flash
(April 2026 preview build) and `R1-distill` = DeepSeek-R1-0528-Qwen3-8B. The
pooled veto thresholds quoted in the paper use the 16,384-token solver budget
and exclude the Qwen2.5 solvers.
