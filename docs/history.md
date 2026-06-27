> [!Important]
> Due to the tightly coupled nature of LiveCodeBench's codebase, despite our efforts to integrate it with minimal cost, we inevitably had to incorporate a significant amount of code from the original repository.
>
> We only support codegeneration scenario of LiveCodeBench.

03/26/2025 update: We add a new mode for livecodebench, use [mini-judge](https://github.com/ysy-phoenix/mini-judge) as backend.
- The original LiveCodeBench would return upon encountering the first failed test case.
- whereas our new evaluation will execute all test cases.
- As a result, there is a significant difference in speed between the two approaches.
- By default, the original evaluation method is used, but you can modify it [here](../evalhub/benchmarks/code/livecodebench/__init__.py).

04/29/2025 update: Evaluation results of r1 recipe reproduction can be found in [docs/baseline.md](docs/baseline.md).

06/06/2025 update: We have added an experimental feature referencing verl's implementation: integration of multi-turn and tool calls.

06/30/2025 update: We have integrated most of the benchmarks from the Qwen3 technical report (excluding those that already have official implementations).

05/20/2026 update: Major refactor adding a result aggregation surface and
splitting the orchestrator script.

- New `evalhub report` sub-app with `aggregate`, `plot`, `highlights`, and
  `atlas` commands. Every `*_summary.json` / `*_cot_summary.json` under an
  `OUTPUT_ROOT` is parsed into a long-form pandas DataFrame and written to a
  master CSV; `plot` renders Pass@K curves, base-vs-CoT bars, a Pass@1
  heatmap, and CoT veto-rate bars; `highlights` and `atlas` render
  publication-ready PDFs. Lives in `evalhub/report/`.
- Bash orchestrator split into `scripts/run_eval_only.sh`,
  `scripts/run_judge_only.sh`, and `scripts/run_end_to_end.sh`, all sharing
  helpers from `scripts/lib/pipeline_common.sh`.
- New optional-deps group `report = [pandas, matplotlib, seaborn]`; `pandas`
  was added to the existing `base` group.
- New docs `docs/reporting.md` (CSV schema) and `docs/onboarding.md`
  (five-minute demo for new developers).

06/2026 update: One-folder-per-model (V5) results layout (sampling suffix on the
benchmark leaf); G-Pass@k / mG-Pass@k metrics; new benchmarks (`aime2026` EN/TR/PT,
`tubitak_math2026`, `pt_exams_math`); `highlights` / `atlas` report PDFs. The
interactive dashboard was dropped in favour of the static report surface.
