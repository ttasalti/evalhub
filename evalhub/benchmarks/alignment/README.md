# `evalhub/benchmarks/alignment/`

**Modified from the evalhub-original repository.**

Only change: the previous local-only judge benchmarks (`math_judge`,
`math_judge_tr`, `math_judge_pt`) were removed. Their replacement —
`CoTJudgeDataset`, parameterised by language — lives in
[`evalhub/benchmarks/cot/`](../cot/README.md) and is registered under the
names `cot_judge`, `cot_judge_tr`, `cot_judge_pt`.

The remaining contents (`ifeval/`, `writingbench/`) are unchanged from
upstream.
