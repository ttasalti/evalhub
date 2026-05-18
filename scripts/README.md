# `scripts/`

**Modified from the evalhub-original repository.**

The folder was reset to evalhub-original-style minimalism. All legacy Slurm
orchestrators and helper sub-pipelines (`vllm*.sh`, `judge_vllm3.sh`,
`judge_only_all.sh`, `cot_judge_pipeline/`, `pass_k_pipeline/`,
`configs/`, `utils/`) were deleted because their logic now lives inside the
Python package and is exercised via `evalhub` CLI commands.

Newly created for this project:

| File | Purpose |
|---|---|
| `run_cot_pass_at_k.sh` | Single end-to-end CoT-Pass@K pipeline. Reads an env file, starts a vLLM server with the target model + its registered chat template, runs `evalhub gen` + `evalhub eval`, then repeats for the judge model, then calls `evalhub cot finalize`. Environment-agnostic — Slurm wrapping is left to the caller. |
| `cot_pipeline.env.example` | Annotated default values. Copy to `cot_pipeline.env` and edit. |
| `templates/` | Jinja chat templates per `(model_family, state)`. Selected by `evalhub.utils.model_state` and passed to `vllm serve --chat-template`. |

Usage:

```bash
cp scripts/cot_pipeline.env.example scripts/cot_pipeline.env
# edit scripts/cot_pipeline.env ...
scripts/run_cot_pass_at_k.sh
```
