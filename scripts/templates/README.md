# `scripts/templates/`

**Newly created for this project.**

Jinja chat templates loaded by `vllm serve --chat-template ...` at server
startup. Selected by `evalhub.utils.model_state.resolve_template_path` based on
the `(model_family, state)` pair, where state ∈ `{base, non-think, think}`.

| File | Family | State |
|---|---|---|
| `qwen3.5-base.jinja` | Qwen | base |
| `qwen3.5-no-think.jinja` | Qwen | non-think |
| `qwen3.5-think.jinja` | Qwen | think |
| `gemma4-base.jinja` | Gemma | base |
| `gemma4-no-think.jinja` | Gemma | non-think |
| `gemma4-think.jinja` | Gemma | think |
| `ministral3-base.jinja` | Ministral / Mistral | base |
| `ministral3-instruct.jinja` | Ministral / Mistral | non-think |
| `ministral3-reasoning.jinja` | Ministral / Mistral | think |

To register a new family, add an entry to `MODEL_FAMILIES` in
`evalhub/utils/model_state.py` and drop the three Jinja files here.

## Note: tokenizer-default fallback

`scripts/lib/pipeline_common.sh::start_vllm` resolves the chat template via
`evalhub.utils.model_state` with `--allow-missing`. When the `(family, state)`
pair has no registered Jinja in this directory, the resolver returns an empty
string and the orchestrator **does not pass `--chat-template` to vLLM**. In
that case vLLM falls back to the model's own `tokenizer.chat_template` shipped
on the Hub.

This is a deliberate trade-off:

- **Pro:** new models work out of the box without us having to author a Jinja
  file first. The legacy `vllm3.sh` hard-errored (`exit 1`) on unknown models.
- **Con:** the resulting prompt format is whatever the model author baked into
  the tokenizer, which may differ from the format used by historical runs
  recorded under `results/`. If you are reproducing an older benchmark and
  need bit-identical prompts, register the family in `MODEL_FAMILIES` and add
  the three Jinja files here so the resolver hits a concrete template.
