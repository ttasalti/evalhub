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
