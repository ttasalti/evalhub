# `evalhub/utils/`

**Modified from the evalhub-original repository.**

Upstream files (`logger.py`, `metrics.py`, `pbar.py`, `typer.py`) are unchanged.

Added for this project:

| File | Responsibility |
|---|---|
| `model_state.py` | Single source of truth for the three supported model states — `base`, `non-think`, `think` — and the mapping `(model_family, state) -> chat_template.jinja`. Importable Python API (`resolve_template_path`, `normalise_state`, `infer_state_from_model_name`) and a CLI entry point (`python -m evalhub.utils.model_state --model ... --state ...`) used by `scripts/run_cot_pass_at_k.sh`. |

Adding a new model family is one dataclass entry in `MODEL_FAMILIES` plus the
three Jinja templates under `scripts/templates/`.
