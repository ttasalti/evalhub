"""Resolution of model + state -> chat-template Jinja file.

Three explicit states are supported:

  * ``base``      — pretraining checkpoint, completion-style template.
  * ``non-think`` — instruct/chat checkpoint, plain reply template.
  * ``think``     — chat checkpoint configured to emit a ``<think>`` reasoning
                    block before the answer.

Resolution is a two-key lookup ``(model_family, state)`` against
:data:`MODEL_STATE_REGISTRY`. The shell launcher delegates to
:func:`resolve_template_path` so all selection logic lives in one place; the
generator delegates to the same function to log which template a run used.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_ROOT = PROJECT_ROOT / "scripts" / "templates"

STATE_BASE = "base"
STATE_NON_THINK = "non-think"
STATE_THINK = "think"
MODEL_STATES: tuple[str, ...] = (STATE_BASE, STATE_NON_THINK, STATE_THINK)


@dataclass(frozen=True)
class ModelFamily:
    name: str
    aliases: tuple[str, ...]
    base_template: str | None
    non_think_template: str | None
    think_template: str | None

    def template_for(self, state: str) -> str | None:
        return {
            STATE_BASE: self.base_template,
            STATE_NON_THINK: self.non_think_template,
            STATE_THINK: self.think_template,
        }[state]


MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        name="qwen",
        aliases=("qwen", "qwen2", "qwen3"),
        base_template="qwen3.5-base.jinja",
        non_think_template="qwen3.5-no-think.jinja",
        think_template="qwen3.5-think.jinja",
    ),
    ModelFamily(
        name="gemma",
        aliases=("gemma",),
        base_template="gemma4-base.jinja",
        non_think_template="gemma4-no-think.jinja",
        think_template="gemma4-think.jinja",
    ),
    ModelFamily(
        name="ministral",
        aliases=("ministral", "mistral"),
        base_template="ministral3-base.jinja",
        non_think_template="ministral3-instruct.jinja",
        think_template="ministral3-reasoning.jinja",
    ),
)


def normalise_state(state: str | None) -> str:
    if state is None or state == "":
        return STATE_NON_THINK
    value = state.strip().lower().replace("_", "-")
    if value in MODEL_STATES:
        return value
    if value in {"reasoning", "thinking"}:
        return STATE_THINK
    if value in {"instruct", "chat", "no-think", "non-think"}:
        return STATE_NON_THINK
    if value == "pretrain":
        return STATE_BASE
    raise ValueError(f"Unknown model state {state!r}. Allowed: {MODEL_STATES}")


def infer_state_from_model_name(model_path: str) -> str:
    name = model_path.lower()
    if any(token in name for token in ("base", "e2b", "e4b", "pretrain")):
        return STATE_BASE
    if any(token in name for token in ("reasoning", "think", "r1", "qwq")):
        return STATE_THINK
    return STATE_NON_THINK


def _match_family(model_path: str) -> ModelFamily | None:
    name = model_path.lower()
    for family in MODEL_FAMILIES:
        for alias in family.aliases:
            if alias in name:
                return family
    return None


def resolve_template_path(
    model_path: str,
    state: str | None,
    template_root: os.PathLike | None = None,
) -> Path | None:
    """Return the absolute path of the Jinja template for ``(model, state)``.

    Returns ``None`` if no family is registered for this model — the caller is
    responsible for either failing or letting vLLM apply the tokenizer's
    built-in template.
    """
    resolved_state = normalise_state(state)
    family = _match_family(model_path)
    if family is None:
        return None
    relative = family.template_for(resolved_state)
    if relative is None:
        return None
    root = Path(template_root) if template_root is not None else TEMPLATE_ROOT
    return (root / relative).resolve()


def known_states_for_model(model_path: str) -> Iterable[str]:
    family = _match_family(model_path)
    if family is None:
        return ()
    return tuple(state for state in MODEL_STATES if family.template_for(state) is not None)


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evalhub.utils.model_state",
        description="Resolve a chat template for a (model, state) pair.",
    )
    parser.add_argument("--model", required=True, help="Model name or path.")
    parser.add_argument(
        "--state",
        default=None,
        help=f"One of {MODEL_STATES}. Defaults to inference from model name.",
    )
    parser.add_argument(
        "--template-root",
        default=None,
        help="Override the directory containing Jinja templates.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit 0 with empty stdout if no template is registered.",
    )
    args = parser.parse_args(argv)

    state = args.state if args.state is not None else infer_state_from_model_name(args.model)
    path = resolve_template_path(args.model, state, template_root=args.template_root)
    if path is None:
        if args.allow_missing:
            return 0
        print(f"No template registered for model={args.model!r} state={state!r}", file=sys.stderr)
        return 1
    if not path.exists():
        print(f"Template path resolved but does not exist on disk: {path}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
