import pytest

from evalhub.utils.model_state import (
    MODEL_STATES,
    infer_state_from_model_name,
    known_states_for_model,
    normalise_state,
    resolve_template_path,
)


def test_model_states_are_canonical():
    assert MODEL_STATES == ("base", "non-think", "think")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("base", "base"),
        ("BASE", "base"),
        ("non-think", "non-think"),
        ("non_think", "non-think"),
        ("no-think", "non-think"),
        ("instruct", "non-think"),
        ("chat", "non-think"),
        ("think", "think"),
        ("reasoning", "think"),
        ("thinking", "think"),
        ("pretrain", "base"),
        ("", "non-think"),
        (None, "non-think"),
    ],
)
def test_normalise_state(raw, expected):
    assert normalise_state(raw) == expected


def test_normalise_rejects_unknown():
    with pytest.raises(ValueError):
        normalise_state("turbo")


@pytest.mark.parametrize(
    "model,expected",
    [
        ("google/gemma-4-base", "base"),
        ("google/gemma-4-e2b", "base"),
        ("Qwen/Qwen3-7B-Instruct", "non-think"),
        ("DeepSeek-R1-Distill-Qwen-7B", "think"),
        ("Qwen/QwQ-32B", "think"),
        ("Qwen/Qwen3-7B-Reasoning", "think"),
    ],
)
def test_infer_state_from_model_name(model, expected):
    assert infer_state_from_model_name(model) == expected


def test_resolve_template_paths_for_each_state():
    for state in MODEL_STATES:
        path = resolve_template_path("Qwen/Qwen3-7B-Instruct", state)
        assert path is not None
        assert path.name.startswith("qwen3.5-")


def test_resolve_returns_none_for_unknown_family():
    assert resolve_template_path("openai/some-unknown-llm", "non-think") is None


def test_known_states_lists_three_for_supported_family():
    states = set(known_states_for_model("Qwen/Qwen3-7B-Instruct"))
    assert states == set(MODEL_STATES)
