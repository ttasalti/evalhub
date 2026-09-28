"""Regression tests for the request payload built by ``LLMGenerator.complete``.

Some OpenAI-compatible endpoints (notably DeepSeek) silently ignore
``max_completion_tokens`` and fall back to their own model ceiling, so the
requested cap is not enforced. litellm forwards the field verbatim (it does not
translate it to ``max_tokens``). The generator therefore mirrors the cap into
``max_tokens``, which both vLLM and DeepSeek honour. These tests pin that
behaviour so the cap can never silently stop being sent again.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from evalhub.inference.generator import LLMGenerator
from evalhub.inference.schemas import GenerationConfig, SamplingParams


def _fake_response():
    """Minimal stand-in for a litellm ModelResponse used by ``complete``."""
    choice = SimpleNamespace(finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model_dump=lambda: {"choices": [{"finish_reason": "stop"}]},
    )


def _make_generator(**sampling_kwargs):
    sampling = SamplingParams(model="hosted_vllm/some-model", **sampling_kwargs)
    config = GenerationConfig(tasks=["dummy"], sampling_params=sampling)
    return LLMGenerator(config=config)


def _capture_completion_kwargs(generator):
    """Run ``complete`` once with a mocked acompletion; return the call kwargs."""
    mock = AsyncMock(return_value=_fake_response())
    with patch("evalhub.inference.generator.acompletion", mock):
        asyncio.run(generator.complete([{"role": "user", "content": "hi"}]))
    assert mock.await_count == 1
    return mock.await_args.kwargs


def test_max_tokens_mirrors_max_completion_tokens():
    """The cap must be sent as BOTH fields so DeepSeek-style endpoints honour it."""
    gen = _make_generator(max_completion_tokens=5000)
    kwargs = _capture_completion_kwargs(gen)
    assert kwargs["max_completion_tokens"] == 5000
    assert kwargs["max_tokens"] == 5000, "max_tokens must mirror the cap (DeepSeek ignores max_completion_tokens)"


def test_think_cap_mirrored_with_reasoning_controls():
    """Mirroring still happens alongside reasoning_effort / extra_body forwarding."""
    gen = _make_generator(
        max_completion_tokens=32768,
        reasoning_effort="high",
        extra_body='{"thinking": {"type": "enabled"}}',
    )
    kwargs = _capture_completion_kwargs(gen)
    assert kwargs["max_tokens"] == 32768
    assert kwargs["max_completion_tokens"] == 32768
    # reasoning controls must survive the mirroring change
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


def test_none_reasoning_controls_not_sent():
    """'none'/unset reasoning controls are still dropped (mirroring did not regress this)."""
    gen = _make_generator(max_completion_tokens=16384, reasoning_effort="none")
    kwargs = _capture_completion_kwargs(gen)
    assert kwargs["max_tokens"] == 16384
    assert "reasoning_effort" not in kwargs
    assert "extra_body" not in kwargs
