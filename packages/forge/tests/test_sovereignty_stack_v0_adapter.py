"""Tests for the sovereignty-stack-v0 eval adapter.

Two runtime paths under test:

1. OpenAI-compat (``ANIMUS_SOVEREIGNTY_BASE_URL`` set) — direct httpx
   call to ``/v1/chat/completions``.
2. Forge provider (env var absent) — routes through ``get_provider()``
   with ``MODEL_UNDER_TEST`` as the default model.

Both paths are mocked so the test suite has no live-model dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _ensure_adapter_importable():
    """The adapter lives outside the src/ package tree — insert the
    package root once per test so ``eval_suites.adapters.*`` resolves.
    Mirrors what ``eval_cmd.py`` does at CLI runtime.
    """
    forge_root = Path(__file__).resolve().parent.parent
    if str(forge_root) not in sys.path:
        sys.path.insert(0, str(forge_root))


@pytest.fixture
def adapter_module():
    """Import the adapter fresh — each test mutates env, so we want a
    clean module-level state without lru_cache leakage."""
    import importlib

    import eval_suites.adapters.sovereignty_stack_v0 as mod

    importlib.reload(mod)
    return mod


@pytest.fixture
def _clear_env(monkeypatch):
    """Strip the two env vars the adapter reads so tests start fresh."""
    monkeypatch.delenv("MODEL_UNDER_TEST", raising=False)
    monkeypatch.delenv("ANIMUS_SOVEREIGNTY_BASE_URL", raising=False)


def _fake_oai_response(content: str, *, in_tokens: int = 10, out_tokens: int = 5) -> MagicMock:
    """Build an OpenAI-compat chat-completions response shape."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(
        return_value={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": in_tokens, "completion_tokens": out_tokens},
        }
    )
    return resp


def test_oai_compat_path_when_base_url_set(adapter_module, monkeypatch, _clear_env):
    """Env var present → adapter calls httpx, not the provider stack."""
    monkeypatch.setenv("ANIMUS_SOVEREIGNTY_BASE_URL", "http://127.0.0.1:8081")
    monkeypatch.setenv("MODEL_UNDER_TEST", "hauhaucs")

    fake_post = MagicMock(return_value=_fake_oai_response("hello world"))
    monkeypatch.setattr("httpx.post", fake_post)

    result = adapter_module.run_query({"query": "say hi", "max_tokens": 64})

    assert result == "hello world"
    assert fake_post.call_count == 1
    call = fake_post.call_args
    assert call.args[0] == "http://127.0.0.1:8081/v1/chat/completions"
    body = call.kwargs["json"]
    assert body["model"] == "hauhaucs"
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.2
    assert body["messages"] == [{"role": "user", "content": "say hi"}]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_oai_compat_strips_inline_thinking_block(adapter_module, monkeypatch, _clear_env):
    """If backend ignores ``enable_thinking=false`` and emits the block
    inline, the adapter strips it so the rubric judges the answer, not
    the reasoning trace."""
    monkeypatch.setenv("ANIMUS_SOVEREIGNTY_BASE_URL", "http://127.0.0.1:8081")
    monkeypatch.setattr(
        "httpx.post",
        MagicMock(return_value=_fake_oai_response("<think>step 1\nstep 2</think>final answer")),
    )

    result = adapter_module.run_query({"query": "q"})

    assert result == "final answer"


def test_string_input_treated_as_query_with_default_max_tokens(
    adapter_module, monkeypatch, _clear_env
):
    """A plain-string ``case.input`` is valid — defaults max_tokens to 512."""
    monkeypatch.setenv("ANIMUS_SOVEREIGNTY_BASE_URL", "http://127.0.0.1:8081")
    fake_post = MagicMock(return_value=_fake_oai_response("ok"))
    monkeypatch.setattr("httpx.post", fake_post)

    result = adapter_module.run_query("just a string")

    assert result == "ok"
    body = fake_post.call_args.kwargs["json"]
    assert body["messages"][0]["content"] == "just a string"
    assert body["max_tokens"] == 512


def test_provider_path_when_no_base_url(adapter_module, monkeypatch, _clear_env):
    """Env var absent → adapter calls ``get_provider().complete()``
    with the requested model passed *on the request*, not as a mutation
    of the shared singleton's default."""
    fake_resp = MagicMock(content="provider answer")
    fake_provider = MagicMock()
    fake_provider.config = MagicMock(default_model="placeholder")
    fake_provider.complete = MagicMock(return_value=fake_resp)
    monkeypatch.setattr(
        "animus_forge.providers.get_provider", MagicMock(return_value=fake_provider)
    )
    monkeypatch.setenv("MODEL_UNDER_TEST", "claude-sonnet-4-6")

    result = adapter_module.run_query({"query": "factual", "max_tokens": 256})

    assert result == "provider answer"
    # The adapter must NOT mutate the shared provider's default — that
    # would race with the judge in adapter+rubric mode.
    assert fake_provider.config.default_model == "placeholder"
    req = fake_provider.complete.call_args.args[0]
    assert req.prompt == "factual"
    assert req.model == "claude-sonnet-4-6"
    assert req.max_tokens == 256
    assert req.temperature == 0.2


def test_default_model_when_env_unset(adapter_module, monkeypatch, _clear_env):
    """When ``MODEL_UNDER_TEST`` is unset, the adapter falls back to
    ``DEFAULT_MODEL`` on the per-call CompletionRequest."""
    fake_provider = MagicMock()
    fake_provider.config = MagicMock(default_model="placeholder")
    fake_provider.complete = MagicMock(return_value=MagicMock(content="x"))
    monkeypatch.setattr(
        "animus_forge.providers.get_provider", MagicMock(return_value=fake_provider)
    )

    adapter_module.run_query({"query": "q"})

    req = fake_provider.complete.call_args.args[0]
    assert req.model == adapter_module.DEFAULT_MODEL
    # Confirm the provider's default was left untouched.
    assert fake_provider.config.default_model == "placeholder"
