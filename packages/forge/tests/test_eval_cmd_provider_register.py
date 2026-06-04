"""Tests for the eval-CLI provider auto-registration helper.

``_ensure_provider_for_model`` runs at the top of ``eval run`` so both
the UUT adapter and the rubric judge have a default provider to talk
to. Three precedence rules under test:

1. ``ANIMUS_SOVEREIGNTY_BASE_URL`` set → register llamacpp regardless of
   ``--model`` hint or other env keys.
2. ``--model`` containing ``claude`` → register anthropic only (don't
   fall through to OpenAI on missing Anthropic key).
3. ``--model`` containing ``gpt`` → register openai only.
4. Idempotency: re-calling with an already-configured manager is a
   no-op.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from animus_forge.cli.commands.eval_cmd import _ensure_provider_for_model
from animus_forge.providers.base import ProviderType
from animus_forge.providers.manager import reset_manager


@pytest.fixture(autouse=True)
def _isolated_manager():
    """Each test gets a fresh, empty provider manager."""
    reset_manager()
    yield
    reset_manager()


@pytest.fixture
def _clear_env(monkeypatch):
    """Strip every env var the helper reads."""
    for key in (
        "ANIMUS_SOVEREIGNTY_BASE_URL",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_sovereignty_url_registers_llamacpp(monkeypatch, _clear_env):
    """Highest-precedence path: env var set → llamacpp registered, no
    cloud lookup."""
    monkeypatch.setenv("ANIMUS_SOVEREIGNTY_BASE_URL", "http://127.0.0.1:8081")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-should-not-be-used")

    _ensure_provider_for_model(model="hauhaucs")

    from animus_forge.providers import get_provider

    provider = get_provider()
    assert provider is not None
    assert provider.provider_type == ProviderType.LLAMACPP
    assert provider.config.base_url == "http://127.0.0.1:8081/v1"
    assert provider.config.default_model == "hauhaucs"


def test_sovereignty_url_appends_v1_suffix(monkeypatch, _clear_env):
    """Trailing slash + already-v1 are both handled cleanly."""
    monkeypatch.setenv("ANIMUS_SOVEREIGNTY_BASE_URL", "http://127.0.0.1:8081/v1/")

    _ensure_provider_for_model(model="hauhaucs")

    from animus_forge.providers import get_provider

    assert get_provider().config.base_url == "http://127.0.0.1:8081/v1"


def test_claude_model_registers_anthropic(monkeypatch, _clear_env):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")

    _ensure_provider_for_model(model="claude-haiku-4-5")

    from animus_forge.providers import get_provider

    provider = get_provider()
    assert provider is not None
    assert provider.provider_type == ProviderType.ANTHROPIC


def test_gpt_model_registers_openai(monkeypatch, _clear_env):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-xxx")

    _ensure_provider_for_model(model="gpt-4o")

    from animus_forge.providers import get_provider

    provider = get_provider()
    assert provider is not None
    assert provider.provider_type == ProviderType.OPENAI


def test_idempotent_when_already_configured(monkeypatch, _clear_env):
    """If a provider is already the default, the helper bails out."""
    from animus_forge.providers import get_manager

    fake_default = MagicMock()
    mgr = get_manager()
    monkeypatch.setattr(mgr, "get_default", lambda: fake_default)
    register_spy = MagicMock()
    monkeypatch.setattr(mgr, "register", register_spy)

    monkeypatch.setenv("ANIMUS_SOVEREIGNTY_BASE_URL", "http://127.0.0.1:8081")

    _ensure_provider_for_model(model="hauhaucs")

    register_spy.assert_not_called()


def test_no_keys_no_env_is_silent_noop(monkeypatch, _clear_env):
    """When nothing is reachable, the helper silently returns. The
    caller will see ``get_provider() is None`` and surface a clearer
    error downstream (the existing non-mock branch already does)."""
    _ensure_provider_for_model(model="claude-haiku-4-5")

    from animus_forge.providers import get_provider

    assert get_provider() is None
