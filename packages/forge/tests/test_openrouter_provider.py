"""Tests for the OpenRouter provider (open-weights only, PUBLIC-only egress).

Security-relevant: the egress + open-weights guarantees here are the reason
this provider exists. See docs/specs/openrouter-provider.md.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from animus_types import Sensitivity

from animus_forge.network import EgressDeniedError
from animus_forge.providers.base import (
    CompletionRequest,
    ModelTier,
    ProviderConfig,
    ProviderType,
)
from animus_forge.providers.manager import ProviderManager
from animus_forge.providers.openrouter_provider import (
    DEFAULT_TIER_MODELS,
    OPENROUTER_BASE_URL,
    OpenRouterProvider,
    OpenWeightViolationError,
)


def _provider(api_key: str = "sk-or-test") -> OpenRouterProvider:
    cfg = ProviderConfig(provider_type=ProviderType.OPENROUTER, api_key=api_key)
    return OpenRouterProvider(config=cfg)


def _fake_response(content: str = "hi", model: str = "meta-llama/llama-3.1-8b-instruct"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        model=model,
        usage=SimpleNamespace(total_tokens=10, prompt_tokens=4, completion_tokens=6),
        id="gen-123",
    )


class TestContract:
    def test_identity(self):
        p = _provider()
        assert p.name == "openrouter"
        assert p.provider_type is ProviderType.OPENROUTER
        assert p.default_model == DEFAULT_TIER_MODELS[ModelTier.STANDARD]

    def test_base_url_defaults(self):
        p = _provider()
        assert p.config.base_url == OPENROUTER_BASE_URL

    def test_is_configured(self):
        assert _provider("sk-or-test").is_configured() is True
        assert _provider("").is_configured() is False

    def test_registered_in_manager(self):
        assert ProviderManager.PROVIDER_CLASSES[ProviderType.OPENROUTER] is OpenRouterProvider


class TestOpenWeightGuarantee:
    @pytest.mark.parametrize(
        "slug",
        [
            "openai/gpt-4o",
            "anthropic/claude-3-5-sonnet",
            "google/gemini-pro-1.5",
            "x-ai/grok-2",
            "OpenAI/GPT-4O",  # case-insensitive
        ],
    )
    def test_closed_vendor_rejected(self, slug):
        p = _provider()
        with pytest.raises(OpenWeightViolationError):
            p._assert_open_weight(slug)

    @pytest.mark.parametrize(
        "slug",
        [
            "meta-llama/llama-3.1-70b-instruct",
            "deepseek/deepseek-r1",
            "qwen/qwen-2.5-72b-instruct",
            "mistralai/mistral-nemo",
        ],
    )
    def test_open_weight_allowed(self, slug):
        p = _provider()
        p._assert_open_weight(slug)  # no raise

    def test_override_prefixes_via_metadata(self):
        cfg = ProviderConfig(
            provider_type=ProviderType.OPENROUTER,
            api_key="k",
            metadata={"closed_vendor_prefixes": ["meta-llama/"]},
        )
        p = OpenRouterProvider(config=cfg)
        with pytest.raises(OpenWeightViolationError):
            p._assert_open_weight("meta-llama/llama-3.1-8b-instruct")


class TestTierResolution:
    def test_explicit_model_wins(self):
        p = _provider()
        req = CompletionRequest(prompt="x", model="deepseek/deepseek-r1")
        assert p._resolve_model(req) == "deepseek/deepseek-r1"

    def test_tier_maps_to_default(self):
        p = _provider()
        req = CompletionRequest(prompt="x", model_tier=ModelTier.FAST)
        assert p._resolve_model(req) == DEFAULT_TIER_MODELS[ModelTier.FAST]

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_OPENROUTER_MODEL_FAST", "qwen/qwen-2.5-7b-instruct")
        p = _provider()
        req = CompletionRequest(prompt="x", model_tier=ModelTier.FAST)
        assert p._resolve_model(req) == "qwen/qwen-2.5-7b-instruct"

    def test_resolve_rejects_closed_vendor_env(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_OPENROUTER_MODEL_FAST", "openai/gpt-4o-mini")
        p = _provider()
        req = CompletionRequest(prompt="x", model_tier=ModelTier.FAST)
        with pytest.raises(OpenWeightViolationError):
            p._resolve_model(req)


class TestPublicOnlyEgress:
    @pytest.mark.parametrize(
        "tier",
        [Sensitivity.PERSONAL, Sensitivity.CONFIDENTIAL, Sensitivity.SECRET],
    )
    def test_non_public_denied(self, tier, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        monkeypatch.delenv("ANIMUS_LOCAL_ONLY", raising=False)
        p = _provider()
        req = CompletionRequest(prompt="secret stuff", sensitivity=tier)
        with pytest.raises(EgressDeniedError, match=tier.value):
            p._check_request_egress(req)

    def test_public_allowed(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        p = _provider()
        req = CompletionRequest(prompt="public fact", sensitivity=Sensitivity.PUBLIC)
        p._check_request_egress(req)  # no raise

    def test_public_denied_when_offline(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        p = _provider()
        p._egress_endpoint = OPENROUTER_BASE_URL
        req = CompletionRequest(prompt="public fact", sensitivity=Sensitivity.PUBLIC)
        with pytest.raises(EgressDeniedError):
            p._check_request_egress(req)


class TestInitialize:
    def test_blocks_when_offline(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        p = _provider()
        with pytest.raises(EgressDeniedError, match="ANIMUS_OFFLINE"):
            p.initialize()


class TestComplete:
    def test_public_open_weight_completes(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        p = _provider()
        p._initialized = True
        p._egress_endpoint = OPENROUTER_BASE_URL
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response()
        p._client = mock_client

        req = CompletionRequest(
            prompt="hello",
            model="meta-llama/llama-3.1-8b-instruct",
            sensitivity=Sensitivity.PUBLIC,
        )
        resp = p.complete(req)
        assert resp.content == "hi"
        assert resp.provider == "openrouter"
        assert resp.input_tokens == 4
        mock_client.chat.completions.create.assert_called_once()

    def test_closed_vendor_raises_before_network(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        p = _provider()
        p._initialized = True
        p._egress_endpoint = OPENROUTER_BASE_URL
        mock_client = MagicMock()
        p._client = mock_client

        req = CompletionRequest(
            prompt="hello",
            model="openai/gpt-4o",
            sensitivity=Sensitivity.PUBLIC,
        )
        with pytest.raises(OpenWeightViolationError):
            p.complete(req)
        mock_client.chat.completions.create.assert_not_called()

    def test_non_public_raises_before_network(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        monkeypatch.delenv("ANIMUS_LOCAL_ONLY", raising=False)
        p = _provider()
        p._initialized = True
        p._egress_endpoint = OPENROUTER_BASE_URL
        mock_client = MagicMock()
        p._client = mock_client

        req = CompletionRequest(
            prompt="my private notes",
            model="meta-llama/llama-3.1-8b-instruct",
            sensitivity=Sensitivity.PERSONAL,
        )
        with pytest.raises(EgressDeniedError):
            p.complete(req)
        mock_client.chat.completions.create.assert_not_called()


class TestRegistration:
    def test_registered_when_key_present(self, monkeypatch):
        from animus_forge.tui.providers import create_provider_manager

        monkeypatch.setattr(
            "animus_forge.tui.providers.get_settings",
            lambda: SimpleNamespace(
                anthropic_api_key=None,
                openai_api_key="",
                openrouter_api_key="sk-or-test",
            ),
        )
        mgr = create_provider_manager()
        assert "openrouter" in mgr.list_providers()

    def test_not_default(self, monkeypatch):
        from animus_forge.tui.providers import create_provider_manager

        monkeypatch.setattr(
            "animus_forge.tui.providers.get_settings",
            lambda: SimpleNamespace(
                anthropic_api_key=None,
                openai_api_key="",
                openrouter_api_key="sk-or-test",
            ),
        )
        mgr = create_provider_manager()
        # openrouter registered but must not be the auto-default (local-default posture)
        assert mgr._default_provider != "openrouter"
