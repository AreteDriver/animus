"""Tests for tier-based LLM router."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from animus_forge.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelTier,
    ProviderError,
    ProviderType,
)
from animus_forge.providers.manager import ProviderManager
from animus_forge.providers.ollama_provider import OllamaProvider
from animus_forge.providers.router import (
    RoutingConfig,
    RoutingDecision,
    RoutingMode,
    TierRouter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_response(provider: str = "openai", model: str = "gpt-4o") -> CompletionResponse:
    return CompletionResponse(
        content="Hello!",
        model=model,
        provider=provider,
        tokens_used=20,
        input_tokens=10,
        output_tokens=10,
    )


def _make_request(**kwargs) -> CompletionRequest:
    defaults = {"prompt": "Hello"}
    defaults.update(kwargs)
    return CompletionRequest(**defaults)


@pytest.fixture
def manager():
    """ProviderManager with a cloud (openai) and local (ollama) provider."""
    pm = ProviderManager()

    # Register mock OpenAI (cloud)
    mock_openai = MagicMock()
    mock_openai.name = "openai"
    mock_openai.provider_type = ProviderType.OPENAI
    mock_openai.complete.return_value = _mock_response("openai", "gpt-4o")
    mock_openai.complete_async = AsyncMock(return_value=_mock_response("openai", "gpt-4o"))
    pm.register("openai", provider=mock_openai)

    # Register mock Ollama (local)
    mock_ollama = MagicMock(spec=OllamaProvider)
    mock_ollama.name = "ollama"
    mock_ollama.provider_type = ProviderType.OLLAMA
    mock_ollama.complete.return_value = _mock_response("ollama", "llama3.2")
    mock_ollama.complete_async = AsyncMock(return_value=_mock_response("ollama", "llama3.2"))
    mock_ollama.select_model_for_tier.return_value = "llama3.2"
    pm.register("ollama", provider=mock_ollama)

    return pm


@pytest.fixture
def request_simple():
    return _make_request()


# ---------------------------------------------------------------------------
# RoutingMode.LOCAL
# ---------------------------------------------------------------------------


class TestRoutingModeLocal:
    def test_local_mode_uses_local_provider(self, manager, request_simple):
        config = RoutingConfig(mode=RoutingMode.LOCAL)
        router = TierRouter(manager, config)
        resp = router.complete(request_simple)
        assert resp.provider == "ollama"

    def test_force_local_overrides_cloud_mode(self, manager, request_simple):
        config = RoutingConfig(mode=RoutingMode.CLOUD)
        router = TierRouter(manager, config)
        router.force_local_only(True)
        resp = router.complete(request_simple)
        assert resp.provider == "ollama"

    def test_force_local_toggle_off(self, manager, request_simple):
        config = RoutingConfig(mode=RoutingMode.CLOUD)
        router = TierRouter(manager, config)
        router.force_local_only(True)
        router.force_local_only(False)
        resp = router.complete(request_simple)
        assert resp.provider == "openai"

    def test_local_mode_no_cloud_attempted(self, manager, request_simple):
        config = RoutingConfig(mode=RoutingMode.LOCAL)
        router = TierRouter(manager, config)
        router.complete(request_simple)
        manager.get("openai").complete.assert_not_called()

    def test_local_only_no_local_providers_raises(self, request_simple):
        pm = ProviderManager()
        mock_cloud = MagicMock()
        mock_cloud.provider_type = ProviderType.OPENAI
        pm.register("cloud", provider=mock_cloud)

        config = RoutingConfig(mode=RoutingMode.LOCAL)
        router = TierRouter(pm, config)
        # Should still attempt (falls through to "try all") but cloud will fail
        mock_cloud.complete.side_effect = ProviderError("cloud only")
        with pytest.raises(ProviderError):
            router.complete(request_simple)


# ---------------------------------------------------------------------------
# RoutingMode.CLOUD
# ---------------------------------------------------------------------------


class TestRoutingModeCloud:
    def test_cloud_mode_uses_cloud_provider(self, manager, request_simple):
        config = RoutingConfig(mode=RoutingMode.CLOUD)
        router = TierRouter(manager, config)
        resp = router.complete(request_simple)
        assert resp.provider == "openai"

    def test_cloud_mode_no_local_attempted(self, manager, request_simple):
        config = RoutingConfig(mode=RoutingMode.CLOUD)
        router = TierRouter(manager, config)
        router.complete(request_simple)
        manager.get("ollama").complete.assert_not_called()


# ---------------------------------------------------------------------------
# RoutingMode.HYBRID
# ---------------------------------------------------------------------------


class TestRoutingModeHybrid:
    def test_fast_tier_routes_local(self, manager):
        req = _make_request(model_tier=ModelTier.FAST)
        config = RoutingConfig(mode=RoutingMode.HYBRID)
        router = TierRouter(manager, config)
        resp = router.complete(req)
        assert resp.provider == "ollama"

    def test_embedding_tier_routes_local(self, manager):
        req = _make_request(model_tier=ModelTier.EMBEDDING)
        config = RoutingConfig(mode=RoutingMode.HYBRID)
        router = TierRouter(manager, config)
        resp = router.complete(req)
        assert resp.provider == "ollama"

    def test_reasoning_tier_routes_cloud(self, manager):
        req = _make_request(model_tier=ModelTier.REASONING)
        config = RoutingConfig(mode=RoutingMode.HYBRID)
        router = TierRouter(manager, config)
        resp = router.complete(req)
        assert resp.provider == "openai"

    def test_standard_tier_prefers_local(self, manager):
        req = _make_request(model_tier=ModelTier.STANDARD)
        config = RoutingConfig(mode=RoutingMode.HYBRID)
        router = TierRouter(manager, config)
        resp = router.complete(req)
        assert resp.provider == "ollama"

    def test_standard_tier_falls_back_to_cloud(self):
        """STANDARD prefers local but falls back to cloud if local fails."""
        pm = ProviderManager()

        mock_cloud = MagicMock()
        mock_cloud.provider_type = ProviderType.OPENAI
        mock_cloud.complete.return_value = _mock_response("openai")
        pm.register("openai", provider=mock_cloud)

        mock_local = MagicMock(spec=OllamaProvider)
        mock_local.provider_type = ProviderType.OLLAMA
        mock_local.complete.side_effect = ProviderError("ollama down")
        mock_local.select_model_for_tier.return_value = "llama3.2"
        pm.register("ollama", provider=mock_local)

        config = RoutingConfig(mode=RoutingMode.HYBRID)
        router = TierRouter(pm, config)
        req = _make_request(model_tier=ModelTier.STANDARD)
        resp = router.complete(req)
        assert resp.provider == "openai"

    def test_no_tier_uses_default_order(self, manager, request_simple):
        config = RoutingConfig(mode=RoutingMode.HYBRID)
        router = TierRouter(manager, config)
        resp = router.complete(request_simple)
        # Default order = registration order → openai first
        assert resp.provider == "openai"

    def test_reasoning_no_cloud_falls_to_local(self):
        """If no cloud providers, REASONING tier falls back to local."""
        pm = ProviderManager()
        mock_local = MagicMock(spec=OllamaProvider)
        mock_local.provider_type = ProviderType.OLLAMA
        mock_local.complete.return_value = _mock_response("ollama")
        mock_local.select_model_for_tier.return_value = "qwen2.5:72b"
        pm.register("ollama", provider=mock_local)

        config = RoutingConfig(mode=RoutingMode.HYBRID)
        router = TierRouter(pm, config)
        req = _make_request(model_tier=ModelTier.REASONING)
        resp = router.complete(req)
        assert resp.provider == "ollama"

    def test_fast_no_local_falls_to_cloud(self):
        """If no local providers, FAST tier falls back to cloud."""
        pm = ProviderManager()
        mock_cloud = MagicMock()
        mock_cloud.provider_type = ProviderType.OPENAI
        mock_cloud.complete.return_value = _mock_response("openai")
        pm.register("openai", provider=mock_cloud)

        config = RoutingConfig(mode=RoutingMode.HYBRID)
        router = TierRouter(pm, config)
        req = _make_request(model_tier=ModelTier.FAST)
        resp = router.complete(req)
        assert resp.provider == "openai"


# ---------------------------------------------------------------------------
# Tier preference overrides
# ---------------------------------------------------------------------------


class TestTierPreferences:
    def test_preference_override_routes_to_specified_provider(self, manager):
        config = RoutingConfig(
            mode=RoutingMode.HYBRID,
            tier_preferences={"reasoning": ["ollama"]},
        )
        router = TierRouter(manager, config)
        req = _make_request(model_tier=ModelTier.REASONING)
        resp = router.complete(req)
        assert resp.provider == "ollama"

    def test_preference_ignores_unavailable_provider(self, manager):
        config = RoutingConfig(
            mode=RoutingMode.HYBRID,
            tier_preferences={"fast": ["nonexistent", "ollama"]},
        )
        router = TierRouter(manager, config)
        req = _make_request(model_tier=ModelTier.FAST)
        resp = router.complete(req)
        assert resp.provider == "ollama"


# ---------------------------------------------------------------------------
# Budget-aware routing
# ---------------------------------------------------------------------------


class TestBudgetAwareRouting:
    def test_budget_ok_routes_normally(self, manager):
        budget = MagicMock()
        budget.remaining = 50000
        config = RoutingConfig(mode=RoutingMode.HYBRID, budget_force_local_threshold=5000)
        router = TierRouter(manager, config, budget_manager=budget)

        req = _make_request(model_tier=ModelTier.REASONING)
        resp = router.complete(req)
        assert resp.provider == "openai"

    def test_budget_low_forces_local(self, manager):
        budget = MagicMock()
        budget.remaining = 3000
        config = RoutingConfig(mode=RoutingMode.HYBRID, budget_force_local_threshold=5000)
        router = TierRouter(manager, config, budget_manager=budget)

        req = _make_request(model_tier=ModelTier.REASONING)
        resp = router.complete(req)
        assert resp.provider == "ollama"

    def test_no_budget_manager_no_check(self, manager):
        config = RoutingConfig(mode=RoutingMode.HYBRID)
        router = TierRouter(manager, config, budget_manager=None)

        req = _make_request(model_tier=ModelTier.REASONING)
        resp = router.complete(req)
        assert resp.provider == "openai"

    def test_budget_at_threshold_still_forces_local(self, manager):
        budget = MagicMock()
        budget.remaining = 4999
        config = RoutingConfig(mode=RoutingMode.HYBRID, budget_force_local_threshold=5000)
        router = TierRouter(manager, config, budget_manager=budget)

        req = _make_request(model_tier=ModelTier.REASONING)
        resp = router.complete(req)
        assert resp.provider == "ollama"

    def test_budget_exactly_at_threshold_routes_normally(self, manager):
        budget = MagicMock()
        budget.remaining = 5000
        config = RoutingConfig(mode=RoutingMode.HYBRID, budget_force_local_threshold=5000)
        router = TierRouter(manager, config, budget_manager=budget)

        req = _make_request(model_tier=ModelTier.REASONING)
        resp = router.complete(req)
        assert resp.provider == "openai"


# ---------------------------------------------------------------------------
# Failover
# ---------------------------------------------------------------------------


class TestFailover:
    def test_primary_fails_fallback_succeeds(self, manager):
        manager.get("openai").complete.side_effect = ProviderError("openai down")
        config = RoutingConfig(mode=RoutingMode.CLOUD)
        router = TierRouter(manager, config)
        # Cloud mode picks openai, fails, falls back to ollama
        resp = router.complete(_make_request())
        assert resp.provider == "ollama"

    def test_no_fallback_raises(self):
        pm = ProviderManager()
        mock = MagicMock()
        mock.provider_type = ProviderType.OPENAI
        mock.complete.side_effect = ProviderError("down")
        pm.register("only", provider=mock)

        config = RoutingConfig(mode=RoutingMode.CLOUD)
        router = TierRouter(pm, config)
        with pytest.raises(ProviderError):
            router.complete(_make_request())

    def test_fallback_recorded_in_history(self, manager):
        manager.get("openai").complete.side_effect = ProviderError("openai down")
        config = RoutingConfig(mode=RoutingMode.CLOUD)
        router = TierRouter(manager, config)
        router.complete(_make_request())

        history = router.get_routing_history()
        assert len(history) == 2
        assert history[0].was_fallback is False
        assert history[1].was_fallback is True
        assert history[1].provider_name == "ollama"

    def test_local_mode_fallback_stays_local(self):
        """In LOCAL mode, fallback should not go to cloud providers."""
        pm = ProviderManager()

        mock_local1 = MagicMock(spec=OllamaProvider)
        mock_local1.provider_type = ProviderType.OLLAMA
        mock_local1.complete.side_effect = ProviderError("ollama1 down")
        mock_local1.select_model_for_tier.return_value = None
        pm.register("ollama1", provider=mock_local1)

        mock_cloud = MagicMock()
        mock_cloud.provider_type = ProviderType.OPENAI
        mock_cloud.complete.return_value = _mock_response("openai")
        pm.register("cloud", provider=mock_cloud)

        config = RoutingConfig(mode=RoutingMode.LOCAL)
        router = TierRouter(pm, config)
        # Only local providers allowed — no fallback to cloud
        with pytest.raises(ProviderError):
            router.complete(_make_request())


# ---------------------------------------------------------------------------
# Tier-to-model resolution
# ---------------------------------------------------------------------------


class TestTierToModelResolution:
    def test_ollama_tier_resolves_model(self, manager):
        req = _make_request(model_tier=ModelTier.FAST)
        config = RoutingConfig(mode=RoutingMode.LOCAL)
        router = TierRouter(manager, config)
        router.complete(req)
        manager.get("ollama").select_model_for_tier.assert_called_with(ModelTier.FAST)

    def test_explicit_model_skips_tier_resolution(self, manager):
        req = _make_request(model="custom-model", model_tier=ModelTier.FAST)
        config = RoutingConfig(mode=RoutingMode.LOCAL)
        router = TierRouter(manager, config)
        router.complete(req)
        manager.get("ollama").select_model_for_tier.assert_not_called()

    def test_cloud_provider_ignores_tier(self, manager):
        req = _make_request(model_tier=ModelTier.REASONING)
        config = RoutingConfig(mode=RoutingMode.CLOUD)
        router = TierRouter(manager, config)
        router.complete(req)
        # OpenAI is not OllamaProvider — select_model_for_tier not called on it
        manager.get("ollama").select_model_for_tier.assert_not_called()

    def test_no_tier_no_resolution(self, manager):
        req = _make_request()
        config = RoutingConfig(mode=RoutingMode.LOCAL)
        router = TierRouter(manager, config)
        router.complete(req)
        manager.get("ollama").select_model_for_tier.assert_not_called()


# ---------------------------------------------------------------------------
# Routing history
# ---------------------------------------------------------------------------


class TestRoutingHistory:
    def test_decisions_recorded(self, manager, request_simple):
        router = TierRouter(manager)
        router.complete(request_simple)
        history = router.get_routing_history()
        assert len(history) == 1
        assert isinstance(history[0], RoutingDecision)
        assert history[0].provider_name in ("openai", "ollama")

    def test_history_limit(self, manager, request_simple):
        config = RoutingConfig(history_limit=3)
        router = TierRouter(manager, config)
        for _ in range(5):
            router.complete(request_simple)
        history = router.get_routing_history(limit=10)
        assert len(history) == 3  # Capped by deque maxlen

    def test_history_limit_parameter(self, manager, request_simple):
        router = TierRouter(manager)
        for _ in range(5):
            router.complete(request_simple)
        history = router.get_routing_history(limit=2)
        assert len(history) == 2

    def test_decision_fields(self, manager):
        req = _make_request(model_tier=ModelTier.FAST)
        config = RoutingConfig(mode=RoutingMode.LOCAL)
        router = TierRouter(manager, config)
        router.complete(req)

        d = router.get_routing_history()[0]
        assert d.model_tier == ModelTier.FAST
        assert d.provider_name == "ollama"
        assert d.timestamp > 0
        assert isinstance(d.reason, str)


# ---------------------------------------------------------------------------
# Async routing
# ---------------------------------------------------------------------------


class TestAsyncRouting:
    def test_async_routes_same_as_sync(self, manager):
        req = _make_request(model_tier=ModelTier.FAST)
        config = RoutingConfig(mode=RoutingMode.HYBRID)
        router = TierRouter(manager, config)
        resp = asyncio.run(router.complete_async(req))
        assert resp.provider == "ollama"

    def test_async_fallback(self, manager):
        manager.get("openai").complete_async = AsyncMock(side_effect=ProviderError("openai down"))
        config = RoutingConfig(mode=RoutingMode.CLOUD)
        router = TierRouter(manager, config)
        resp = asyncio.run(router.complete_async(_make_request()))
        assert resp.provider == "ollama"

    def test_async_history_recorded(self, manager):
        req = _make_request(model_tier=ModelTier.EMBEDDING)
        config = RoutingConfig(mode=RoutingMode.HYBRID)
        router = TierRouter(manager, config)
        asyncio.run(router.complete_async(req))
        assert len(router.get_routing_history()) >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_providers_raises(self):
        pm = ProviderManager()
        router = TierRouter(pm)
        with pytest.raises(ProviderError):
            router.complete(_make_request())

    def test_config_defaults(self):
        config = RoutingConfig()
        assert config.mode == RoutingMode.HYBRID
        assert config.budget_force_local_threshold == 5000
        assert config.history_limit == 200

    def test_routing_decision_defaults(self):
        d = RoutingDecision(provider_name="test", model=None, model_tier=None, reason="test")
        assert d.was_fallback is False
        assert d.forced_local is False
        assert d.timestamp > 0

    def test_request_with_agent_and_workflow_ids(self, manager):
        req = _make_request(
            agent_id="builder-01",
            workflow_id="wf-123",
            model_tier=ModelTier.STANDARD,
        )
        config = RoutingConfig(mode=RoutingMode.LOCAL)
        router = TierRouter(manager, config)
        resp = router.complete(req)
        assert resp.provider == "ollama"

    def test_model_tier_enum_values(self):
        assert ModelTier.REASONING.value == "reasoning"
        assert ModelTier.STANDARD.value == "standard"
        assert ModelTier.FAST.value == "fast"
        assert ModelTier.EMBEDDING.value == "embedding"


# ---------------------------------------------------------------------------
# Fallback Chain (TODO 4)
# ---------------------------------------------------------------------------


class TestFallbackChain:
    """Test configurable fallback chain and output validation."""

    def test_fallback_triggers_when_primary_fails(self, manager, request_simple):
        """Primary provider fails → fallback to next."""
        manager.get("openai").complete.side_effect = ProviderError("down")
        config = RoutingConfig(mode=RoutingMode.CLOUD)
        router = TierRouter(manager, config)

        # Cloud-only mode but openai fails — should try ollama
        # (ollama is registered, just not cloud, but it's the only remaining)
        resp = router.complete(request_simple)
        assert resp.provider == "ollama"
        history = router.get_routing_history()
        assert any(d.was_fallback for d in history)

    def test_fallback_chain_respects_order(self, manager, request_simple):
        """Explicit fallback_chain order is respected."""
        # Register a third provider
        mock_anthropic = MagicMock()
        mock_anthropic.name = "anthropic"
        mock_anthropic.provider_type = ProviderType.ANTHROPIC
        mock_anthropic.complete.return_value = _mock_response("anthropic", "claude-3")
        manager.register("anthropic", provider=mock_anthropic)

        # openai fails, chain says try anthropic before ollama
        manager.get("openai").complete.side_effect = ProviderError("down")
        config = RoutingConfig(
            mode=RoutingMode.CLOUD,
            fallback_chain=["anthropic", "ollama"],
        )
        router = TierRouter(manager, config)
        resp = router.complete(request_simple)
        assert resp.provider == "anthropic"

    def test_fallback_chain_skips_failed_provider(self, manager, request_simple):
        """Fallback chain skips the provider that already failed."""
        config = RoutingConfig(
            mode=RoutingMode.HYBRID,
            fallback_chain=["openai", "ollama"],
        )
        manager.get("openai").complete.side_effect = ProviderError("down")
        router = TierRouter(manager, config)
        resp = router.complete(request_simple)
        # openai is in chain but was the one that failed, so ollama should be used
        assert resp.provider == "ollama"

    def test_output_validation_rejects_empty_response(self, manager, request_simple):
        """Empty response triggers fallback."""
        empty_response = CompletionResponse(
            content="",
            model="gpt-4o",
            provider="openai",
            tokens_used=0,
        )
        manager.get("openai").complete.return_value = empty_response
        config = RoutingConfig(mode=RoutingMode.CLOUD)
        router = TierRouter(manager, config)
        resp = router.complete(request_simple)
        # Should have fallen back to ollama
        assert resp.provider == "ollama"

    def test_output_validation_rejects_whitespace_only(self, manager, request_simple):
        """Whitespace-only response triggers fallback."""
        ws_response = CompletionResponse(
            content="   \n  ",
            model="gpt-4o",
            provider="openai",
            tokens_used=5,
        )
        manager.get("openai").complete.return_value = ws_response
        config = RoutingConfig(mode=RoutingMode.CLOUD)
        router = TierRouter(manager, config)
        resp = router.complete(request_simple)
        assert resp.provider == "ollama"

    def test_all_providers_fail_raises(self, manager, request_simple):
        """When all providers fail, ProviderError is raised."""
        manager.get("openai").complete.side_effect = ProviderError("down")
        manager.get("ollama").complete.side_effect = ProviderError("also down")
        config = RoutingConfig(mode=RoutingMode.HYBRID)
        router = TierRouter(manager, config)
        with pytest.raises(ProviderError):
            router.complete(request_simple)

    def test_fallback_chain_empty_uses_registration_order(self, manager, request_simple):
        """Empty fallback_chain falls back to remaining providers."""
        manager.get("openai").complete.side_effect = ProviderError("down")
        config = RoutingConfig(fallback_chain=[])
        router = TierRouter(manager, config)
        resp = router.complete(request_simple)
        assert resp.provider == "ollama"

    def test_async_fallback_with_validation(self, manager, request_simple):
        """Async path also validates and falls back."""
        empty_response = CompletionResponse(
            content="",
            model="gpt-4o",
            provider="openai",
            tokens_used=0,
        )
        manager.get("openai").complete_async = AsyncMock(return_value=empty_response)
        config = RoutingConfig(mode=RoutingMode.CLOUD)
        router = TierRouter(manager, config)
        resp = asyncio.run(router.complete_async(request_simple))
        assert resp.provider == "ollama"


class TestOllamaValidateOutput:
    """Test OllamaProvider.validate_output()."""

    def test_valid_response(self):
        resp = CompletionResponse(
            content="Hello world",
            model="llama3.2",
            provider="ollama",
            tokens_used=10,
        )
        assert OllamaProvider.validate_output(resp) is True

    def test_empty_response(self):
        resp = CompletionResponse(
            content="",
            model="llama3.2",
            provider="ollama",
            tokens_used=0,
        )
        assert OllamaProvider.validate_output(resp) is False

    def test_whitespace_response(self):
        resp = CompletionResponse(
            content="  \n\t  ",
            model="llama3.2",
            provider="ollama",
            tokens_used=2,
        )
        assert OllamaProvider.validate_output(resp) is False
