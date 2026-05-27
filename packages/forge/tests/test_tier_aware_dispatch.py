"""End-to-end tests for Stage 3.D — tier-aware LLM dispatch.

Exercises every link in the chain:

  RAG → CompletionRequest.sensitivity → TierRouter → Provider.complete()
       → egress helper → EgressDeniedError

Each layer has its own focused test class. The composition test seals
the headline contract: a CONFIDENTIAL prompt never reaches a cloud
provider.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from animus_types import Sensitivity

from animus_forge.network import EgressDeniedError, is_egress_allowed
from animus_forge.providers.base import (
    CompletionRequest,
    ProviderConfig,
    ProviderType,
)

# ---------------------------------------------------------------------------
# 3.D.1 — CompletionRequest.sensitivity field
# ---------------------------------------------------------------------------


class TestCompletionRequestSensitivity:
    def test_default_is_public(self):
        req = CompletionRequest(prompt="hi")
        assert req.sensitivity is Sensitivity.PUBLIC

    def test_explicit_sensitivity_preserved(self):
        req = CompletionRequest(prompt="hi", sensitivity=Sensitivity.CONFIDENTIAL)
        assert req.sensitivity is Sensitivity.CONFIDENTIAL

    def test_sensitivity_is_animus_types_enum(self):
        """The enum must be the shared one — identity-preserving import."""
        from animus_types import Sensitivity as SharedSensitivity

        req = CompletionRequest(prompt="hi", sensitivity=SharedSensitivity.SECRET)
        assert req.sensitivity is SharedSensitivity.SECRET


# ---------------------------------------------------------------------------
# 3.D.2 — Egress helper accepts sensitivity
# ---------------------------------------------------------------------------


class TestEgressHelperSensitivity:
    def test_public_allowed_to_cloud(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        assert (
            is_egress_allowed("https://api.anthropic.com", sensitivity=Sensitivity.PUBLIC) is True
        )

    def test_confidential_blocked_from_cloud(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        assert (
            is_egress_allowed("https://api.anthropic.com", sensitivity=Sensitivity.CONFIDENTIAL)
            is False
        )

    def test_secret_blocked_from_cloud(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        assert is_egress_allowed("https://api.openai.com", sensitivity=Sensitivity.SECRET) is False

    def test_personal_allowed_by_default(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        monkeypatch.delenv("ANIMUS_LOCAL_ONLY", raising=False)
        assert (
            is_egress_allowed("https://api.anthropic.com", sensitivity=Sensitivity.PERSONAL) is True
        )

    def test_personal_blocked_with_local_only_env(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        monkeypatch.setenv("ANIMUS_LOCAL_ONLY", "1")
        assert (
            is_egress_allowed("https://api.anthropic.com", sensitivity=Sensitivity.PERSONAL)
            is False
        )

    def test_loopback_allowed_for_any_sensitivity(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        for s in Sensitivity:
            assert is_egress_allowed("http://localhost:11434", sensitivity=s) is True

    def test_none_sensitivity_preserves_legacy_behavior(self, monkeypatch):
        """Pre-Stage-3.D callers that don't pass sensitivity still work."""
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        assert is_egress_allowed("https://api.anthropic.com") is True


# ---------------------------------------------------------------------------
# 3.D.3 — Provider per-request gates
# ---------------------------------------------------------------------------


class TestProviderRequestGates:
    """All cloud providers refuse to call out for sensitive requests."""

    def test_anthropic_provider_blocks_confidential(self, monkeypatch):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        cfg = ProviderConfig(provider_type=ProviderType.ANTHROPIC, api_key="dummy")
        provider = AnthropicProvider(config=cfg)
        # Bypass initialize() side effects by setting state manually
        provider._initialized = True
        provider._client = MagicMock()
        provider._async_client = MagicMock()
        provider._egress_endpoint = "https://api.anthropic.com"

        req = CompletionRequest(prompt="x", sensitivity=Sensitivity.CONFIDENTIAL)
        with pytest.raises(EgressDeniedError, match="confidential"):
            provider.complete(req)
        # Critical: the cloud client was never called.
        provider._client.messages.create.assert_not_called()

    def test_anthropic_provider_allows_public(self, monkeypatch):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        cfg = ProviderConfig(provider_type=ProviderType.ANTHROPIC, api_key="dummy")
        provider = AnthropicProvider(config=cfg)
        provider._initialized = True
        provider._client = MagicMock()
        provider._egress_endpoint = "https://api.anthropic.com"

        # Mock the API call to return a minimal response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="ok")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_response.model = "claude-x"
        with patch.object(provider, "_call_api", return_value=mock_response):
            req = CompletionRequest(prompt="x", sensitivity=Sensitivity.PUBLIC)
            response = provider.complete(req)
        assert "ok" in response.content

    def test_openai_provider_blocks_secret(self, monkeypatch):
        from animus_forge.providers.openai_provider import OpenAIProvider

        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        cfg = ProviderConfig(provider_type=ProviderType.OPENAI, api_key="dummy")
        provider = OpenAIProvider(config=cfg)
        provider._initialized = True
        provider._client = MagicMock()
        provider._egress_endpoint = "https://api.openai.com"

        req = CompletionRequest(prompt="x", sensitivity=Sensitivity.SECRET)
        with pytest.raises(EgressDeniedError, match="secret"):
            provider.complete(req)
        provider._client.chat.completions.create.assert_not_called()

    def test_azure_provider_blocks_confidential(self, monkeypatch):
        from animus_forge.providers.azure_openai_provider import AzureOpenAIProvider

        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        cfg = ProviderConfig(
            provider_type=ProviderType.AZURE_OPENAI,
            api_key="dummy",
            base_url="https://x.openai.azure.com",
        )
        provider = AzureOpenAIProvider(config=cfg)
        provider._initialized = True
        provider._client = MagicMock()
        provider._egress_endpoint = "https://x.openai.azure.com"

        req = CompletionRequest(prompt="x", sensitivity=Sensitivity.CONFIDENTIAL)
        with pytest.raises(EgressDeniedError, match="confidential"):
            provider.complete(req)


# ---------------------------------------------------------------------------
# 3.D.4 — TierRouter sensitivity-driven routing
# ---------------------------------------------------------------------------


class TestTierRouterSensitivity:
    def _make_router(self, local_providers: list[str], cloud_providers: list[str]):
        from animus_forge.providers.manager import ProviderManager
        from animus_forge.providers.router import RoutingConfig, TierRouter

        pm = MagicMock(spec=ProviderManager)

        def get(name: str):
            p = MagicMock()
            p.provider_type = (
                ProviderType.OLLAMA if name in local_providers else ProviderType.ANTHROPIC
            )
            return p

        pm.list_providers.return_value = local_providers + cloud_providers
        pm.get.side_effect = get
        return TierRouter(pm, RoutingConfig())

    def test_confidential_forces_local(self):
        router = self._make_router(["ollama"], ["anthropic"])
        req = CompletionRequest(prompt="x", sensitivity=Sensitivity.CONFIDENTIAL)
        decision = router._select_provider(req)
        assert decision.provider_name == "ollama"
        assert "sensitivity=confidential" in decision.reason

    def test_secret_forces_local(self):
        router = self._make_router(["ollama"], ["anthropic"])
        req = CompletionRequest(prompt="x", sensitivity=Sensitivity.SECRET)
        decision = router._select_provider(req)
        assert decision.provider_name == "ollama"

    def test_confidential_raises_when_no_local_provider(self):
        from animus_forge.providers.base import ProviderError

        router = self._make_router([], ["anthropic"])
        req = CompletionRequest(prompt="x", sensitivity=Sensitivity.CONFIDENTIAL)
        with pytest.raises(ProviderError, match="No local provider"):
            router._select_provider(req)

    def test_personal_with_local_only_env_forces_local(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_LOCAL_ONLY", "1")
        router = self._make_router(["ollama"], ["anthropic"])
        req = CompletionRequest(prompt="x", sensitivity=Sensitivity.PERSONAL)
        decision = router._select_provider(req)
        assert decision.provider_name == "ollama"

    def test_public_routes_normally(self):
        """PUBLIC tier doesn't trigger force-local; existing routing logic applies."""
        router = self._make_router(["ollama"], ["anthropic"])
        req = CompletionRequest(prompt="x", sensitivity=Sensitivity.PUBLIC)
        decision = router._select_provider(req)
        # Default HYBRID mode with no tier → "no tier specified" reason
        assert "sensitivity=" not in decision.reason


# ---------------------------------------------------------------------------
# 3.D.5 — RAG sibling method returning AssembledContext
# ---------------------------------------------------------------------------


class TestAssembledContextRAG:
    def _make_workflow(self, sensitivities: list[str]):
        from animus_forge.intelligence.cross_workflow_memory import CrossWorkflowMemory

        agent_memory = MagicMock()
        mems = []
        for i, sens in enumerate(sensitivities):
            m = MagicMock()
            m.id = f"mem-{i}"
            m.content = f"learning {i}"
            m.importance = 0.5
            m.created_at = datetime.now(UTC)
            m.metadata = {"tags": [], "sensitivity": sens}
            mems.append(m)
        agent_memory.recall.return_value = mems
        return CrossWorkflowMemory(agent_memory)

    def test_returns_assembled_context_dataclass(self):
        from animus_forge.intelligence.cross_workflow_memory import AssembledContext

        wf = self._make_workflow(["public"])
        result = wf.build_context_for_agent_assembled(
            agent_role="planner",
            task_description="anything",
            max_tokens=1024,
        )
        assert isinstance(result, AssembledContext)
        assert result.max_sensitivity is Sensitivity.PUBLIC
        assert result.text  # non-empty

    def test_max_sensitivity_is_max_of_chunks(self):
        wf = self._make_workflow(["public", "confidential", "personal"])
        result = wf.build_context_for_agent_assembled(
            agent_role="planner",
            task_description="anything",
            max_tokens=2048,
        )
        assert result.max_sensitivity is Sensitivity.CONFIDENTIAL

    def test_empty_recall_returns_public(self):
        wf = self._make_workflow([])
        result = wf.build_context_for_agent_assembled(agent_role="x", task_description="y")
        assert result.text == ""
        assert result.max_sensitivity is Sensitivity.PUBLIC

    def test_missing_sensitivity_metadata_defaults_to_public(self):
        from animus_forge.intelligence.cross_workflow_memory import CrossWorkflowMemory

        agent_memory = MagicMock()
        m = MagicMock()
        m.id = "mem"
        m.content = "no sens metadata"
        m.importance = 0.5
        m.created_at = datetime.now(UTC)
        m.metadata = {"tags": []}  # no "sensitivity" key
        agent_memory.recall.return_value = [m]

        wf = CrossWorkflowMemory(agent_memory)
        result = wf.build_context_for_agent_assembled(
            agent_role="x", task_description="y", max_tokens=512
        )
        assert result.max_sensitivity is Sensitivity.PUBLIC

    def test_legacy_method_emits_deprecation_warning(self):
        wf = self._make_workflow(["public"])
        with pytest.warns(DeprecationWarning, match="build_context_for_agent_assembled"):
            wf.build_context_for_agent(
                agent_role="planner", task_description="anything", max_tokens=512
            )


# ---------------------------------------------------------------------------
# Composition — the headline contract
# ---------------------------------------------------------------------------


class TestEndToEndComposition:
    def test_confidential_rag_routes_to_local_never_cloud(self, monkeypatch):
        """The headline test: a RAG-assembled context with CONFIDENTIAL chunks
        flows through Sensitivity → CompletionRequest → TierRouter and lands
        on the local Ollama provider, never the cloud Anthropic provider."""
        from animus_forge.intelligence.cross_workflow_memory import CrossWorkflowMemory
        from animus_forge.providers.manager import ProviderManager
        from animus_forge.providers.router import RoutingConfig, TierRouter

        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)

        # Stage 1: assemble a context whose max tier is CONFIDENTIAL
        agent_memory = MagicMock()
        confidential_mem = MagicMock()
        confidential_mem.id = "client-1"
        confidential_mem.content = "client kickoff details"
        confidential_mem.importance = 0.9
        confidential_mem.created_at = datetime.now(UTC)
        confidential_mem.metadata = {"tags": ["tiaid"], "sensitivity": "confidential"}
        agent_memory.recall.return_value = [confidential_mem]
        wf = CrossWorkflowMemory(agent_memory)
        assembled = wf.build_context_for_agent_assembled(
            agent_role="x", task_description="y", max_tokens=1024
        )
        assert assembled.max_sensitivity is Sensitivity.CONFIDENTIAL

        # Stage 2: construct a request with that tier
        req = CompletionRequest(prompt=assembled.text, sensitivity=assembled.max_sensitivity)

        # Stage 3: route the request — must pick local
        pm = MagicMock(spec=ProviderManager)
        pm.list_providers.return_value = ["ollama", "anthropic"]

        def get(name):
            p = MagicMock()
            p.provider_type = ProviderType.OLLAMA if name == "ollama" else ProviderType.ANTHROPIC
            return p

        pm.get.side_effect = get
        router = TierRouter(pm, RoutingConfig())
        decision = router._select_provider(req)
        assert decision.provider_name == "ollama", (
            "CONFIDENTIAL request must route to local; got cloud"
        )
