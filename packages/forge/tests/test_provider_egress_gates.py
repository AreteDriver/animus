"""Per-provider egress gates for cloud providers (roadmap C2 + C3).

The 10/10 review found that A4 wired content-aware DLP into Anthropic / OpenAI /
OpenRouter / Azure(complete) but MISSED Bedrock, Vertex, and Azure's two
streaming entrypoints. A credential mis-tagged PUBLIC, or any request under
ANIMUS_OFFLINE, would have crossed those wires. These tests exercise the REAL
``_check_request_egress`` path on each provider (not a mock of it) and the
offline construction gate.
"""

from __future__ import annotations

import pytest
from animus_types import Sensitivity

from animus_forge.network import EgressDeniedError
from animus_forge.providers.azure_openai_provider import AzureOpenAIProvider
from animus_forge.providers.base import (
    CompletionRequest,
    ProviderConfig,
    ProviderType,
)
from animus_forge.providers.bedrock_provider import BedrockProvider
from animus_forge.providers.vertex_provider import VertexProvider

_KEY = "sk-ant-abc1234567890ABCDEFGHIJ"


def _secret_request() -> CompletionRequest:
    """A request tagged PUBLIC but carrying a live credential in a tool arg."""
    return CompletionRequest(
        prompt="summarize this",
        sensitivity=Sensitivity.PUBLIC,
        messages=[
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "deploy", "input": {"token": _KEY}}],
            }
        ],
    )


class TestBedrockEgressGate:
    def _provider(self) -> BedrockProvider:
        p = BedrockProvider(
            ProviderConfig(provider_type=ProviderType.BEDROCK, metadata={"region": "us-east-1"})
        )
        # Pretend the client is constructed; the egress gate must fire BEFORE
        # the runtime client is ever touched.
        p._egress_endpoint = "https://bedrock-runtime.us-east-1.amazonaws.com"
        return p

    def test_secret_payload_blocked(self):
        with pytest.raises(EgressDeniedError, match="credential"):
            self._provider()._check_request_egress(_secret_request())

    def test_confidential_request_blocked(self):
        req = CompletionRequest(prompt="hi", sensitivity=Sensitivity.CONFIDENTIAL)
        with pytest.raises(EgressDeniedError):
            self._provider()._check_request_egress(req)

    def test_public_clean_request_allowed(self):
        req = CompletionRequest(prompt="hello world", sensitivity=Sensitivity.PUBLIC)
        # Should not raise.
        self._provider()._check_request_egress(req)

    def test_offline_init_blocked(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        # Force the SDK-present branch so the offline gate (which runs after the
        # install check, before the network client is built) is reachable.
        import animus_forge.providers.bedrock_provider as mod

        monkeypatch.setattr(mod, "boto3", object(), raising=False)
        p = BedrockProvider(
            ProviderConfig(provider_type=ProviderType.BEDROCK, metadata={"region": "us-east-1"})
        )
        with pytest.raises(EgressDeniedError, match="ANIMUS_OFFLINE"):
            p.initialize()


class TestVertexEgressGate:
    def _provider(self) -> VertexProvider:
        p = VertexProvider(
            ProviderConfig(provider_type=ProviderType.VERTEX, metadata={"project": "proj"})
        )
        p._egress_endpoint = "https://us-central1-aiplatform.googleapis.com"
        return p

    def test_secret_payload_blocked(self):
        with pytest.raises(EgressDeniedError, match="credential"):
            self._provider()._check_request_egress(_secret_request())

    def test_confidential_request_blocked(self):
        req = CompletionRequest(prompt="hi", sensitivity=Sensitivity.CONFIDENTIAL)
        with pytest.raises(EgressDeniedError):
            self._provider()._check_request_egress(req)

    def test_public_clean_request_allowed(self):
        req = CompletionRequest(prompt="hello world", sensitivity=Sensitivity.PUBLIC)
        self._provider()._check_request_egress(req)

    def test_offline_init_blocked(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        import animus_forge.providers.vertex_provider as mod

        monkeypatch.setattr(mod, "vertexai", object(), raising=False)
        p = VertexProvider(
            ProviderConfig(provider_type=ProviderType.VERTEX, metadata={"project": "proj"})
        )
        with pytest.raises(EgressDeniedError, match="ANIMUS_OFFLINE"):
            p.initialize()


class TestAzureStreamingEgressGate:
    """C3: A4 gated Azure complete()/complete_async() but left both streaming
    entrypoints ungated — a secret could stream out unchecked."""

    def _provider(self) -> AzureOpenAIProvider:
        p = AzureOpenAIProvider(
            ProviderConfig(
                provider_type=ProviderType.AZURE_OPENAI,
                base_url="https://example.openai.azure.com",
                api_key="x",
            )
        )
        # Stand in a truthy client + mark initialized so the gate is the FIRST
        # thing that can stop the request, before any network call.
        p._initialized = True
        p._client = object()
        p._async_client = object()
        p._egress_endpoint = "https://example.openai.azure.com"
        return p

    def test_complete_stream_blocks_secret(self):
        with pytest.raises(EgressDeniedError, match="credential"):
            list(self._provider().complete_stream(_secret_request()))

    async def test_complete_stream_async_blocks_secret(self):
        with pytest.raises(EgressDeniedError, match="credential"):
            agen = self._provider().complete_stream_async(_secret_request())
            async for _ in agen:
                pass
