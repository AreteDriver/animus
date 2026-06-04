"""C1-7 — regression net: EVERY cloud provider gates egress.

The review flagged that egress enforcement is per-provider convention, not
structurally enforced, so a new/edited provider could silently ship without the
gate. This parametrized net drives each provider's real ``_check_request_egress``
with (a) a CONFIDENTIAL request and (b) a credential-bearing PUBLIC request
against a cloud endpoint, and asserts both are denied. If someone adds a provider
without wiring the gate, this fails.

It also asserts every concrete Provider subclass that can reach a non-loopback
host actually defines ``_check_request_egress`` — catching a missing gate at the
class level, not just for the providers we happened to list.
"""

from __future__ import annotations

import pytest
from animus_types import Sensitivity

from animus_forge.network import EgressDeniedError
from animus_forge.providers.anthropic_provider import AnthropicProvider
from animus_forge.providers.azure_openai_provider import AzureOpenAIProvider
from animus_forge.providers.base import (
    CompletionRequest,
    ProviderConfig,
    ProviderType,
)
from animus_forge.providers.bedrock_provider import BedrockProvider
from animus_forge.providers.llamacpp_provider import LlamaCppProvider
from animus_forge.providers.openai_provider import OpenAIProvider
from animus_forge.providers.openrouter_provider import OpenRouterProvider
from animus_forge.providers.vertex_provider import VertexProvider

_KEY = "sk-ant-abc1234567890ABCDEFGHIJ"
_CLOUD = "https://provider.example.com/v1"

# (class, provider_type) for every provider that can reach a remote host.
_CLOUD_PROVIDERS = [
    (AnthropicProvider, ProviderType.ANTHROPIC),
    (OpenAIProvider, ProviderType.OPENAI),
    (AzureOpenAIProvider, ProviderType.AZURE_OPENAI),
    (BedrockProvider, ProviderType.BEDROCK),
    (VertexProvider, ProviderType.VERTEX),
    (OpenRouterProvider, ProviderType.OPENROUTER),
    (LlamaCppProvider, ProviderType.LLAMACPP),
]


def _make(cls, ptype):
    p = cls(ProviderConfig(provider_type=ptype, base_url=_CLOUD, api_key="x"))
    # Pretend initialize() ran: the gate must fire before any network client is
    # touched, so we only need the endpoint set.
    p._egress_endpoint = _CLOUD
    return p


@pytest.mark.parametrize(
    "cls,ptype", _CLOUD_PROVIDERS, ids=[c.__name__ for c, _ in _CLOUD_PROVIDERS]
)
class TestEveryProviderGatesEgress:
    def test_confidential_blocked(self, cls, ptype):
        p = _make(cls, ptype)
        with pytest.raises(EgressDeniedError):
            p._check_request_egress(
                CompletionRequest(prompt="hi", sensitivity=Sensitivity.CONFIDENTIAL)
            )

    def test_secret_payload_blocked(self, cls, ptype):
        p = _make(cls, ptype)
        with pytest.raises(EgressDeniedError, match="credential"):
            p._check_request_egress(
                CompletionRequest(prompt=f"deploy {_KEY}", sensitivity=Sensitivity.PUBLIC)
            )

    def test_clean_public_allowed(self, cls, ptype):
        p = _make(cls, ptype)
        # No raise — a clean PUBLIC request may leave for a cloud endpoint.
        p._check_request_egress(CompletionRequest(prompt="hello", sensitivity=Sensitivity.PUBLIC))


def test_all_cloud_capable_providers_define_the_gate():
    """Structural net: any concrete Provider subclass that isn't purely local
    must define its own ``_check_request_egress``. Catches a new provider that
    forgets the gate even if it's not in the list above."""
    from animus_forge.providers.base import Provider
    from animus_forge.providers.mock_provider import MockProvider

    # Ollama is loopback-only by contract (no remote egress surface).
    from animus_forge.providers.ollama_provider import OllamaProvider

    exempt = {Provider, OllamaProvider, MockProvider}
    missing = []
    for sub in Provider.__subclasses__():
        if sub in exempt:
            continue
        if "_check_request_egress" not in vars(sub):
            missing.append(sub.__name__)
    assert not missing, f"providers missing the egress gate: {missing}"
