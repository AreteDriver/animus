"""Tests for the vendored Forge egress helper + provider/client adopters
(Stage 3.C sibling adopters).
"""

from __future__ import annotations

import pytest

from animus_forge.network import EgressDeniedError, is_egress_allowed


class TestVendoredHelper:
    """Forge's vendored ``is_egress_allowed`` matches the core contract:
    loopback always allowed; ANIMUS_OFFLINE=1 blocks non-loopback."""

    @pytest.mark.parametrize(
        "destination",
        [
            "localhost",
            "127.0.0.1",
            "::1",
            "http://localhost:11434",
            "https://127.0.0.1:8000/v1",
            "0.0.0.0:8080",
            "ollama.local",
        ],
    )
    def test_loopback_always_allowed(self, destination, monkeypatch):
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        assert is_egress_allowed(destination) is True

    def test_offline_blocks_cloud(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        assert is_egress_allowed("https://api.anthropic.com") is False
        assert is_egress_allowed("https://api.openai.com") is False
        assert is_egress_allowed("https://x.openai.azure.com/v1") is False

    def test_default_online_allows_cloud(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        assert is_egress_allowed("https://api.anthropic.com") is True

    def test_strict_env_value_match(self, monkeypatch):
        """Only ``"1"`` triggers offline; ``"true"``/anything else does not."""
        monkeypatch.setenv("ANIMUS_OFFLINE", "true")
        assert is_egress_allowed("https://api.anthropic.com") is True

    def test_egress_denied_error_is_runtime_error(self):
        """``EgressDeniedError`` subclasses RuntimeError so existing
        ``except RuntimeError`` blocks catch it."""
        assert issubclass(EgressDeniedError, RuntimeError)


class TestAnthropicProviderAdopter:
    """``providers.AnthropicProvider.initialize`` refuses when offline."""

    def test_initialize_blocks_when_offline(self, monkeypatch):
        from animus_forge.providers.anthropic_provider import AnthropicProvider
        from animus_forge.providers.base import ProviderConfig, ProviderType

        cfg = ProviderConfig(provider_type=ProviderType.ANTHROPIC, api_key="dummy")
        provider = AnthropicProvider(config=cfg)
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        with pytest.raises(EgressDeniedError, match="ANIMUS_OFFLINE"):
            provider.initialize()


class TestOpenAIProviderAdopter:
    """``providers.OpenAIProvider.initialize`` refuses when offline."""

    def test_initialize_blocks_when_offline(self, monkeypatch):
        from animus_forge.providers.base import ProviderConfig, ProviderType
        from animus_forge.providers.openai_provider import OpenAIProvider

        cfg = ProviderConfig(provider_type=ProviderType.OPENAI, api_key="dummy")
        provider = OpenAIProvider(config=cfg)
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        with pytest.raises(EgressDeniedError, match="ANIMUS_OFFLINE"):
            provider.initialize()


class TestAzureOpenAIProviderAdopter:
    """``providers.AzureOpenAIProvider.initialize`` refuses when offline."""

    def test_initialize_blocks_when_offline(self, monkeypatch):
        from animus_forge.providers.azure_openai_provider import AzureOpenAIProvider
        from animus_forge.providers.base import ProviderConfig, ProviderType

        cfg = ProviderConfig(
            provider_type=ProviderType.AZURE_OPENAI,
            api_key="dummy",
            base_url="https://myresource.openai.azure.com",
        )
        provider = AzureOpenAIProvider(config=cfg)
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        with pytest.raises(EgressDeniedError, match="ANIMUS_OFFLINE"):
            provider.initialize()


class TestOpenAIClientAdopter:
    """``api_clients.OpenAIClient.__init__`` refuses when offline."""

    def test_construction_blocks_when_offline(self, monkeypatch):
        from animus_forge.api_clients.openai_client import OpenAIClient

        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        with pytest.raises(EgressDeniedError, match="ANIMUS_OFFLINE"):
            OpenAIClient()


class TestClaudeCodeClientAdopter:
    """``api_clients.ClaudeCodeClient.__init__`` refuses when offline (api mode)."""

    def test_api_mode_blocks_when_offline(self, monkeypatch):
        from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

        # Force api mode + a fake api key so the gate fires before the real
        # anthropic client would have been constructed.
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        monkeypatch.setenv("CLAUDE_MODE", "api")
        # Settings cache may have remembered prior values; force a refresh.
        try:
            from animus_forge.config import get_settings

            get_settings.cache_clear()
        except Exception:
            pass

        with pytest.raises(EgressDeniedError, match="ANIMUS_OFFLINE"):
            ClaudeCodeClient()
