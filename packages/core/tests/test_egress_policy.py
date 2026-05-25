"""Tests for the ANIMUS_OFFLINE sentinel + egress helper (Stage 3.C)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from animus.memory.types import Sensitivity
from animus.network import EgressDeniedError, is_egress_allowed


class TestLoopbackAlwaysAllowed:
    """Loopback destinations are unconditionally permitted."""

    @pytest.mark.parametrize(
        "destination",
        [
            "localhost",
            "127.0.0.1",
            "::1",
            "http://localhost:11434",
            "https://127.0.0.1:8000/v1/messages",
            "localhost:11434/api/generate",
            "0.0.0.0:8080",
        ],
    )
    def test_loopback_allowed_for_any_tier(self, destination, monkeypatch):
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        for tier in Sensitivity:
            assert is_egress_allowed(destination, tier) is True

    def test_dotlocal_allowed(self):
        """mDNS .local hosts are treated as loopback-equivalent (home LAN)."""
        assert is_egress_allowed("ollama.local:11434", Sensitivity.PUBLIC) is True


class TestAnimusOfflineSentinel:
    """ANIMUS_OFFLINE=1 blocks all non-loopback egress."""

    def test_offline_blocks_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        assert is_egress_allowed("https://api.anthropic.com", Sensitivity.PUBLIC) is False

    def test_offline_blocks_openai(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        assert is_egress_allowed("api.openai.com", Sensitivity.PUBLIC) is False

    def test_offline_does_not_block_loopback(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        assert is_egress_allowed("http://localhost:11434", Sensitivity.SECRET) is True

    def test_unset_offline_allows_public(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        assert is_egress_allowed("https://api.anthropic.com", Sensitivity.PUBLIC) is True

    def test_offline_value_other_than_1_does_not_trigger(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_OFFLINE", "true")  # not "1"
        # Strict gate — only "1" means offline
        assert is_egress_allowed("https://api.anthropic.com", Sensitivity.PUBLIC) is True


class TestTierGating:
    """Sensitive tiers cannot leave the box regardless of ANIMUS_OFFLINE."""

    def test_confidential_blocks_cloud(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        assert is_egress_allowed("https://api.anthropic.com", Sensitivity.CONFIDENTIAL) is False

    def test_secret_blocks_cloud(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        assert is_egress_allowed("https://api.openai.com", Sensitivity.SECRET) is False

    def test_personal_allowed_by_default(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        monkeypatch.delenv("ANIMUS_LOCAL_ONLY", raising=False)
        assert is_egress_allowed("https://api.anthropic.com", Sensitivity.PERSONAL) is True

    def test_personal_blocked_by_local_only_env(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        monkeypatch.setenv("ANIMUS_LOCAL_ONLY", "1")
        assert is_egress_allowed("https://api.anthropic.com", Sensitivity.PERSONAL) is False

    def test_public_always_allowed_offline_excepted(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        monkeypatch.setenv("ANIMUS_LOCAL_ONLY", "1")  # personal-only — public still flies
        assert is_egress_allowed("https://api.anthropic.com", Sensitivity.PUBLIC) is True

    def test_tier_none_treated_as_public(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        assert is_egress_allowed("https://api.anthropic.com") is True


class TestAnthropicModelHonorsOffline:
    """The cognitive layer's AnthropicModel refuses to construct under ANIMUS_OFFLINE=1."""

    def test_get_client_raises_when_offline(self, monkeypatch):
        from animus.cognitive import AnthropicModel, ModelConfig

        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        model = AnthropicModel(ModelConfig.anthropic())
        model.config.api_key = "dummy"
        with pytest.raises(EgressDeniedError, match="ANIMUS_OFFLINE=1"):
            model._get_client()

    def test_get_client_proceeds_when_online(self, monkeypatch):
        from animus.cognitive import AnthropicModel, ModelConfig

        monkeypatch.delenv("ANIMUS_OFFLINE", raising=False)
        model = AnthropicModel(ModelConfig.anthropic())
        model.config.api_key = "dummy"
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_anthropic.return_value = MagicMock()
            client = model._get_client()
            mock_anthropic.assert_called_once_with(api_key="dummy")
            assert client is mock_anthropic.return_value

    def test_generate_returns_egress_blocked_message_when_offline(self, monkeypatch):
        from animus.cognitive import AnthropicModel, ModelConfig

        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        model = AnthropicModel(ModelConfig.anthropic())
        model.config.api_key = "dummy"
        result = model.generate("hello")
        assert "Egress blocked" in result
        assert "ANIMUS_OFFLINE" in result
