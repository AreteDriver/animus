"""Tests for offline default detection and Ollama probing."""

from __future__ import annotations

from animus_kernel.config.offline_defaults import (
    _has_cloud_keys,
    _is_reachable,
    detect_default_provider,
    get_ollama_host,
    warn_if_ollama_unreachable,
)


class TestHasCloudKeys:
    def test_no_keys(self, monkeypatch):
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        assert _has_cloud_keys() is False

    def test_anthropic_key_present(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _has_cloud_keys() is True

    def test_openai_key_present(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert _has_cloud_keys() is True


class TestDetectDefaultProvider:
    def test_no_keys_returns_ollama(self, monkeypatch):
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANIMUS_FORCE_PROVIDER"):
            monkeypatch.delenv(key, raising=False)
        assert detect_default_provider() == "ollama"

    def test_cloud_keys_present_returns_none(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANIMUS_FORCE_PROVIDER", raising=False)
        assert detect_default_provider() is None

    def test_force_ollama_returns_ollama(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_FORCE_PROVIDER", "ollama")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert detect_default_provider() == "ollama"

    def test_force_other_returns_none(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_FORCE_PROVIDER", "anthropic")
        assert detect_default_provider() is None


class TestGetOllamaHost:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        assert get_ollama_host() == "http://localhost:11434"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.5:11434")
        assert get_ollama_host() == "http://192.168.1.5:11434"


class TestIsReachable:
    def test_localhost_port_80_is_not_reachable(self):
        # Port 80 should not be open in test env; thread join ensures < 100 ms
        assert _is_reachable("localhost", 80, timeout=0.05) is False

    def test_unreachable_host(self):
        # 192.0.2.1 is TEST-NET-1, guaranteed non-routable
        assert _is_reachable("192.0.2.1", 12345, timeout=0.05) is False


class TestWarnIfOllamaUnreachable:
    def test_unreachable_warns(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "animus_kernel.config.offline_defaults._is_reachable",
            lambda *a, **kw: False,
        )
        with caplog.at_level("WARNING"):
            result = warn_if_ollama_unreachable("http://localhost:11434")
        assert result is False
        assert "Ollama not found" in caplog.text

    def test_reachable_no_warning(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "animus_kernel.config.offline_defaults._is_reachable",
            lambda *a, **kw: True,
        )
        with caplog.at_level("WARNING"):
            result = warn_if_ollama_unreachable("http://localhost:11434")
        assert result is True
        assert "Ollama not found" not in caplog.text
