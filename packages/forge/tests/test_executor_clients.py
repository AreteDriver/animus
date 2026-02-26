"""Tests for executor client factories and circuit breakers."""

from __future__ import annotations

from unittest.mock import patch

from animus_forge.workflow.executor_clients import (
    _get_claude_client,
    _get_openai_client,
    configure_circuit_breaker,
    get_circuit_breaker,
    reset_circuit_breakers,
)


class TestCircuitBreakerConfig:
    """Tests for circuit breaker configuration."""

    def setup_method(self):
        reset_circuit_breakers()

    def test_configure_and_get(self):
        cb = configure_circuit_breaker("test-key", failure_threshold=3)
        assert cb is not None
        assert cb.name == "test-key"
        assert get_circuit_breaker("test-key") is cb

    def test_get_missing_returns_none(self):
        assert get_circuit_breaker("nonexistent") is None

    def test_reset_clears_all(self):
        configure_circuit_breaker("a")
        configure_circuit_breaker("b")
        reset_circuit_breakers()
        assert get_circuit_breaker("a") is None
        assert get_circuit_breaker("b") is None


class TestClientFactories:
    """Tests for lazy client factories."""

    def setup_method(self):
        # Reset global state
        import animus_forge.workflow.executor_clients as mod

        mod._claude_client = None
        mod._openai_client = None

    def test_get_claude_client_import_error(self):
        import animus_forge.workflow.executor_clients as mod

        mod._claude_client = None
        with patch.dict(
            "sys.modules",
            {"animus_forge.api_clients.claude_code_client": None},
        ):
            result = _get_claude_client()
            # Should return None when import fails
            assert result is None
            # Second call should still return None (cached as False)
            result2 = _get_claude_client()
            assert result2 is None

    def test_get_openai_client_import_error(self):
        import animus_forge.workflow.executor_clients as mod

        mod._openai_client = None
        with patch.dict(
            "sys.modules",
            {"animus_forge.api_clients.openai_client": None},
        ):
            result = _get_openai_client()
            assert result is None
            result2 = _get_openai_client()
            assert result2 is None

    def test_get_claude_client_success(self):
        import animus_forge.workflow.executor_clients as mod

        mod._claude_client = None
        result = _get_claude_client()
        # In Forge venv, ClaudeCodeClient is available
        assert result is not None
        # Second call returns cached
        assert _get_claude_client() is result

    def test_get_openai_client_success(self):
        import animus_forge.workflow.executor_clients as mod

        mod._openai_client = None
        result = _get_openai_client()
        assert result is not None
        assert _get_openai_client() is result
