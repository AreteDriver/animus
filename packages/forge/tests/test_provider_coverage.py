"""Additional coverage tests for OpenAI and Anthropic providers."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from animus_forge.providers.base import (
    CompletionRequest,
    ProviderError,
    ProviderType,
    RateLimitError,
)

# =============================================================================
# Anthropic Provider
# =============================================================================


class TestAnthropicProviderInit:
    def test_init_without_package(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            # Force reimport
            import importlib

            from animus_forge.providers import anthropic_provider

            importlib.reload(anthropic_provider)
            provider = anthropic_provider.AnthropicProvider(api_key="test")
            assert provider.is_configured() is False

    def test_init_with_api_key(self):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        assert provider.name == "anthropic"
        assert provider.provider_type == ProviderType.ANTHROPIC

    def test_fallback_model(self):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test")
        assert "claude" in provider._get_fallback_model()


class TestAnthropicProviderComplete:
    @patch("animus_forge.providers.anthropic_provider.anthropic")
    def test_complete_success(self, mock_anthropic):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [SimpleNamespace(text="Hello!")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_response.id = "msg_123"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.AsyncAnthropic.return_value = MagicMock()

        provider = AnthropicProvider(api_key="test-key")
        provider.initialize()

        request = CompletionRequest(prompt="Hi", max_tokens=100)
        response = provider.complete(request)
        assert response.content == "Hello!"
        assert response.tokens_used == 15

    @patch("animus_forge.providers.anthropic_provider.anthropic")
    def test_complete_with_messages(self, mock_anthropic):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [SimpleNamespace(text="Response")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_response.id = "msg_456"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.AsyncAnthropic.return_value = MagicMock()

        provider = AnthropicProvider(api_key="test-key")
        provider.initialize()

        request = CompletionRequest(
            prompt="",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ],
        )
        response = provider.complete(request)
        # System messages should be filtered from messages list
        assert response.content == "Response"

    @patch("animus_forge.providers.anthropic_provider.anthropic")
    def test_complete_rate_limit(self, mock_anthropic):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        mock_client = MagicMock()
        # The rate limit error class
        mock_anthropic.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_client.messages.create.side_effect = mock_anthropic.RateLimitError("rate limited")
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.AsyncAnthropic.return_value = MagicMock()

        provider = AnthropicProvider(api_key="test-key")
        provider.initialize()

        request = CompletionRequest(prompt="Hi", max_tokens=100)
        with pytest.raises(RateLimitError):
            provider.complete(request)

    @patch("animus_forge.providers.anthropic_provider.anthropic")
    def test_complete_generic_error(self, mock_anthropic):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        mock_client = MagicMock()
        mock_anthropic.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_client.messages.create.side_effect = ValueError("bad request")
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.AsyncAnthropic.return_value = MagicMock()

        provider = AnthropicProvider(api_key="test-key")
        provider.initialize()

        request = CompletionRequest(prompt="Hi", max_tokens=100)
        with pytest.raises(ProviderError):
            provider.complete(request)

    @patch("animus_forge.providers.anthropic_provider.anthropic")
    def test_complete_not_initialized(self, mock_anthropic):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [SimpleNamespace(text="Hi")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = SimpleNamespace(input_tokens=5, output_tokens=3)
        mock_response.stop_reason = "end_turn"
        mock_response.id = "msg_789"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.AsyncAnthropic.return_value = MagicMock()

        provider = AnthropicProvider(api_key="test-key")
        # Don't call initialize() — complete() should auto-init
        response = provider.complete(CompletionRequest(prompt="Hi"))
        assert response.content == "Hi"

    @patch("animus_forge.providers.anthropic_provider.anthropic")
    def test_complete_no_usage(self, mock_anthropic):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [SimpleNamespace(text="Hi")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = None
        mock_response.stop_reason = "end_turn"
        mock_response.id = "msg_000"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.AsyncAnthropic.return_value = MagicMock()

        provider = AnthropicProvider(api_key="test-key")
        provider.initialize()
        response = provider.complete(CompletionRequest(prompt="Hi"))
        assert response.tokens_used == 0

    @patch("animus_forge.providers.anthropic_provider.anthropic")
    def test_complete_with_stop_sequences(self, mock_anthropic):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [SimpleNamespace(text="Result")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = SimpleNamespace(input_tokens=5, output_tokens=3)
        mock_response.stop_reason = "stop_sequence"
        mock_response.id = "msg_stop"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.AsyncAnthropic.return_value = MagicMock()

        provider = AnthropicProvider(api_key="test-key")
        provider.initialize()
        request = CompletionRequest(prompt="Hi", stop_sequences=["END"])
        response = provider.complete(request)
        assert response.content == "Result"

    def test_list_models(self):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test")
        models = provider.list_models()
        assert len(models) > 0
        assert any("claude" in m for m in models)

    def test_get_model_info(self):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test")
        info = provider.get_model_info("claude-opus-4-5-20251101")
        assert info["provider"] == "anthropic"
        assert "context_window" in info

    def test_get_model_info_unknown(self):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test")
        info = provider.get_model_info("unknown-model")
        assert info["model"] == "unknown-model"
        assert "context_window" not in info


# =============================================================================
# OpenAI Provider
# =============================================================================


class TestOpenAIProviderComplete:
    @patch("animus_forge.providers.openai_provider.OpenAI")
    @patch("animus_forge.providers.openai_provider.AsyncOpenAI")
    def test_complete_success(self, mock_async_cls, mock_cls):
        from animus_forge.providers.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        mock_choice = SimpleNamespace(
            message=SimpleNamespace(content="Hello!"),
            finish_reason="stop",
        )
        mock_response = SimpleNamespace(
            choices=[mock_choice],
            model="gpt-4o",
            usage=SimpleNamespace(total_tokens=15, prompt_tokens=10, completion_tokens=5),
            id="chatcmpl-123",
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key")
        provider.initialize()
        response = provider.complete(CompletionRequest(prompt="Hi"))
        assert response.content == "Hello!"
        assert response.tokens_used == 15

    @patch("animus_forge.providers.openai_provider.OpenAI")
    @patch("animus_forge.providers.openai_provider.AsyncOpenAI")
    def test_complete_rate_limit(self, mock_async_cls, mock_cls):
        import animus_forge.providers.openai_provider as oai_mod
        from animus_forge.providers.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        oai_mod.OpenAIRateLimitError = type("RateLimitError", (Exception,), {})
        mock_client.chat.completions.create.side_effect = oai_mod.OpenAIRateLimitError(
            "rate limited"
        )
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key")
        provider.initialize()
        with pytest.raises(RateLimitError):
            provider.complete(CompletionRequest(prompt="Hi"))

    @patch("animus_forge.providers.openai_provider.OpenAI")
    @patch("animus_forge.providers.openai_provider.AsyncOpenAI")
    def test_complete_generic_error(self, mock_async_cls, mock_cls):
        import animus_forge.providers.openai_provider as oai_mod
        from animus_forge.providers.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        oai_mod.OpenAIRateLimitError = type("RateLimitError", (Exception,), {})
        mock_client.chat.completions.create.side_effect = ValueError("bad")
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key")
        provider.initialize()
        with pytest.raises(ProviderError):
            provider.complete(CompletionRequest(prompt="Hi"))

    @patch("animus_forge.providers.openai_provider.OpenAI")
    @patch("animus_forge.providers.openai_provider.AsyncOpenAI")
    def test_complete_no_usage(self, mock_async_cls, mock_cls):
        from animus_forge.providers.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        mock_choice = SimpleNamespace(message=SimpleNamespace(content="Hi"), finish_reason="stop")
        mock_response = SimpleNamespace(choices=[mock_choice], model="gpt-4o", usage=None, id="x")
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key")
        provider.initialize()
        response = provider.complete(CompletionRequest(prompt="Hi"))
        assert response.tokens_used == 0

    @patch("animus_forge.providers.openai_provider.OpenAI")
    @patch("animus_forge.providers.openai_provider.AsyncOpenAI")
    def test_complete_with_messages(self, mock_async_cls, mock_cls):
        from animus_forge.providers.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        mock_choice = SimpleNamespace(message=SimpleNamespace(content="OK"), finish_reason="stop")
        mock_response = SimpleNamespace(
            choices=[mock_choice],
            model="gpt-4o",
            usage=SimpleNamespace(total_tokens=10, prompt_tokens=5, completion_tokens=5),
            id="x",
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key")
        provider.initialize()
        request = CompletionRequest(
            prompt="",
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="Be helpful",
        )
        response = provider.complete(request)
        assert response.content == "OK"

    @patch("animus_forge.providers.openai_provider.OpenAI")
    @patch("animus_forge.providers.openai_provider.AsyncOpenAI")
    def test_complete_with_stop_sequences(self, mock_async_cls, mock_cls):
        from animus_forge.providers.openai_provider import OpenAIProvider

        mock_client = MagicMock()
        mock_choice = SimpleNamespace(message=SimpleNamespace(content="Done"), finish_reason="stop")
        mock_response = SimpleNamespace(
            choices=[mock_choice],
            model="gpt-4o",
            usage=SimpleNamespace(total_tokens=5, prompt_tokens=3, completion_tokens=2),
            id="x",
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key")
        provider.initialize()
        request = CompletionRequest(prompt="Hi", stop_sequences=["END"], max_tokens=50)
        provider.complete(request)

    def test_list_models(self):
        from animus_forge.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test")
        models = provider.list_models()
        assert "gpt-4o" in models

    def test_get_model_info(self):
        from animus_forge.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test")
        info = provider.get_model_info("gpt-4o")
        assert info["provider"] == "openai"
        assert info["context_window"] == 128000

    def test_get_model_info_unknown(self):
        from animus_forge.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test")
        info = provider.get_model_info("unknown")
        assert info["model"] == "unknown"

    def test_fallback_model(self):
        from animus_forge.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test")
        assert provider._get_fallback_model() == "gpt-4o-mini"
