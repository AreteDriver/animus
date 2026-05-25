"""Tests for LlamaCppProvider (local llama-server, OpenAI-compatible)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_forge.providers import (
    CompletionRequest,
    LlamaCppProvider,
    ProviderConfig,
    ProviderError,
    ProviderManager,
    ProviderNotConfiguredError,
    ProviderType,
    RateLimitError,
)
from animus_forge.providers.llamacpp_provider import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def llamacpp_config():
    """A fully-specified config (overrides defaults)."""
    return ProviderConfig(
        provider_type=ProviderType.LLAMACPP,
        api_key="test-key",
        base_url="http://localhost:9999/v1",
        default_model="custom-model",
    )


@pytest.fixture
def basic_request():
    return CompletionRequest(prompt="hi", max_tokens=8, temperature=0.0)


@pytest.fixture
def thinking_request():
    return CompletionRequest(
        prompt="hi",
        max_tokens=2048,
        temperature=0.0,
        metadata={"enable_thinking": True},
    )


def _fake_response(content: str = "pong", model: str = DEFAULT_MODEL):
    """Build a fake openai-python response object."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.total_tokens = 22
    usage.prompt_tokens = 20
    usage.completion_tokens = 2
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    response.id = "chatcmpl-test"
    return response


# =============================================================================
# Defaults & init paths
# =============================================================================


class TestLlamaCppInit:
    def test_default_init_sets_defaults(self):
        provider = LlamaCppProvider()
        assert provider.config.provider_type == ProviderType.LLAMACPP
        assert provider.config.base_url == DEFAULT_BASE_URL
        assert provider.config.default_model == DEFAULT_MODEL
        assert provider.config.api_key == "sk-no-key-required"

    def test_init_with_base_url_arg_overrides_default(self):
        provider = LlamaCppProvider(base_url="http://192.168.1.50:11435/v1")
        assert provider.config.base_url == "http://192.168.1.50:11435/v1"

    def test_init_with_config_missing_base_url_backfills(self):
        config = ProviderConfig(provider_type=ProviderType.LLAMACPP, api_key="custom")
        provider = LlamaCppProvider(config=config)
        assert provider.config.base_url == DEFAULT_BASE_URL
        # User-supplied api_key is preserved
        assert provider.config.api_key == "custom"

    def test_init_with_config_missing_api_key_backfills(self):
        config = ProviderConfig(
            provider_type=ProviderType.LLAMACPP,
            base_url="http://example:1234/v1",
        )
        provider = LlamaCppProvider(config=config)
        assert provider.config.api_key == "sk-no-key-required"
        # User-supplied base_url is preserved
        assert provider.config.base_url == "http://example:1234/v1"

    def test_init_with_full_config_preserves_values(self, llamacpp_config):
        provider = LlamaCppProvider(config=llamacpp_config)
        assert provider.config.api_key == "test-key"
        assert provider.config.base_url == "http://localhost:9999/v1"
        assert provider.config.default_model == "custom-model"


# =============================================================================
# Identity / introspection
# =============================================================================


class TestLlamaCppIdentity:
    def test_name(self):
        assert LlamaCppProvider().name == "llamacpp"

    def test_provider_type(self):
        assert LlamaCppProvider().provider_type == ProviderType.LLAMACPP

    def test_get_fallback_model(self):
        # Bypass default config so _get_fallback_model is exercised
        config = ProviderConfig(
            provider_type=ProviderType.LLAMACPP,
            base_url="http://x/v1",
        )
        # Manually clear default_model so the property falls back
        provider = LlamaCppProvider(config=config)
        provider.config.default_model = None
        assert provider.default_model == DEFAULT_MODEL

    def test_list_models(self):
        assert LlamaCppProvider().list_models() == [DEFAULT_MODEL]

    def test_get_model_info(self):
        info = LlamaCppProvider().get_model_info(DEFAULT_MODEL)
        assert info["model"] == DEFAULT_MODEL
        assert info["provider"] == "llamacpp"
        assert info["context_window"] == 262144
        assert "Qwen3.6" in info["description"]


# =============================================================================
# is_configured / initialize
# =============================================================================


class TestLlamaCppConfiguration:
    def test_is_configured_with_openai_and_base_url(self):
        provider = LlamaCppProvider()
        assert provider.is_configured() is True

    def test_is_configured_without_openai_package(self):
        with patch("animus_forge.providers.llamacpp_provider.OpenAI", None):
            assert LlamaCppProvider().is_configured() is False

    def test_is_configured_without_base_url(self):
        provider = LlamaCppProvider()
        provider.config.base_url = None
        assert provider.is_configured() is False

    def test_initialize_raises_without_openai_package(self):
        with patch("animus_forge.providers.llamacpp_provider.OpenAI", None):
            provider = LlamaCppProvider()
            with pytest.raises(ProviderNotConfiguredError, match="openai package"):
                provider.initialize()

    def test_initialize_raises_without_base_url(self):
        provider = LlamaCppProvider()
        provider.config.base_url = None
        with pytest.raises(ProviderNotConfiguredError, match="base_url"):
            provider.initialize()

    @patch("animus_forge.providers.llamacpp_provider.AsyncOpenAI")
    @patch("animus_forge.providers.llamacpp_provider.OpenAI")
    def test_initialize_creates_sync_and_async_clients(self, mock_openai, mock_async_openai):
        provider = LlamaCppProvider()
        provider.initialize()
        assert provider._initialized is True
        mock_openai.assert_called_once()
        mock_async_openai.assert_called_once()
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["base_url"] == DEFAULT_BASE_URL
        assert kwargs["api_key"] == "sk-no-key-required"

    @patch("animus_forge.providers.llamacpp_provider.AsyncOpenAI", None)
    @patch("animus_forge.providers.llamacpp_provider.OpenAI")
    def test_initialize_skips_async_client_when_unavailable(self, mock_openai):
        provider = LlamaCppProvider()
        provider.initialize()
        assert provider._client is not None
        assert provider._async_client is None


# =============================================================================
# _build_call_kwargs — thinking-mode default
# =============================================================================


class TestThinkingMode:
    def test_thinking_off_by_default(self, basic_request):
        kwargs = LlamaCppProvider()._build_call_kwargs(basic_request)
        assert kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_thinking_on_via_request_metadata(self, thinking_request):
        kwargs = LlamaCppProvider()._build_call_kwargs(thinking_request)
        assert kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}

    def test_thinking_metadata_other_keys_ignored(self):
        req = CompletionRequest(prompt="x", metadata={"unrelated": "value"})
        kwargs = LlamaCppProvider()._build_call_kwargs(req)
        assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


# =============================================================================
# complete (sync)
# =============================================================================


class TestComplete:
    @patch("animus_forge.providers.llamacpp_provider.OpenAI")
    def test_complete_calls_api_with_extra_body(self, mock_openai_class, basic_request):
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_response("pong")
        mock_openai_class.return_value = client

        provider = LlamaCppProvider()
        resp = provider.complete(basic_request)

        client.chat.completions.create.assert_called_once()
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
        assert call_kwargs["model"] == DEFAULT_MODEL
        assert call_kwargs["max_tokens"] == 8
        assert call_kwargs["temperature"] == 0.0
        assert resp.content == "pong"
        assert resp.provider == "llamacpp"
        assert resp.tokens_used == 22
        assert resp.input_tokens == 20
        assert resp.output_tokens == 2
        assert resp.finish_reason == "stop"
        assert resp.metadata["id"] == "chatcmpl-test"

    @patch("animus_forge.providers.llamacpp_provider.OpenAI")
    def test_complete_passes_thinking_on(self, mock_openai_class, thinking_request):
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_response("answer")
        mock_openai_class.return_value = client

        LlamaCppProvider().complete(thinking_request)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    @patch("animus_forge.providers.llamacpp_provider.OpenAI")
    def test_complete_with_stop_sequences(self, mock_openai_class):
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_response()
        mock_openai_class.return_value = client

        req = CompletionRequest(prompt="x", stop_sequences=["</end>"])
        LlamaCppProvider().complete(req)
        assert client.chat.completions.create.call_args.kwargs["stop"] == ["</end>"]

    @patch("animus_forge.providers.llamacpp_provider.OpenAIRateLimitError", ValueError)
    @patch("animus_forge.providers.llamacpp_provider.OpenAI")
    def test_complete_wraps_rate_limit(self, mock_openai_class, basic_request):
        client = MagicMock()
        client.chat.completions.create.side_effect = ValueError("429")
        mock_openai_class.return_value = client

        with pytest.raises(RateLimitError):
            LlamaCppProvider().complete(basic_request)

    @patch("animus_forge.providers.llamacpp_provider.OpenAI")
    def test_complete_wraps_generic_error(self, mock_openai_class, basic_request):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("boom")
        mock_openai_class.return_value = client

        with pytest.raises(ProviderError, match="llama.cpp API error"):
            LlamaCppProvider().complete(basic_request)

    def test_complete_raises_when_client_missing(self, basic_request):
        provider = LlamaCppProvider()
        provider._initialized = True  # skip auto-initialize
        provider._client = None
        with pytest.raises(ProviderNotConfiguredError):
            provider.complete(basic_request)

    @patch("animus_forge.providers.llamacpp_provider.OpenAI")
    def test_complete_uses_request_model_override(self, mock_openai_class, basic_request):
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_response()
        mock_openai_class.return_value = client

        basic_request.model = "other-alias"
        LlamaCppProvider().complete(basic_request)
        assert client.chat.completions.create.call_args.kwargs["model"] == "other-alias"


# =============================================================================
# complete_async
# =============================================================================


class TestCompleteAsync:
    @pytest.mark.asyncio
    @patch("animus_forge.providers.llamacpp_provider.AsyncOpenAI")
    @patch("animus_forge.providers.llamacpp_provider.OpenAI")
    async def test_complete_async_calls_api_with_extra_body(
        self, mock_openai_class, mock_async_class, basic_request
    ):
        async_client = MagicMock()
        async_client.chat.completions.create = AsyncMock(return_value=_fake_response("ack"))
        mock_async_class.return_value = async_client

        provider = LlamaCppProvider()
        resp = await provider.complete_async(basic_request)

        async_client.chat.completions.create.assert_awaited_once()
        call_kwargs = async_client.chat.completions.create.await_args.kwargs
        assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        assert resp.content == "ack"
        assert resp.provider == "llamacpp"

    @pytest.mark.asyncio
    @patch("animus_forge.providers.llamacpp_provider.OpenAIRateLimitError", ValueError)
    @patch("animus_forge.providers.llamacpp_provider.AsyncOpenAI")
    @patch("animus_forge.providers.llamacpp_provider.OpenAI")
    async def test_complete_async_wraps_rate_limit(
        self, mock_openai_class, mock_async_class, basic_request
    ):
        async_client = MagicMock()
        async_client.chat.completions.create = AsyncMock(side_effect=ValueError("429"))
        mock_async_class.return_value = async_client

        with pytest.raises(RateLimitError):
            await LlamaCppProvider().complete_async(basic_request)

    @pytest.mark.asyncio
    @patch("animus_forge.providers.llamacpp_provider.AsyncOpenAI")
    @patch("animus_forge.providers.llamacpp_provider.OpenAI")
    async def test_complete_async_wraps_generic_error(
        self, mock_openai_class, mock_async_class, basic_request
    ):
        async_client = MagicMock()
        async_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
        mock_async_class.return_value = async_client

        with pytest.raises(ProviderError, match="llama.cpp API error"):
            await LlamaCppProvider().complete_async(basic_request)

    @pytest.mark.asyncio
    @patch("animus_forge.providers.llamacpp_provider.AsyncOpenAI")
    @patch("animus_forge.providers.llamacpp_provider.OpenAI")
    async def test_complete_async_with_stop_sequences(self, mock_openai_class, mock_async_class):
        async_client = MagicMock()
        async_client.chat.completions.create = AsyncMock(return_value=_fake_response())
        mock_async_class.return_value = async_client

        req = CompletionRequest(prompt="x", stop_sequences=["</end>"])
        await LlamaCppProvider().complete_async(req)
        assert async_client.chat.completions.create.await_args.kwargs["stop"] == ["</end>"]

    @pytest.mark.asyncio
    async def test_complete_async_raises_when_client_missing(self, basic_request):
        provider = LlamaCppProvider()
        provider._initialized = True  # skip auto-initialize
        provider._async_client = None
        with pytest.raises(ProviderNotConfiguredError):
            await provider.complete_async(basic_request)


# =============================================================================
# ProviderManager registration
# =============================================================================


class TestManagerRegistration:
    def test_manager_can_register_llamacpp_by_type(self):
        manager = ProviderManager()
        manager.register("research", provider_type=ProviderType.LLAMACPP)
        provider = manager._providers["research"]
        assert isinstance(provider, LlamaCppProvider)
        assert provider.config.base_url == DEFAULT_BASE_URL

    def test_manager_can_register_llamacpp_by_config(self):
        manager = ProviderManager()
        cfg = ProviderConfig(
            provider_type=ProviderType.LLAMACPP,
            base_url="http://other:8080/v1",
        )
        manager.register("alt", config=cfg)
        provider = manager._providers["alt"]
        assert isinstance(provider, LlamaCppProvider)
        assert provider.config.base_url == "http://other:8080/v1"


# =============================================================================
# Live integration (skipif server not running)
# =============================================================================


def _server_alive() -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:11435/health", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


@pytest.mark.skipif(not _server_alive(), reason="llama-server not running on :11435")
class TestLlamaCppLive:
    def test_live_round_trip(self):
        provider = LlamaCppProvider()
        provider.initialize()
        resp = provider.complete(
            CompletionRequest(
                prompt="Reply with exactly the word: pong",
                max_tokens=8,
                temperature=0.0,
            )
        )
        assert resp.content.strip().lower() == "pong"
        assert resp.finish_reason == "stop"
        assert resp.provider == "llamacpp"
