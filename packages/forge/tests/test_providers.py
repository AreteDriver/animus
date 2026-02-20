"""Tests for multi-provider support."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from animus_forge.providers import (
    AnthropicProvider,
    CompletionRequest,
    CompletionResponse,
    OpenAIProvider,
    ProviderConfig,
    ProviderError,
    ProviderManager,
    ProviderNotConfiguredError,
    ProviderType,
    RateLimitError,
    get_manager,
    get_provider,
    list_providers,
    reset_manager,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def openai_config():
    """Create OpenAI provider config."""
    return ProviderConfig(
        provider_type=ProviderType.OPENAI,
        api_key="test-openai-key",
        default_model="gpt-4o",
    )


@pytest.fixture
def anthropic_config():
    """Create Anthropic provider config."""
    return ProviderConfig(
        provider_type=ProviderType.ANTHROPIC,
        api_key="test-anthropic-key",
        default_model="claude-3-opus-20240229",
    )


@pytest.fixture
def completion_request():
    """Create a basic completion request."""
    return CompletionRequest(
        prompt="Hello, world!",
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
        max_tokens=100,
    )


@pytest.fixture
def completion_response():
    """Create a mock completion response."""
    return CompletionResponse(
        content="Hello! How can I help you today?",
        model="gpt-4o",
        provider="openai",
        tokens_used=25,
        input_tokens=10,
        output_tokens=15,
        finish_reason="stop",
        latency_ms=150.0,
    )


@pytest.fixture
def manager():
    """Create a fresh provider manager."""
    return ProviderManager()


@pytest.fixture(autouse=True)
def reset_global_manager():
    """Reset global manager before each test."""
    reset_manager()
    yield
    reset_manager()


# =============================================================================
# Test ProviderConfig
# =============================================================================


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_create_openai_config(self):
        """Test creating OpenAI config."""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key="test-key",
        )
        assert config.provider_type == ProviderType.OPENAI
        assert config.api_key == "test-key"
        assert config.base_url is None
        assert config.timeout == 120.0

    def test_create_anthropic_config(self):
        """Test creating Anthropic config."""
        config = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            api_key="test-key",
            timeout=60.0,
        )
        assert config.provider_type == ProviderType.ANTHROPIC
        assert config.timeout == 60.0

    def test_config_with_custom_base_url(self):
        """Test config with custom base URL."""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key="test-key",
            base_url="https://custom.openai.com/v1",
        )
        assert config.base_url == "https://custom.openai.com/v1"

    def test_config_with_default_model(self):
        """Test config with default model."""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key="test-key",
            default_model="gpt-4-turbo",
        )
        assert config.default_model == "gpt-4-turbo"

    def test_config_metadata(self):
        """Test config with metadata."""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key="test-key",
            metadata={"org_id": "org-123"},
        )
        assert config.metadata == {"org_id": "org-123"}


# =============================================================================
# Test CompletionRequest
# =============================================================================


class TestCompletionRequest:
    """Tests for CompletionRequest dataclass."""

    def test_minimal_request(self):
        """Test minimal request with just prompt."""
        request = CompletionRequest(prompt="Hello")
        assert request.prompt == "Hello"
        assert request.temperature == 0.7
        assert request.system_prompt is None

    def test_full_request(self):
        """Test request with all fields."""
        request = CompletionRequest(
            prompt="Explain Python",
            system_prompt="You are a coding teacher.",
            model="gpt-4o",
            temperature=0.5,
            max_tokens=500,
            stop_sequences=["END", "STOP"],
            metadata={"user_id": "123"},
        )
        assert request.prompt == "Explain Python"
        assert request.system_prompt == "You are a coding teacher."
        assert request.model == "gpt-4o"
        assert request.temperature == 0.5
        assert request.max_tokens == 500
        assert request.stop_sequences == ["END", "STOP"]

    def test_request_with_messages(self):
        """Test request with message history."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        request = CompletionRequest(prompt="", messages=messages)
        assert request.messages == messages


# =============================================================================
# Test CompletionResponse
# =============================================================================


class TestCompletionResponse:
    """Tests for CompletionResponse dataclass."""

    def test_response_creation(self, completion_response):
        """Test response creation."""
        assert completion_response.content == "Hello! How can I help you today?"
        assert completion_response.model == "gpt-4o"
        assert completion_response.provider == "openai"
        assert completion_response.tokens_used == 25

    def test_response_to_dict(self, completion_response):
        """Test response to_dict conversion."""
        data = completion_response.to_dict()
        assert data["content"] == "Hello! How can I help you today?"
        assert data["model"] == "gpt-4o"
        assert data["provider"] == "openai"
        assert data["tokens_used"] == 25
        assert "timestamp" in data

    def test_response_with_metadata(self):
        """Test response with metadata."""
        response = CompletionResponse(
            content="Test",
            model="gpt-4o",
            provider="openai",
            metadata={"id": "msg-123", "usage_tier": "premium"},
        )
        assert response.metadata["id"] == "msg-123"
        assert response.metadata["usage_tier"] == "premium"

    def test_response_timestamp(self):
        """Test response has timestamp."""
        response = CompletionResponse(content="Test", model="gpt-4o", provider="openai")
        assert response.timestamp is not None
        assert isinstance(response.timestamp, datetime)


# =============================================================================
# Test Error Classes
# =============================================================================


class TestProviderErrors:
    """Tests for provider error classes."""

    def test_provider_error(self):
        """Test base ProviderError."""
        error = ProviderError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_provider_not_configured_error(self):
        """Test ProviderNotConfiguredError."""
        error = ProviderNotConfiguredError("API key missing")
        assert str(error) == "API key missing"
        assert isinstance(error, ProviderError)

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Rate limit exceeded", retry_after=30.0)
        assert str(error) == "Rate limit exceeded"
        assert error.retry_after == 30.0
        assert isinstance(error, ProviderError)

    def test_rate_limit_error_no_retry_after(self):
        """Test RateLimitError without retry_after."""
        error = RateLimitError("Rate limit exceeded")
        assert error.retry_after is None


# =============================================================================
# Test OpenAIProvider
# =============================================================================


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_provider_name(self, openai_config):
        """Test provider name."""
        provider = OpenAIProvider(config=openai_config)
        assert provider.name == "openai"

    def test_provider_type(self, openai_config):
        """Test provider type."""
        provider = OpenAIProvider(config=openai_config)
        assert provider.provider_type == ProviderType.OPENAI

    def test_default_model(self, openai_config):
        """Test default model from config."""
        provider = OpenAIProvider(config=openai_config)
        assert provider.default_model == "gpt-4o"

    def test_fallback_model(self):
        """Test fallback model when none configured."""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key="test-key",
        )
        provider = OpenAIProvider(config=config)
        assert provider.default_model == "gpt-4o-mini"

    def test_is_configured_with_key(self, openai_config):
        """Test is_configured returns True with key."""
        with patch.object(OpenAIProvider, "is_configured", return_value=True):
            provider = OpenAIProvider(config=openai_config)
            assert provider.is_configured() is True

    def test_is_configured_without_key(self):
        """Test is_configured returns False without key."""
        config = ProviderConfig(provider_type=ProviderType.OPENAI)
        provider = OpenAIProvider(config=config)
        assert provider.is_configured() is False

    def test_list_models(self, openai_config):
        """Test listing available models."""
        provider = OpenAIProvider(config=openai_config)
        models = provider.list_models()
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "gpt-4-turbo" in models

    def test_get_model_info(self, openai_config):
        """Test getting model info."""
        provider = OpenAIProvider(config=openai_config)
        info = provider.get_model_info("gpt-4o")
        assert info["model"] == "gpt-4o"
        assert info["provider"] == "openai"
        assert "context_window" in info

    def test_create_with_api_key_only(self):
        """Test creating provider with just API key."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.config.api_key == "test-key"
        assert provider.config.provider_type == ProviderType.OPENAI

    @patch("animus_forge.providers.openai_provider.OpenAI")
    def test_initialize(self, mock_openai_class, openai_config):
        """Test provider initialization."""
        provider = OpenAIProvider(config=openai_config)
        provider.initialize()
        mock_openai_class.assert_called_once()
        assert provider._initialized is True

    def test_initialize_without_package(self, openai_config):
        """Test initialization fails without openai package."""
        with patch("animus_forge.providers.openai_provider.OpenAI", None):
            provider = OpenAIProvider(config=openai_config)
            with pytest.raises(ProviderNotConfiguredError):
                provider.initialize()


# =============================================================================
# Test AnthropicProvider
# =============================================================================


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_provider_name(self, anthropic_config):
        """Test provider name."""
        provider = AnthropicProvider(config=anthropic_config)
        assert provider.name == "anthropic"

    def test_provider_type(self, anthropic_config):
        """Test provider type."""
        provider = AnthropicProvider(config=anthropic_config)
        assert provider.provider_type == ProviderType.ANTHROPIC

    def test_default_model(self, anthropic_config):
        """Test default model from config."""
        provider = AnthropicProvider(config=anthropic_config)
        assert provider.default_model == "claude-3-opus-20240229"

    def test_fallback_model(self):
        """Test fallback model when none configured."""
        config = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            api_key="test-key",
        )
        provider = AnthropicProvider(config=config)
        assert provider.default_model == "claude-sonnet-4-20250514"

    def test_is_configured_with_key(self):
        """Test is_configured returns True with key."""
        with patch.object(AnthropicProvider, "is_configured", return_value=True):
            config = ProviderConfig(
                provider_type=ProviderType.ANTHROPIC,
                api_key="test-key",
            )
            provider = AnthropicProvider(config=config)
            assert provider.is_configured() is True

    def test_is_configured_without_key(self):
        """Test is_configured returns False without key."""
        config = ProviderConfig(provider_type=ProviderType.ANTHROPIC)
        provider = AnthropicProvider(config=config)
        assert provider.is_configured() is False

    def test_list_models(self, anthropic_config):
        """Test listing available models."""
        provider = AnthropicProvider(config=anthropic_config)
        models = provider.list_models()
        assert "claude-opus-4-5-20251101" in models
        assert "claude-sonnet-4-20250514" in models
        assert "claude-3-5-haiku-20241022" in models

    def test_get_model_info(self, anthropic_config):
        """Test getting model info."""
        provider = AnthropicProvider(config=anthropic_config)
        info = provider.get_model_info("claude-opus-4-5-20251101")
        assert info["model"] == "claude-opus-4-5-20251101"
        assert info["provider"] == "anthropic"
        assert "context_window" in info

    def test_create_with_api_key_only(self):
        """Test creating provider with just API key."""
        provider = AnthropicProvider(api_key="test-key")
        assert provider.config.api_key == "test-key"
        assert provider.config.provider_type == ProviderType.ANTHROPIC

    @patch("animus_forge.providers.anthropic_provider.anthropic")
    def test_initialize(self, mock_anthropic_module, anthropic_config):
        """Test provider initialization."""
        mock_anthropic_module.Anthropic = MagicMock()
        provider = AnthropicProvider(config=anthropic_config)
        provider.initialize()
        mock_anthropic_module.Anthropic.assert_called_once()
        assert provider._initialized is True

    def test_initialize_without_package(self, anthropic_config):
        """Test initialization fails without anthropic package."""
        with patch("animus_forge.providers.anthropic_provider.anthropic", None):
            provider = AnthropicProvider(config=anthropic_config)
            with pytest.raises(ProviderNotConfiguredError):
                provider.initialize()


# =============================================================================
# Test ProviderManager
# =============================================================================


class TestProviderManager:
    """Tests for ProviderManager."""

    def test_empty_manager(self, manager):
        """Test empty manager state."""
        assert manager.list_providers() == []
        assert manager.get_default() is None

    def test_register_with_provider(self, manager, openai_config):
        """Test registering a pre-configured provider."""
        provider = OpenAIProvider(config=openai_config)
        result = manager.register("openai", provider=provider)
        assert result is provider
        assert manager.get("openai") is provider

    def test_register_with_provider_type(self, manager):
        """Test registering with provider type."""
        provider = manager.register(
            "openai",
            provider_type=ProviderType.OPENAI,
            api_key="test-key",
        )
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.api_key == "test-key"

    def test_register_with_config(self, manager, openai_config):
        """Test registering with config."""
        provider = manager.register("openai", config=openai_config)
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.api_key == "test-openai-key"

    def test_register_sets_default(self, manager, openai_config):
        """Test first registered provider becomes default."""
        provider = manager.register("openai", config=openai_config)
        assert manager.get_default() is provider

    def test_register_explicit_default(self, manager, openai_config, anthropic_config):
        """Test explicitly setting default on registration."""
        manager.register("openai", config=openai_config)
        provider = manager.register("anthropic", config=anthropic_config, set_default=True)
        assert manager.get_default() is provider

    def test_register_unknown_provider_type(self, manager):
        """Test registering unknown provider type raises error."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            manager.register("unknown", provider_type="invalid")

    def test_register_no_provider_info(self, manager):
        """Test registering without provider info raises error."""
        with pytest.raises(ValueError, match="Must specify"):
            manager.register("test")

    def test_unregister(self, manager, openai_config):
        """Test unregistering a provider."""
        manager.register("openai", config=openai_config)
        assert manager.get("openai") is not None
        manager.unregister("openai")
        assert manager.get("openai") is None

    def test_unregister_default(self, manager, openai_config, anthropic_config):
        """Test unregistering default provider."""
        manager.register("openai", config=openai_config)
        manager.register("anthropic", config=anthropic_config)
        manager.set_default("openai")
        manager.unregister("openai")
        # Should fall back to next in fallback order
        assert manager._default_provider == "anthropic"

    def test_unregister_nonexistent(self, manager):
        """Test unregistering nonexistent provider is safe."""
        manager.unregister("nonexistent")  # Should not raise

    def test_get_provider(self, manager, openai_config):
        """Test getting a provider by name."""
        provider = manager.register("openai", config=openai_config)
        assert manager.get("openai") is provider

    def test_get_nonexistent_provider(self, manager):
        """Test getting nonexistent provider returns None."""
        assert manager.get("nonexistent") is None

    def test_set_default(self, manager, openai_config, anthropic_config):
        """Test setting default provider."""
        manager.register("openai", config=openai_config)
        anthropic = manager.register("anthropic", config=anthropic_config)
        manager.set_default("anthropic")
        assert manager.get_default() is anthropic

    def test_set_default_unregistered(self, manager):
        """Test setting default to unregistered provider raises error."""
        with pytest.raises(ValueError, match="not registered"):
            manager.set_default("nonexistent")

    def test_set_fallback_order(self, manager, openai_config, anthropic_config):
        """Test setting fallback order."""
        manager.register("openai", config=openai_config)
        manager.register("anthropic", config=anthropic_config)
        manager.set_fallback_order(["anthropic", "openai"])
        assert manager._fallback_order == ["anthropic", "openai"]

    def test_set_fallback_order_unregistered(self, manager, openai_config):
        """Test setting fallback with unregistered provider raises error."""
        manager.register("openai", config=openai_config)
        with pytest.raises(ValueError, match="not registered"):
            manager.set_fallback_order(["openai", "nonexistent"])

    def test_list_providers(self, manager, openai_config, anthropic_config):
        """Test listing registered providers."""
        manager.register("openai", config=openai_config)
        manager.register("anthropic", config=anthropic_config)
        providers = manager.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers

    def test_list_configured(self, manager, openai_config):
        """Test listing configured providers."""
        # Register provider with key
        with patch.object(OpenAIProvider, "is_configured", return_value=True):
            manager.register("openai", config=openai_config)
            configured = manager.list_configured()
            assert "openai" in configured

    def test_complete_no_providers(self, manager, completion_request):
        """Test complete raises error with no providers."""
        with pytest.raises(ProviderNotConfiguredError):
            manager.complete(completion_request)

    def test_complete_with_provider(
        self, manager, openai_config, completion_request, completion_response
    ):
        """Test completing with a provider."""
        provider = manager.register("openai", config=openai_config)
        with patch.object(provider, "complete", return_value=completion_response):
            result = manager.complete(completion_request)
            assert result.content == "Hello! How can I help you today?"

    def test_complete_with_specific_provider(
        self, manager, openai_config, anthropic_config, completion_request
    ):
        """Test completing with specific provider."""
        manager.register("openai", config=openai_config)
        anthropic = manager.register("anthropic", config=anthropic_config)

        response = CompletionResponse(
            content="From Anthropic",
            model="claude-3-opus",
            provider="anthropic",
        )
        with patch.object(anthropic, "complete", return_value=response):
            result = manager.complete(completion_request, provider_name="anthropic")
            assert result.content == "From Anthropic"

    def test_complete_with_fallback(
        self, manager, openai_config, anthropic_config, completion_request
    ):
        """Test fallback when primary provider fails."""
        openai_provider = manager.register("openai", config=openai_config)
        anthropic_provider = manager.register("anthropic", config=anthropic_config)

        response = CompletionResponse(
            content="Fallback response",
            model="claude-3-opus",
            provider="anthropic",
        )

        # First provider fails, second succeeds
        with patch.object(openai_provider, "complete", side_effect=ProviderError("API down")):
            with patch.object(anthropic_provider, "complete", return_value=response):
                result = manager.complete(completion_request, use_fallback=True)
                assert result.content == "Fallback response"

    def test_complete_no_fallback(self, manager, openai_config, completion_request):
        """Test no fallback when disabled."""
        provider = manager.register("openai", config=openai_config)

        with patch.object(provider, "complete", side_effect=ProviderError("API down")):
            with pytest.raises(ProviderError):
                manager.complete(completion_request, use_fallback=False)

    def test_complete_all_providers_fail(
        self, manager, openai_config, anthropic_config, completion_request
    ):
        """Test error when all providers fail."""
        openai = manager.register("openai", config=openai_config)
        anthropic = manager.register("anthropic", config=anthropic_config)

        with patch.object(openai, "complete", side_effect=ProviderError("Error 1")):
            with patch.object(anthropic, "complete", side_effect=ProviderError("Error 2")):
                with pytest.raises(ProviderError, match="All providers failed"):
                    manager.complete(completion_request)

    def test_complete_rate_limit_fallback(
        self, manager, openai_config, anthropic_config, completion_request
    ):
        """Test fallback on rate limit."""
        openai = manager.register("openai", config=openai_config)
        anthropic = manager.register("anthropic", config=anthropic_config)

        response = CompletionResponse(
            content="Success",
            model="claude-3-opus",
            provider="anthropic",
        )

        with patch.object(openai, "complete", side_effect=RateLimitError("Rate limited")):
            with patch.object(anthropic, "complete", return_value=response):
                result = manager.complete(completion_request)
                assert result.content == "Success"

    def test_generate_convenience_method(self, manager, openai_config, completion_response):
        """Test generate convenience method."""
        provider = manager.register("openai", config=openai_config)

        with patch.object(provider, "complete", return_value=completion_response):
            result = manager.generate("Hello", system_prompt="Be helpful")
            assert result == "Hello! How can I help you today?"

    def test_health_check_all(self, manager, openai_config, anthropic_config):
        """Test health check for all providers."""
        openai = manager.register("openai", config=openai_config)
        anthropic = manager.register("anthropic", config=anthropic_config)

        with patch.object(openai, "health_check", return_value=True):
            with patch.object(anthropic, "health_check", return_value=False):
                results = manager.health_check()
                assert results["openai"] is True
                assert results["anthropic"] is False

    def test_health_check_specific(self, manager, openai_config):
        """Test health check for specific provider."""
        provider = manager.register("openai", config=openai_config)

        with patch.object(provider, "health_check", return_value=True):
            results = manager.health_check(provider_name="openai")
            assert results["openai"] is True

    def test_health_check_nonexistent(self, manager):
        """Test health check for nonexistent provider."""
        results = manager.health_check(provider_name="nonexistent")
        assert results["nonexistent"] is False


# =============================================================================
# Test Module Functions
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_manager(self):
        """Test getting global manager."""
        manager1 = get_manager()
        manager2 = get_manager()
        assert manager1 is manager2

    def test_reset_manager(self):
        """Test resetting global manager."""
        manager1 = get_manager()
        manager1.register("test", provider_type=ProviderType.OPENAI, api_key="key")
        reset_manager()
        manager2 = get_manager()
        assert manager1 is not manager2
        assert manager2.list_providers() == []

    def test_get_provider_default(self):
        """Test getting default provider from global manager."""
        manager = get_manager()
        provider = manager.register("openai", provider_type=ProviderType.OPENAI, api_key="key")
        assert get_provider() is provider

    def test_get_provider_by_name(self):
        """Test getting provider by name from global manager."""
        manager = get_manager()
        provider = manager.register("openai", provider_type=ProviderType.OPENAI, api_key="key")
        assert get_provider("openai") is provider

    def test_get_provider_nonexistent(self):
        """Test getting nonexistent provider returns None."""
        assert get_provider("nonexistent") is None

    def test_list_providers_global(self):
        """Test listing providers from global manager."""
        manager = get_manager()
        manager.register("openai", provider_type=ProviderType.OPENAI, api_key="key")
        manager.register("anthropic", provider_type=ProviderType.ANTHROPIC, api_key="key")
        providers = list_providers()
        assert "openai" in providers
        assert "anthropic" in providers


# =============================================================================
# Test Message Building
# =============================================================================


class TestMessageBuilding:
    """Tests for message building in providers."""

    def test_openai_build_messages_simple(self, openai_config):
        """Test OpenAI message building from prompt."""
        provider = OpenAIProvider(config=openai_config)
        request = CompletionRequest(
            prompt="Hello",
            system_prompt="Be helpful",
        )
        messages = provider._build_messages(request)
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "Be helpful"}
        assert messages[1] == {"role": "user", "content": "Hello"}

    def test_openai_build_messages_from_history(self, openai_config):
        """Test OpenAI message building from history."""
        provider = OpenAIProvider(config=openai_config)
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        request = CompletionRequest(prompt="", messages=history)
        messages = provider._build_messages(request)
        assert messages == history

    def test_anthropic_build_messages_simple(self, anthropic_config):
        """Test Anthropic message building from prompt."""
        provider = AnthropicProvider(config=anthropic_config)
        request = CompletionRequest(prompt="Hello")
        messages = provider._build_messages(request)
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}

    def test_anthropic_build_messages_filters_system(self, anthropic_config):
        """Test Anthropic filters system messages from history."""
        provider = AnthropicProvider(config=anthropic_config)
        history = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        request = CompletionRequest(prompt="", messages=history)
        messages = provider._build_messages(request)
        # Should exclude system message
        assert len(messages) == 2
        assert all(m["role"] != "system" for m in messages)


# =============================================================================
# Test Provider Completion Flow
# =============================================================================


class TestCompletionFlow:
    """Tests for the full completion flow."""

    @patch("animus_forge.providers.openai_provider.OpenAI")
    def test_openai_complete_flow(self, mock_openai_class, openai_config):
        """Test full OpenAI completion flow."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.id = "msg-123"
        mock_response.usage.total_tokens = 25
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 15

        mock_client.chat.completions.create.return_value = mock_response

        # Test
        provider = OpenAIProvider(config=openai_config)
        request = CompletionRequest(prompt="Hi", max_tokens=100)
        response = provider.complete(request)

        assert response.content == "Hello!"
        assert response.model == "gpt-4o"
        assert response.provider == "openai"
        assert response.tokens_used == 25

    @patch("animus_forge.providers.anthropic_provider.anthropic")
    def test_anthropic_complete_flow(self, mock_anthropic_module, anthropic_config):
        """Test full Anthropic completion flow."""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Hello from Claude!"
        mock_response.model = "claude-3-opus-20240229"
        mock_response.id = "msg-456"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 15

        mock_client.messages.create.return_value = mock_response

        # Test
        provider = AnthropicProvider(config=anthropic_config)
        request = CompletionRequest(prompt="Hi")
        response = provider.complete(request)

        assert response.content == "Hello from Claude!"
        assert response.model == "claude-3-opus-20240229"
        assert response.provider == "anthropic"
