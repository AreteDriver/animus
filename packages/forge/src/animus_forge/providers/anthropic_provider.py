"""Anthropic Claude provider implementation."""

from __future__ import annotations

import time
from typing import Any

from animus_forge.utils.retry import async_with_retry, with_retry

from .base import (
    CompletionRequest,
    CompletionResponse,
    Provider,
    ProviderConfig,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderType,
    RateLimitError,
)

try:
    import anthropic
    from anthropic import RateLimitError as AnthropicRateLimitError
except ImportError:
    anthropic = None  # Optional import: anthropic package not installed
    AnthropicRateLimitError = Exception


class AnthropicProvider(Provider):
    """Anthropic Claude API provider."""

    MODELS = [
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    def __init__(self, config: ProviderConfig | None = None, api_key: str | None = None):
        """Initialize Anthropic provider.

        Args:
            config: Provider configuration
            api_key: API key (alternative to config)
        """
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.ANTHROPIC,
                api_key=api_key,
            )
        super().__init__(config)
        self._client: Any | None = None
        self._async_client: Any | None = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC

    def _get_fallback_model(self) -> str:
        return "claude-sonnet-4-20250514"

    def is_configured(self) -> bool:
        """Check if Anthropic API key is available."""
        if not anthropic:
            return False
        return bool(self.config.api_key)

    def initialize(self) -> None:
        """Initialize Anthropic client."""
        if not anthropic:
            raise ProviderNotConfiguredError("anthropic package not installed")

        if not self.config.api_key:
            # Try to get from environment
            try:
                from animus_forge.config import get_settings

                self.config.api_key = get_settings().anthropic_api_key
            except Exception:
                pass  # Non-critical fallback: settings unavailable, check api_key below

        if not self.config.api_key:
            raise ProviderNotConfiguredError("Anthropic API key not configured")

        self._client = anthropic.Anthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        self._async_client = anthropic.AsyncAnthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        self._initialized = True

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using Anthropic API."""
        if not self._initialized:
            self.initialize()

        if not self._client:
            raise ProviderNotConfiguredError("Anthropic client not initialized")

        model = request.model or self.default_model
        messages = self._build_messages(request)
        system = request.system_prompt or "You are a helpful assistant."

        start_time = time.time()
        try:
            response = self._call_api(
                model=model,
                system=system,
                messages=messages,
                max_tokens=request.max_tokens or 4096,
                stop_sequences=request.stop_sequences,
            )
        except AnthropicRateLimitError as e:
            raise RateLimitError(str(e))
        except Exception as e:
            raise ProviderError(f"Anthropic API error: {e}")

        latency_ms = (time.time() - start_time) * 1000

        content = ""
        if response.content:
            content = response.content[0].text if response.content else ""

        return CompletionResponse(
            content=content,
            model=response.model,
            provider=self.name,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens
            if response.usage
            else 0,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
            finish_reason=response.stop_reason,
            latency_ms=latency_ms,
            metadata={"id": response.id},
        )

    def _build_messages(self, request: CompletionRequest) -> list[dict]:
        """Build message list for API."""
        if request.messages:
            # Filter out system messages (handled separately in Claude API)
            return [m for m in request.messages if m.get("role") != "system"]

        return [{"role": "user", "content": request.prompt}]

    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _call_api(
        self,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int,
        stop_sequences: list[str] | None,
    ) -> Any:
        """Make API call with retry logic."""
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences

        return self._client.messages.create(**kwargs)

    async def complete_async(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion asynchronously using native async Anthropic client."""
        if not self._initialized:
            self.initialize()

        if not self._async_client:
            raise ProviderNotConfiguredError("Anthropic async client not initialized")

        model = request.model or self.default_model
        messages = self._build_messages(request)
        system = request.system_prompt or "You are a helpful assistant."

        start_time = time.time()
        try:
            response = await self._call_api_async(
                model=model,
                system=system,
                messages=messages,
                max_tokens=request.max_tokens or 4096,
                stop_sequences=request.stop_sequences,
            )
        except AnthropicRateLimitError as e:
            raise RateLimitError(str(e))
        except Exception as e:
            raise ProviderError(f"Anthropic API error: {e}")

        latency_ms = (time.time() - start_time) * 1000

        content = ""
        if response.content:
            content = response.content[0].text if response.content else ""

        return CompletionResponse(
            content=content,
            model=response.model,
            provider=self.name,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens
            if response.usage
            else 0,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
            finish_reason=response.stop_reason,
            latency_ms=latency_ms,
            metadata={"id": response.id},
        )

    @async_with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def _call_api_async(
        self,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int,
        stop_sequences: list[str] | None,
    ) -> Any:
        """Make async API call with retry logic."""
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences

        return await self._async_client.messages.create(**kwargs)

    def list_models(self) -> list[str]:
        """List available Anthropic models."""
        return self.MODELS.copy()

    def get_model_info(self, model: str) -> dict:
        """Get information about an Anthropic model."""
        model_info = {
            "claude-opus-4-5-20251101": {
                "context_window": 200000,
                "description": "Most capable Claude model",
            },
            "claude-sonnet-4-20250514": {
                "context_window": 200000,
                "description": "Balanced performance and speed",
            },
            "claude-3-5-sonnet-20241022": {
                "context_window": 200000,
                "description": "Previous generation Sonnet",
            },
            "claude-3-5-haiku-20241022": {
                "context_window": 200000,
                "description": "Fast and lightweight",
            },
            "claude-3-opus-20240229": {
                "context_window": 200000,
                "description": "Previous generation Opus",
            },
            "claude-3-sonnet-20240229": {
                "context_window": 200000,
                "description": "Previous generation Sonnet",
            },
            "claude-3-haiku-20240307": {
                "context_window": 200000,
                "description": "Previous generation Haiku",
            },
        }
        info = model_info.get(model, {})
        return {
            "model": model,
            "provider": self.name,
            **info,
        }
