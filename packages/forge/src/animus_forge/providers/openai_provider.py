"""OpenAI provider implementation."""

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
    from openai import AsyncOpenAI, OpenAI
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAI = None  # Optional import: openai package not installed
    AsyncOpenAI = None
    OpenAIRateLimitError = Exception


class OpenAIProvider(Provider):
    """OpenAI API provider."""

    MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini",
    ]

    def __init__(self, config: ProviderConfig | None = None, api_key: str | None = None):
        """Initialize OpenAI provider.

        Args:
            config: Provider configuration
            api_key: API key (alternative to config)
        """
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.OPENAI,
                api_key=api_key,
            )
        super().__init__(config)
        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI

    def _get_fallback_model(self) -> str:
        return "gpt-4o-mini"

    def is_configured(self) -> bool:
        """Check if OpenAI API key is available."""
        if not OpenAI:
            return False
        return bool(self.config.api_key)

    def initialize(self) -> None:
        """Initialize OpenAI client."""
        if not OpenAI:
            raise ProviderNotConfiguredError("openai package not installed")

        if not self.config.api_key:
            # Try to get from environment
            try:
                from animus_forge.config import get_settings

                self.config.api_key = get_settings().openai_api_key
            except Exception:
                pass  # Non-critical fallback: settings unavailable, check api_key below

        if not self.config.api_key:
            raise ProviderNotConfiguredError("OpenAI API key not configured")

        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        if AsyncOpenAI:
            self._async_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        self._initialized = True

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using OpenAI API."""
        if not self._initialized:
            self.initialize()

        if not self._client:
            raise ProviderNotConfiguredError("OpenAI client not initialized")

        model = request.model or self.default_model
        messages = self._build_messages(request)

        start_time = time.time()
        try:
            response = self._call_api(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences,
            )
        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e))
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {e}")

        latency_ms = (time.time() - start_time) * 1000

        return CompletionResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            provider=self.name,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            finish_reason=response.choices[0].finish_reason,
            latency_ms=latency_ms,
            metadata={"id": response.id},
        )

    def _build_messages(self, request: CompletionRequest) -> list[dict]:
        """Build message list for API."""
        messages = []

        if request.messages:
            messages.extend(request.messages)
        else:
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})

        return messages

    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _call_api(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        stop: list[str] | None,
    ) -> Any:
        """Make API call with retry logic."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop

        return self._client.chat.completions.create(**kwargs)

    async def complete_async(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion asynchronously using native async OpenAI client."""
        if not self._initialized:
            self.initialize()

        if not self._async_client:
            raise ProviderNotConfiguredError("OpenAI async client not initialized")

        model = request.model or self.default_model
        messages = self._build_messages(request)

        start_time = time.time()
        try:
            response = await self._call_api_async(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences,
            )
        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e))
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {e}")

        latency_ms = (time.time() - start_time) * 1000

        return CompletionResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            provider=self.name,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            finish_reason=response.choices[0].finish_reason,
            latency_ms=latency_ms,
            metadata={"id": response.id},
        )

    @async_with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def _call_api_async(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        stop: list[str] | None,
    ) -> Any:
        """Make async API call with retry logic."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop

        return await self._async_client.chat.completions.create(**kwargs)

    def list_models(self) -> list[str]:
        """List available OpenAI models."""
        return self.MODELS.copy()

    def get_model_info(self, model: str) -> dict:
        """Get information about an OpenAI model."""
        model_info = {
            "gpt-4o": {
                "context_window": 128000,
                "description": "Most capable GPT-4 model",
            },
            "gpt-4o-mini": {
                "context_window": 128000,
                "description": "Fast and affordable",
            },
            "gpt-4-turbo": {
                "context_window": 128000,
                "description": "High-capability GPT-4",
            },
            "gpt-4": {"context_window": 8192, "description": "Original GPT-4"},
            "gpt-3.5-turbo": {
                "context_window": 16385,
                "description": "Fast and inexpensive",
            },
            "o1-preview": {
                "context_window": 128000,
                "description": "Reasoning model preview",
            },
            "o1-mini": {
                "context_window": 128000,
                "description": "Fast reasoning model",
            },
        }
        info = model_info.get(model, {})
        return {
            "model": model,
            "provider": self.name,
            **info,
        }
