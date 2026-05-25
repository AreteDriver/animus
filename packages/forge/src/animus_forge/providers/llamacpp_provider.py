"""llama.cpp server provider for local research-tier inference.

Targets a llama.cpp `llama-server` instance running locally with an
OpenAI-compatible `/v1` endpoint. Reuses the OpenAI client transport;
the only differences from OpenAIProvider are:

- No API key is required (llama-server ignores auth by default).
- `base_url` is required and points at the local server.
- `default_model` is the `--alias` configured on llama-server.
- Qwen3.6 ships with thinking mode ON by default; we disable it unless
  the caller sets `request.metadata["enable_thinking"] = True`.

Default target: http://127.0.0.1:11435/v1 (the qwen3.6-research alias).
"""

from __future__ import annotations

import time
from typing import Any

from animus_forge.utils.retry import async_with_retry, with_retry

from .base import (
    CompletionRequest,
    CompletionResponse,
    ProviderConfig,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderType,
    RateLimitError,
)
from .openai_provider import OpenAIProvider

try:
    from openai import AsyncOpenAI, OpenAI
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:  # pragma: no cover
    OpenAI = None
    AsyncOpenAI = None
    OpenAIRateLimitError = Exception


DEFAULT_BASE_URL = "http://127.0.0.1:11435/v1"
DEFAULT_MODEL = "qwen3.6-research"


class LlamaCppProvider(OpenAIProvider):
    """Provider for a local llama.cpp `llama-server` instance."""

    MODELS = [DEFAULT_MODEL]

    def __init__(self, config: ProviderConfig | None = None, base_url: str | None = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.LLAMACPP,
                base_url=base_url or DEFAULT_BASE_URL,
                api_key="sk-no-key-required",
                default_model=DEFAULT_MODEL,
            )
        else:
            if not config.base_url:
                config.base_url = DEFAULT_BASE_URL
            if not config.api_key:
                config.api_key = "sk-no-key-required"
        super().__init__(config=config)

    @property
    def name(self) -> str:
        return "llamacpp"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LLAMACPP

    def _get_fallback_model(self) -> str:
        return DEFAULT_MODEL

    def is_configured(self) -> bool:
        if not OpenAI:
            return False
        return bool(self.config.base_url)

    def initialize(self) -> None:
        if not OpenAI:
            raise ProviderNotConfiguredError("openai package not installed")
        if not self.config.base_url:
            raise ProviderNotConfiguredError("llama.cpp base_url not configured")

        self._client = OpenAI(
            api_key=self.config.api_key or "sk-no-key-required",
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        if AsyncOpenAI:
            self._async_client = AsyncOpenAI(
                api_key=self.config.api_key or "sk-no-key-required",
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        self._initialized = True

    def list_models(self) -> list[str]:
        return self.MODELS.copy()

    def get_model_info(self, model: str) -> dict:
        return {
            "model": model,
            "provider": self.name,
            "context_window": 262144,
            "description": "Local llama.cpp server (Qwen3.6-35B-A3B hybrid MoE)",
        }

    def _build_call_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        """Build the kwargs dict for openai-python passthrough.

        Qwen3.6 ships with thinking mode ON by default — a short max_tokens
        request burns the entire budget on `<think>` and returns empty content.
        Disable it unless the caller opts in via `request.metadata`.
        """
        enable_thinking = bool(request.metadata.get("enable_thinking", False))
        return {
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
        }

    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _call_api_with_extras(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        stop: list[str] | None,
        extras: dict[str, Any],
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **extras,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop
        return self._client.chat.completions.create(**kwargs)

    @async_with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def _call_api_async_with_extras(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        stop: list[str] | None,
        extras: dict[str, Any],
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **extras,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop
        return await self._async_client.chat.completions.create(**kwargs)

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion, injecting llama.cpp chat-template kwargs."""
        if not self._initialized:
            self.initialize()
        if not self._client:
            raise ProviderNotConfiguredError("llama.cpp client not initialized")

        model = request.model or self.default_model
        messages = self._build_messages(request)
        extras = self._build_call_kwargs(request)

        start_time = time.time()
        try:
            response = self._call_api_with_extras(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences,
                extras=extras,
            )
        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e))
        except Exception as e:
            raise ProviderError(f"llama.cpp API error: {e}")

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

    async def complete_async(self, request: CompletionRequest) -> CompletionResponse:
        """Async completion with llama.cpp chat-template kwargs."""
        if not self._initialized:
            self.initialize()
        if not self._async_client:
            raise ProviderNotConfiguredError("llama.cpp async client not initialized")

        model = request.model or self.default_model
        messages = self._build_messages(request)
        extras = self._build_call_kwargs(request)

        start_time = time.time()
        try:
            response = await self._call_api_async_with_extras(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences,
                extras=extras,
            )
        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e))
        except Exception as e:
            raise ProviderError(f"llama.cpp API error: {e}")

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
