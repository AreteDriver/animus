"""Azure OpenAI provider implementation."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
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
    StreamChunk,
)

try:
    from openai import AsyncAzureOpenAI, AzureOpenAI
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    AzureOpenAI = None  # Optional import: openai package not installed
    AsyncAzureOpenAI = None
    OpenAIRateLimitError = Exception


class AzureOpenAIProvider(Provider):
    """Azure OpenAI Service provider.

    Requires:
        - AZURE_OPENAI_API_KEY: API key
        - AZURE_OPENAI_ENDPOINT: Azure endpoint URL
        - AZURE_OPENAI_DEPLOYMENT: Deployment name (optional, can use model)
    """

    MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-35-turbo",
    ]

    def __init__(
        self,
        config: ProviderConfig | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        deployment: str | None = None,
        api_version: str = "2024-02-01",
    ):
        """Initialize Azure OpenAI provider.

        Args:
            config: Provider configuration
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL
            deployment: Deployment name
            api_version: API version to use
        """
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.AZURE_OPENAI,
                api_key=api_key,
                base_url=endpoint,
                metadata={
                    "deployment": deployment,
                    "api_version": api_version,
                },
            )
        super().__init__(config)
        self._client: AzureOpenAI | None = None
        self._async_client: AsyncAzureOpenAI | None = None

    @property
    def name(self) -> str:
        return "azure_openai"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.AZURE_OPENAI

    def _get_fallback_model(self) -> str:
        return self.config.metadata.get("deployment", "gpt-4o")

    def is_configured(self) -> bool:
        """Check if Azure OpenAI is properly configured."""
        if not AzureOpenAI:
            return False
        return bool(self.config.api_key and self.config.base_url)

    def initialize(self) -> None:
        """Initialize Azure OpenAI client."""
        if not AzureOpenAI:
            raise ProviderNotConfiguredError("openai package not installed")

        if not self.config.api_key:
            try:
                from animus_forge.config.settings import get_settings

                settings = get_settings()
                self.config.api_key = settings.azure_openai_api_key
                if not self.config.base_url:
                    self.config.base_url = settings.azure_openai_endpoint
                if not self.config.metadata.get("deployment"):
                    self.config.metadata["deployment"] = settings.azure_openai_deployment
            except Exception:
                pass  # Non-critical fallback: config loading optional

        if not self.config.api_key:
            raise ProviderNotConfiguredError("Azure OpenAI API key not configured")
        if not self.config.base_url:
            raise ProviderNotConfiguredError("Azure OpenAI endpoint not configured")

        api_version = self.config.metadata.get("api_version", "2024-02-01")

        self._client = AzureOpenAI(
            api_key=self.config.api_key,
            azure_endpoint=self.config.base_url,
            api_version=api_version,
            timeout=self.config.timeout,
        )
        if AsyncAzureOpenAI:
            self._async_client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self.config.base_url,
                api_version=api_version,
                timeout=self.config.timeout,
            )
        self._initialized = True

    def _get_deployment(self, model: str | None) -> str:
        """Get deployment name, falling back to model name."""
        return self.config.metadata.get("deployment") or model or self.default_model

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using Azure OpenAI API."""
        if not self._initialized:
            self.initialize()

        if not self._client:
            raise ProviderNotConfiguredError("Azure OpenAI client not initialized")

        model = request.model or self.default_model
        deployment = self._get_deployment(model)
        messages = self._build_messages(request)

        start_time = time.time()
        try:
            response = self._call_api(
                deployment=deployment,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences,
            )
        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e))
        except Exception as e:
            raise ProviderError(f"Azure OpenAI API error: {e}")

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
            metadata={"id": response.id, "deployment": deployment},
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
        deployment: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        stop: list[str] | None,
    ) -> Any:
        """Make API call with retry logic."""
        kwargs: dict[str, Any] = {
            "model": deployment,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop

        return self._client.chat.completions.create(**kwargs)

    async def complete_async(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion asynchronously."""
        if not self._initialized:
            self.initialize()

        if not self._async_client:
            raise ProviderNotConfiguredError("Azure OpenAI async client not initialized")

        model = request.model or self.default_model
        deployment = self._get_deployment(model)
        messages = self._build_messages(request)

        start_time = time.time()
        try:
            response = await self._call_api_async(
                deployment=deployment,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences,
            )
        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e))
        except Exception as e:
            raise ProviderError(f"Azure OpenAI API error: {e}")

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
            metadata={"id": response.id, "deployment": deployment},
        )

    @async_with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def _call_api_async(
        self,
        deployment: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        stop: list[str] | None,
    ) -> Any:
        """Make async API call with retry logic."""
        kwargs: dict[str, Any] = {
            "model": deployment,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop

        return await self._async_client.chat.completions.create(**kwargs)

    @property
    def supports_streaming(self) -> bool:
        return True

    def complete_stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Generate a streaming completion."""
        if not self._initialized:
            self.initialize()

        if not self._client:
            raise ProviderNotConfiguredError("Azure OpenAI client not initialized")

        model = request.model or self.default_model
        deployment = self._get_deployment(model)
        messages = self._build_messages(request)

        kwargs: dict[str, Any] = {
            "model": deployment,
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
        }
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences

        try:
            stream = self._client.chat.completions.create(**kwargs)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        model=chunk.model or deployment,
                        provider=self.name,
                        finish_reason=chunk.choices[0].finish_reason,
                        is_final=chunk.choices[0].finish_reason is not None,
                    )
        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e))
        except Exception as e:
            raise ProviderError(f"Azure OpenAI streaming error: {e}")

    async def complete_stream_async(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Generate an async streaming completion."""
        if not self._initialized:
            self.initialize()

        if not self._async_client:
            raise ProviderNotConfiguredError("Azure OpenAI async client not initialized")

        model = request.model or self.default_model
        deployment = self._get_deployment(model)
        messages = self._build_messages(request)

        kwargs: dict[str, Any] = {
            "model": deployment,
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
        }
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences

        try:
            stream = await self._async_client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        model=chunk.model or deployment,
                        provider=self.name,
                        finish_reason=chunk.choices[0].finish_reason,
                        is_final=chunk.choices[0].finish_reason is not None,
                    )
        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e))
        except Exception as e:
            raise ProviderError(f"Azure OpenAI streaming error: {e}")

    def list_models(self) -> list[str]:
        """List available Azure OpenAI models."""
        return self.MODELS.copy()

    def get_model_info(self, model: str) -> dict:
        """Get information about a model."""
        return {
            "model": model,
            "provider": self.name,
            "deployment": self._get_deployment(model),
            "description": "Azure OpenAI deployment",
        }
