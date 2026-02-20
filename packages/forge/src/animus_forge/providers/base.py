"""Base classes for AI provider abstraction."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class ProviderError(Exception):
    """Base exception for provider errors."""

    pass


class ProviderNotConfiguredError(ProviderError):
    """Provider is not properly configured."""

    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderType(Enum):
    """Supported provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OLLAMA = "ollama"


class ModelTier(Enum):
    """Capability tier for model selection routing.

    Used by TierRouter to pick the right model class for a given task.
    """

    REASONING = "reasoning"  # Deep thinking: 70B+, o1, Claude Opus
    STANDARD = "standard"  # General tasks: 7B-14B, GPT-4o, Claude Sonnet
    FAST = "fast"  # Quick/cheap: 1B-3B, GPT-4o-mini, Claude Haiku
    EMBEDDING = "embedding"  # Vector embeddings: nomic-embed-text, text-embedding-3


@dataclass
class ProviderConfig:
    """Configuration for an AI provider."""

    provider_type: ProviderType
    api_key: str | None = None
    base_url: str | None = None
    default_model: str | None = None
    timeout: float = 120.0
    max_retries: int = 3
    metadata: dict = field(default_factory=dict)


@dataclass
class CompletionRequest:
    """Request for text completion."""

    prompt: str
    system_prompt: str | None = None
    model: str | None = None  # None = use provider default
    temperature: float = 0.7
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    metadata: dict = field(default_factory=dict)

    # Message history for conversation context
    messages: list[dict] | None = None

    # Tier-based routing fields (used by TierRouter, ignored by providers)
    model_tier: ModelTier | None = None
    agent_id: str | None = None
    workflow_id: str | None = None


@dataclass
class CompletionResponse:
    """Response from text completion."""

    content: str
    model: str
    provider: str
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    latency_ms: float = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "tokens_used": self.tokens_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class StreamChunk:
    """A chunk from a streaming completion response."""

    content: str
    model: str
    provider: str
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    is_final: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "tokens_used": self.tokens_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class Provider(ABC):
    """Abstract base class for AI providers.

    Defines a common interface for all AI providers.
    """

    def __init__(self, config: ProviderConfig):
        """Initialize provider with configuration.

        Args:
            config: Provider configuration
        """
        self.config = config
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Provider type."""
        pass

    @property
    def default_model(self) -> str:
        """Default model for this provider."""
        return self.config.default_model or self._get_fallback_model()

    @abstractmethod
    def _get_fallback_model(self) -> str:
        """Get fallback model when none configured."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if provider is properly configured."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider client.

        Should be called before making requests.

        Raises:
            ProviderNotConfiguredError: If configuration is invalid
        """
        pass

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion.

        Args:
            request: Completion request

        Returns:
            Completion response

        Raises:
            ProviderNotConfiguredError: If not configured
            RateLimitError: If rate limited
            ProviderError: For other errors
        """
        pass

    async def complete_async(self, request: CompletionRequest) -> CompletionResponse:
        """Async completion - default wraps sync version in executor.

        Override this method for native async implementations.

        Args:
            request: Completion request

        Returns:
            Completion response
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.complete, request)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Convenience method for simple text generation.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model to use (None = default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        request = CompletionRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = self.complete(request)
        return response.content

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Async convenience method for simple text generation.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model to use (None = default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        request = CompletionRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = await self.complete_async(request)
        return response.content

    def list_models(self) -> list[str]:
        """List available models for this provider.

        Returns:
            List of model identifiers
        """
        return []

    def get_model_info(self, model: str) -> dict:
        """Get information about a specific model.

        Args:
            model: Model identifier

        Returns:
            Model information dict
        """
        return {"model": model, "provider": self.name}

    def health_check(self) -> bool:
        """Check if provider is healthy and responding.

        Returns:
            True if healthy
        """
        try:
            if not self.is_configured():
                return False
            # Try a minimal completion
            self.generate("Hello", max_tokens=5)
            return True
        except Exception:
            return False

    def complete_stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Generate a streaming completion.

        Default implementation yields single chunk from non-streaming response.
        Override for native streaming support.

        Args:
            request: Completion request

        Yields:
            StreamChunk objects as they arrive
        """
        # Default: fall back to non-streaming
        response = self.complete(request)
        yield StreamChunk(
            content=response.content,
            model=response.model,
            provider=response.provider,
            finish_reason=response.finish_reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            is_final=True,
            metadata=response.metadata,
        )

    async def complete_stream_async(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Generate an async streaming completion.

        Default implementation yields single chunk from non-streaming response.
        Override for native streaming support.

        Args:
            request: Completion request

        Yields:
            StreamChunk objects as they arrive
        """
        # Default: fall back to non-streaming
        response = await self.complete_async(request)
        yield StreamChunk(
            content=response.content,
            model=response.model,
            provider=response.provider,
            finish_reason=response.finish_reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            is_final=True,
            metadata=response.metadata,
        )

    @property
    def supports_streaming(self) -> bool:
        """Whether this provider supports native streaming.

        Returns:
            True if native streaming is supported
        """
        return False
