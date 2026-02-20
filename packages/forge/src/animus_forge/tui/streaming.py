"""Unified streaming adapter for all providers."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from animus_forge.providers.base import (
    CompletionRequest,
    Provider,
    ProviderError,
    ProviderType,
)

logger = logging.getLogger(__name__)


@dataclass
class StreamResult:
    """Accumulated metadata from a streaming response."""

    input_tokens: int = 0
    output_tokens: int = 0


async def stream_completion(
    provider: Provider,
    messages: list[dict[str, str]],
    system_prompt: str | None = None,
    model: str | None = None,
    result: StreamResult | None = None,
) -> AsyncGenerator[str, None]:
    """Unified streaming interface for any provider.

    Detects provider type and uses the best available streaming method:
    - Anthropic: native messages.stream() via _async_client
    - OpenAI: native chat.completions.create(stream=True)
    - Ollama: native httpx streaming via complete_stream_async()
    - Fallback: complete_async() yielding single chunk

    Args:
        provider: AI provider instance.
        messages: Conversation messages.
        system_prompt: Optional system prompt override.
        model: Optional model override.
        result: Optional StreamResult to populate with token usage.

    Yields:
        Text chunks as strings.
    """
    # Ensure provider is initialized (idempotent)
    try:
        provider.initialize()
    except Exception:
        pass  # Graceful degradation: already initialized, providers raise or no-op

    # Extract system prompt from messages if not provided
    if system_prompt is None:
        system_parts = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
        if system_parts:
            system_prompt = "\n\n".join(system_parts)

    # Filter out system messages for the message list
    filtered = [m for m in messages if m.get("role") != "system"]

    if result is None:
        result = StreamResult()

    # Route to best streaming method per provider type
    try:
        if provider.provider_type == ProviderType.ANTHROPIC:
            async for chunk in _stream_anthropic(provider, filtered, system_prompt, model, result):
                yield chunk
        elif provider.provider_type == ProviderType.OPENAI:
            async for chunk in _stream_openai(provider, filtered, system_prompt, model, result):
                yield chunk
        elif provider.provider_type == ProviderType.OLLAMA:
            async for chunk in _stream_ollama(provider, filtered, system_prompt, model, result):
                yield chunk
        else:
            async for chunk in _stream_fallback(provider, filtered, system_prompt, model, result):
                yield chunk
    except ProviderError:
        raise
    except Exception as e:
        logger.warning(f"Streaming failed, falling back to non-streaming: {e}")
        async for chunk in _stream_fallback(provider, filtered, system_prompt, model, result):
            yield chunk


async def _stream_anthropic(
    provider: Provider,
    messages: list[dict[str, str]],
    system_prompt: str | None,
    model: str | None,
    result: StreamResult,
) -> AsyncGenerator[str, None]:
    """Stream via Anthropic's native messages.stream() API."""
    client = getattr(provider, "_async_client", None)
    if client is None:
        raise ProviderError("Anthropic async client not available")

    async with client.messages.stream(
        model=model or provider.default_model,
        system=system_prompt or "You are a helpful assistant.",
        messages=messages,
        max_tokens=4096,
    ) as stream:
        async for text in stream.text_stream:
            yield text
        # Capture token usage from final message
        response = await stream.get_final_message()
        if response and response.usage:
            result.input_tokens = response.usage.input_tokens
            result.output_tokens = response.usage.output_tokens


async def _stream_openai(
    provider: Provider,
    messages: list[dict[str, str]],
    system_prompt: str | None,
    model: str | None,
    result: StreamResult,
) -> AsyncGenerator[str, None]:
    """Stream via OpenAI's native streaming API."""
    client = getattr(provider, "_async_client", None)
    if client is None:
        raise ProviderError("OpenAI async client not available")

    api_messages: list[dict[str, str]] = []
    if system_prompt:
        api_messages.append({"role": "system", "content": system_prompt})
    api_messages.extend(messages)

    stream = await client.chat.completions.create(
        model=model or provider.default_model,
        messages=api_messages,
        max_tokens=4096,
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
        # OpenAI sends usage in the final chunk when stream_options.include_usage=True
        if hasattr(chunk, "usage") and chunk.usage:
            result.input_tokens = chunk.usage.prompt_tokens or 0
            result.output_tokens = chunk.usage.completion_tokens or 0


async def _stream_ollama(
    provider: Provider,
    messages: list[dict[str, str]],
    system_prompt: str | None,
    model: str | None,
    result: StreamResult,
) -> AsyncGenerator[str, None]:
    """Stream via Ollama's complete_stream_async (httpx streaming)."""
    request = CompletionRequest(
        prompt=messages[-1].get("content", "") if messages else "",
        system_prompt=system_prompt,
        model=model,
        messages=messages,
    )
    async for chunk in provider.complete_stream_async(request):
        if chunk.content:
            yield chunk.content
        # Capture usage from chunk metadata if available
        if hasattr(chunk, "usage") and chunk.usage:
            result.input_tokens = getattr(chunk.usage, "input_tokens", 0)
            result.output_tokens = getattr(chunk.usage, "output_tokens", 0)


async def _stream_fallback(
    provider: Provider,
    messages: list[dict[str, str]],
    system_prompt: str | None,
    model: str | None,
    result: StreamResult,
) -> AsyncGenerator[str, None]:
    """Fallback: single async completion yielded as one chunk."""
    request = CompletionRequest(
        prompt=messages[-1].get("content", "") if messages else "",
        system_prompt=system_prompt,
        model=model,
        messages=messages,
    )
    response = await provider.complete_async(request)
    if hasattr(response, "usage") and response.usage:
        result.input_tokens = getattr(response.usage, "input_tokens", 0)
        result.output_tokens = getattr(response.usage, "output_tokens", 0)
    yield response.content
