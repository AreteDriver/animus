"""Provider wrapper for agents with streaming support."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from animus_forge.providers.base import Provider

logger = logging.getLogger(__name__)


class AgentProvider:
    """Wrapper around Provider with async and streaming support."""

    def __init__(self, provider: Provider):
        """Initialize with a provider.

        Args:
            provider: The underlying AI provider.
        """
        self.provider = provider
        if not self.provider._initialized:
            self.provider.initialize()

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Complete a conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            The assistant's response.
        """
        from animus_forge.providers.base import CompletionRequest

        # Extract system prompt from messages
        system_prompt = None
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                if system_prompt is None:
                    system_prompt = msg.get("content", "")
                else:
                    system_prompt += "\n\n" + msg.get("content", "")
            else:
                filtered_messages.append(msg)

        request = CompletionRequest(
            prompt=filtered_messages[-1].get("content", "") if filtered_messages else "",
            system_prompt=system_prompt or "You are a helpful assistant.",
            messages=filtered_messages,
            temperature=0.7,
            max_tokens=4096,
        )

        response = await self.provider.complete_async(request)
        return response.content

    async def stream_completion(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Stream a completion response.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Yields:
            Text chunks as they're generated.
        """
        # Check if provider has native streaming
        if hasattr(self.provider, "_async_client") and self.provider._async_client:
            try:
                async for chunk in self._stream_anthropic(messages):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"Streaming failed, falling back to non-streaming: {e}")

        # Fall back to non-streaming
        response = await self.complete(messages)
        yield response

    async def _stream_anthropic(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Stream using Anthropic's native streaming API.

        Args:
            messages: List of message dicts.

        Yields:
            Text chunks.
        """
        # Extract system prompt
        system_prompt = None
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                if system_prompt is None:
                    system_prompt = msg.get("content", "")
                else:
                    system_prompt += "\n\n" + msg.get("content", "")
            else:
                filtered_messages.append(msg)

        try:
            async with self.provider._async_client.messages.stream(
                model=self.provider.default_model,
                system=system_prompt or "You are a helpful assistant.",
                messages=filtered_messages,
                max_tokens=4096,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise


def create_agent_provider(provider_type: str = "anthropic") -> AgentProvider:
    """Create an agent provider.

    Args:
        provider_type: Type of provider ('anthropic' or 'openai').

    Returns:
        Configured AgentProvider.
    """
    if provider_type == "anthropic":
        from animus_forge.config import get_settings
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        settings = get_settings()
        provider = AnthropicProvider(api_key=settings.anthropic_api_key)
        return AgentProvider(provider)

    elif provider_type == "openai":
        from animus_forge.config import get_settings
        from animus_forge.providers.openai_provider import OpenAIProvider

        settings = get_settings()
        provider = OpenAIProvider(api_key=settings.openai_api_key)
        return AgentProvider(provider)

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
