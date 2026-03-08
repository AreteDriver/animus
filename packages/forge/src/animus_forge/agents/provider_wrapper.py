"""Provider wrapper for agents with streaming and tool-use support."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from animus_forge.providers.base import CompletionResponse, Provider
    from animus_forge.tools.registry import ForgeToolRegistry

logger = logging.getLogger(__name__)

# Maximum iterations for the tool loop to prevent runaway execution
MAX_TOOL_ITERATIONS = 8


class AgentProvider:
    """Wrapper around Provider with async, streaming, and tool-use support."""

    def __init__(self, provider: Provider):
        """Initialize with a provider.

        Args:
            provider: The underlying AI provider.
        """
        self.provider = provider
        if not self.provider._initialized:
            self.provider.initialize()

    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
    ) -> str:
        """Complete a conversation (text only, no tools).

        Args:
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens in response.

        Returns:
            The assistant's response.
        """
        from animus_forge.providers.base import CompletionRequest

        system_prompt, filtered_messages = self._split_system(messages)

        request = CompletionRequest(
            prompt=filtered_messages[-1].get("content", "") if filtered_messages else "",
            system_prompt=system_prompt or "You are a helpful assistant.",
            messages=filtered_messages,
            temperature=0.7,
            max_tokens=max_tokens,
        )

        response = await self.provider.complete_async(request)
        return response.content

    async def complete_with_tools(
        self,
        messages: list[dict],
        tool_registry: ForgeToolRegistry,
        max_iterations: int = MAX_TOOL_ITERATIONS,
        max_tokens: int = 4096,
        progress_callback: Any = None,
    ) -> str:
        """Complete with iterative tool use.

        Calls the LLM with available tools. If the model returns tool_use
        blocks, executes them via the registry, feeds results back, and
        repeats until the model responds with text or max_iterations is hit.

        Args:
            messages: Conversation messages (system + user + context).
            tool_registry: Registry of available tools.
            max_iterations: Maximum tool loop iterations.
            max_tokens: Maximum tokens per LLM call.
            progress_callback: Optional callable(stage, detail) for updates.

        Returns:
            Final text response after all tool iterations.
        """
        from animus_forge.providers.base import CompletionRequest

        system_prompt, filtered_messages = self._split_system(messages)

        # Detect provider type for tool format
        provider_name = getattr(self.provider, "name", "")
        if provider_name == "ollama":
            tools = tool_registry.to_ollama_tools()
        else:
            tools = tool_registry.to_anthropic_tools()

        working_messages = list(filtered_messages)

        for iteration in range(max_iterations):
            request = CompletionRequest(
                prompt=working_messages[-1].get("content", "") if working_messages else "",
                system_prompt=system_prompt or "You are a helpful assistant.",
                messages=working_messages,
                temperature=0.7,
                max_tokens=max_tokens,
                tools=tools,
            )

            response = await self.provider.complete_async(request)

            if not response.tool_calls:
                # No tool calls — model is done, return text
                return response.content

            # Execute tool calls and build result messages
            if progress_callback:
                tool_names = [tc.name for tc in response.tool_calls]
                progress_callback(
                    "tools",
                    f"Iteration {iteration + 1}: executing {', '.join(tool_names)}",
                )

            # Add assistant message with tool use
            working_messages.append(
                self._build_assistant_tool_message(response, provider_name)
            )

            # Execute each tool and add results
            for tool_call in response.tool_calls:
                result = tool_registry.execute(tool_call.name, tool_call.arguments)
                working_messages.append(
                    self._build_tool_result_message(tool_call, result, provider_name)
                )

            logger.debug(
                "Tool iteration %d: executed %d tools",
                iteration + 1,
                len(response.tool_calls),
            )

        # Hit max iterations — do one final call without tools
        logger.warning("Hit max tool iterations (%d), forcing final response", max_iterations)
        request = CompletionRequest(
            prompt=working_messages[-1].get("content", "") if working_messages else "",
            system_prompt=system_prompt or "You are a helpful assistant.",
            messages=working_messages,
            temperature=0.7,
            max_tokens=max_tokens,
        )
        response = await self.provider.complete_async(request)
        return response.content

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Split system messages from conversation messages.

        Args:
            messages: Mixed list of messages.

        Returns:
            Tuple of (combined_system_prompt, filtered_messages).
        """
        system_prompt = None
        filtered = []
        for msg in messages:
            if msg.get("role") == "system":
                if system_prompt is None:
                    system_prompt = msg.get("content", "")
                else:
                    system_prompt += "\n\n" + msg.get("content", "")
            else:
                filtered.append(msg)
        return system_prompt, filtered

    @staticmethod
    def _build_assistant_tool_message(
        response: CompletionResponse,
        provider_name: str,
    ) -> dict:
        """Build an assistant message containing tool use blocks.

        For Anthropic: uses content blocks with type=tool_use.
        For Ollama: uses content + tool_calls in message.
        """
        if provider_name == "ollama":
            return {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in response.tool_calls
                ],
            }

        # Anthropic format
        content_blocks: list[dict] = []
        if response.content:
            content_blocks.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            content_blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })
        return {"role": "assistant", "content": content_blocks}

    @staticmethod
    def _build_tool_result_message(
        tool_call: Any,
        result: str,
        provider_name: str,
    ) -> dict:
        """Build a tool result message to feed back to the LLM."""
        if provider_name == "ollama":
            return {
                "role": "tool",
                "content": result,
            }

        # Anthropic format
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result,
                }
            ],
        }

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
        system_prompt, filtered_messages = self._split_system(messages)

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
        provider_type: Type of provider ('anthropic', 'openai', or 'ollama').

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

    elif provider_type == "ollama":
        import os

        from animus_forge.providers.base import ProviderConfig, ProviderType
        from animus_forge.providers.ollama_provider import OllamaProvider

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL", "deepseek-coder-v2")
        provider = OllamaProvider(
            config=ProviderConfig(
                provider_type=ProviderType.OLLAMA,
                base_url=host,
                default_model=model,
                timeout=600.0,
            ),
        )
        return AgentProvider(provider)

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
