"""Provider wrapper for agents with streaming and tool-use support."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from animus_forge.providers.base import CompletionResponse, Provider
    from animus_forge.tools.registry import ForgeToolRegistry

logger = logging.getLogger(__name__)

# Maximum iterations for the tool loop to prevent runaway execution
MAX_TOOL_ITERATIONS = 8

# Models known to support native Ollama tool calling
OLLAMA_TOOL_MODELS = {
    "qwen2.5:14b",
    "qwen2.5:7b",
    "qwen2.5:32b",
    "qwen2.5:72b",
    "qwen2.5-coder",
    "qwen2.5-coder:latest",
    "qwen2.5-coder:7b",
    "llama3.1:8b",
    "llama3.1",
    "llama3.1:70b",
    "llama3.3",
    "llama3.3:latest",
    "mistral",
    "mistral:latest",
    "mistral-nemo",
}

# Default model for tool-equipped agents when primary model lacks tool support
DEFAULT_TOOL_MODEL = "qwen2.5:14b"


class AgentProvider:
    """Wrapper around Provider with async, streaming, and tool-use support."""

    def __init__(self, provider: Provider, tool_model: str | None = None):
        """Initialize with a provider.

        Args:
            provider: The underlying AI provider.
            tool_model: Override model for tool-equipped calls. If the primary
                model doesn't support native tools, this model is used instead.
                Defaults to qwen2.5:14b for Ollama providers.
        """
        self.provider = provider
        self._tool_model = tool_model
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

    def _resolve_tool_model(self) -> str | None:
        """Determine which model to use for tool-equipped calls.

        Returns the tool_model override if the primary model doesn't support
        native tool calling. Returns None to use default model.
        """
        if self._tool_model:
            return self._tool_model

        provider_name = getattr(self.provider, "name", "")
        if provider_name != "ollama":
            return None  # Anthropic/OpenAI always support tools

        default_model = getattr(self.provider, "default_model", "")
        if default_model in OLLAMA_TOOL_MODELS:
            return None  # Current model supports tools

        # Primary model lacks tool support — no auto-fallback,
        # text-based fallback will be used instead
        return None

    @property
    def supports_native_tools(self) -> bool:
        """Whether the current provider/model supports native tool calling."""
        provider_name = getattr(self.provider, "name", "")
        if provider_name != "ollama":
            return True  # Anthropic/OpenAI always support tools
        if self._tool_model:
            return True  # Explicit tool model set
        default_model = getattr(self.provider, "default_model", "")
        return default_model in OLLAMA_TOOL_MODELS

    async def complete_with_tools(
        self,
        messages: list[dict],
        tool_registry: ForgeToolRegistry,
        max_iterations: int = MAX_TOOL_ITERATIONS,
        max_tokens: int = 4096,
        progress_callback: Any = None,
        agent_id: str = "",
    ) -> str:
        """Complete with iterative tool use.

        Uses native tool calling when the model supports it. For models
        without native tool support, falls back to text-based tool parsing
        where the model outputs JSON tool calls in its response text.

        Args:
            messages: Conversation messages (system + user + context).
            tool_registry: Registry of available tools.
            max_iterations: Maximum tool loop iterations.
            max_tokens: Maximum tokens per LLM call.
            progress_callback: Optional callable(stage, detail) for updates.
            agent_id: Agent identifier for audit logging and budget tracking.

        Returns:
            Final text response after all tool iterations.
        """
        from animus_forge.providers.base import CompletionRequest

        system_prompt, filtered_messages = self._split_system(messages)

        # Detect provider type for tool format
        provider_name = getattr(self.provider, "name", "")
        tool_model = self._resolve_tool_model()

        if provider_name == "ollama":
            tools = tool_registry.to_ollama_tools()
        else:
            tools = tool_registry.to_anthropic_tools()

        # Use text-based fallback when no native tool support available
        use_text_fallback = not self.supports_native_tools

        if use_text_fallback:
            return await self._complete_with_text_tools(
                system_prompt,
                filtered_messages,
                tool_registry,
                max_iterations,
                max_tokens,
                progress_callback,
                agent_id=agent_id,
            )

        working_messages = list(filtered_messages)

        for iteration in range(max_iterations):
            request = CompletionRequest(
                prompt=working_messages[-1].get("content", "") if working_messages else "",
                system_prompt=system_prompt or "You are a helpful assistant.",
                messages=working_messages,
                model=tool_model,
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
            working_messages.append(self._build_assistant_tool_message(response, provider_name))

            # Execute each tool and add results
            for tool_call in response.tool_calls:
                result = tool_registry.execute(
                    tool_call.name,
                    tool_call.arguments,
                    agent_id=agent_id,
                )
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
            model=tool_model,
            temperature=0.7,
            max_tokens=max_tokens,
        )
        response = await self.provider.complete_async(request)
        return response.content

    async def _complete_with_text_tools(
        self,
        system_prompt: str | None,
        filtered_messages: list[dict],
        tool_registry: ForgeToolRegistry,
        max_iterations: int,
        max_tokens: int,
        progress_callback: Any,
        agent_id: str = "",
    ) -> str:
        """Text-based tool loop for models without native tool support.

        Injects tool descriptions into the system prompt and parses
        JSON tool calls from the model's text output.
        """
        from animus_forge.providers.base import CompletionRequest

        tool_instructions = self._build_text_tool_prompt(tool_registry)
        enhanced_system = (
            (system_prompt or "You are a helpful assistant.") + "\n\n" + tool_instructions
        )

        working_messages = list(filtered_messages)

        for iteration in range(max_iterations):
            request = CompletionRequest(
                prompt=working_messages[-1].get("content", "") if working_messages else "",
                system_prompt=enhanced_system,
                messages=working_messages,
                temperature=0.7,
                max_tokens=max_tokens,
            )

            response = await self.provider.complete_async(request)
            text = response.content

            # Try to parse tool calls from text
            parsed_calls = self._parse_text_tool_calls(text, tool_registry)
            if not parsed_calls:
                return text

            if progress_callback:
                tool_names = [name for name, _ in parsed_calls]
                progress_callback(
                    "tools",
                    f"Iteration {iteration + 1}: executing {', '.join(tool_names)}",
                )

            # Execute parsed tool calls
            results_text = ""
            for tool_name, tool_args in parsed_calls:
                result = tool_registry.execute(tool_name, tool_args, agent_id=agent_id)
                results_text += f"\n\n[Tool Result: {tool_name}]\n{result}"

            # Add assistant response and tool results back
            working_messages.append({"role": "assistant", "content": text})
            working_messages.append(
                {
                    "role": "user",
                    "content": f"Tool results:{results_text}\n\nContinue with your task based on these results.",
                }
            )

            logger.debug(
                "Text tool iteration %d: executed %d tools",
                iteration + 1,
                len(parsed_calls),
            )

        # Final call without tool instructions
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
    def _build_text_tool_prompt(tool_registry: ForgeToolRegistry) -> str:
        """Build tool instructions for text-based tool calling."""
        lines = [
            "## Available Tools",
            "",
            "You can call tools by including a JSON block in your response:",
            "```tool_call",
            '{"tool": "tool_name", "arguments": {"param": "value"}}',
            "```",
            "",
            "Available tools:",
        ]
        for tool in tool_registry.tools:
            params = tool.parameters.get("properties", {})
            param_list = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in params.items())
            lines.append(f"- **{tool.name}**({param_list}): {tool.description}")

        lines.append("")
        lines.append("Call tools when you need to read files, search code, etc.")
        lines.append(
            "When you're done, respond with your final answer without any tool_call blocks."
        )
        return "\n".join(lines)

    @staticmethod
    def _parse_text_tool_calls(
        text: str,
        tool_registry: ForgeToolRegistry,
    ) -> list[tuple[str, dict]]:
        """Parse tool calls from model text output.

        Looks for ```tool_call blocks or JSON with "tool" key.

        Returns:
            List of (tool_name, arguments) tuples.
        """
        calls: list[tuple[str, dict]] = []

        # Pattern 1: ```tool_call ... ```
        for match in re.finditer(r"```tool_call\s*(.*?)\s*```", text, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                name = data.get("tool", "")
                args = data.get("arguments", {})
                if tool_registry.get(name):
                    calls.append((name, args))
            except (json.JSONDecodeError, AttributeError):
                pass  # Malformed tool_call block — skip and try next match

        if calls:
            return calls

        # Pattern 2: ```json with "tool" key
        for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "tool" in data:
                    name = data.get("tool", "")
                    args = data.get("arguments", {})
                    if tool_registry.get(name):
                        calls.append((name, args))
            except (json.JSONDecodeError, AttributeError):
                pass  # Malformed JSON block — skip and try next match

        if calls:
            return calls

        # Pattern 3: bare JSON object with "tool" key (may contain nested braces)
        for match in re.finditer(r'\{[^{}]*"tool"\s*:\s*"[^"]+?"[^}]*\}', text):
            # Ensure balanced braces by extending to next closing brace if needed
            depth = 0
            end = match.start()
            for i in range(match.start(), len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            raw = text[match.start() : end]
            try:
                data = json.loads(raw)
                name = data.get("tool", "")
                args = data.get("arguments", {})
                if tool_registry.get(name):
                    calls.append((name, args))
            except (json.JSONDecodeError, AttributeError):
                pass  # Malformed bare JSON — skip and try next match

        return calls

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
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            )
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
        tool_model = os.environ.get("OLLAMA_TOOL_MODEL", DEFAULT_TOOL_MODEL)
        provider = OllamaProvider(
            config=ProviderConfig(
                provider_type=ProviderType.OLLAMA,
                base_url=host,
                default_model=model,
                timeout=600.0,
            ),
        )
        return AgentProvider(provider, tool_model=tool_model)

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
