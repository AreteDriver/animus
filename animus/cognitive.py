"""
Animus Cognitive Layer

Handles reasoning, analysis, and response generation.
Phase 2: Tool use, analysis modes, briefings.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from animus.logging import get_logger
from animus.protocols.intelligence import IntelligenceProvider

if TYPE_CHECKING:
    from animus.learning import LearningLayer
    from animus.memory import MemoryLayer
    from animus.tools import ToolRegistry

logger = get_logger("cognitive")


class ModelProvider(Enum):
    """Supported model providers."""

    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    MOCK = "mock"


class ReasoningMode(Enum):
    """Different reasoning modes for different tasks."""

    QUICK = "quick"  # Fast response, shorter context
    DEEP = "deep"  # Extended thinking, full context
    RESEARCH = "research"  # Web search + synthesis
    BACKGROUND = "background"  # Async processing


# Patterns for automatic mode detection
MODE_PATTERNS = {
    ReasoningMode.DEEP: [
        r"^think\s+(about|through)",
        r"^analyze",
        r"^consider",
        r"^explain\s+in\s+detail",
        r"^what\s+are\s+the\s+(pros|cons|implications)",
    ],
    ReasoningMode.RESEARCH: [
        r"^research",
        r"^find\s+out",
        r"^look\s+up",
        r"^what\s+is\s+the\s+latest",
        r"^search\s+for",
    ],
}


def detect_mode(prompt: str) -> ReasoningMode:
    """Detect reasoning mode from prompt patterns."""
    prompt_lower = prompt.lower().strip()
    for mode, patterns in MODE_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, prompt_lower):
                return mode
    return ReasoningMode.QUICK


@dataclass
class ModelConfig:
    """Configuration for a model."""

    provider: ModelProvider
    model_name: str
    api_key: str | None = None
    base_url: str | None = None

    @classmethod
    def ollama(cls, model: str = "llama3:8b") -> "ModelConfig":
        return cls(
            provider=ModelProvider.OLLAMA, model_name=model, base_url="http://localhost:11434"
        )

    @classmethod
    def anthropic(cls, model: str = "claude-3-haiku-20240307") -> "ModelConfig":
        return cls(
            provider=ModelProvider.ANTHROPIC,
            model_name=model,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

    @classmethod
    def openai(cls, model: str = "gpt-4-turbo-preview") -> "ModelConfig":
        return cls(
            provider=ModelProvider.OPENAI,
            model_name=model,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    @classmethod
    def mock(
        cls,
        default_response: str = "This is a mock response.",
        response_map: dict[str, str] | None = None,
    ) -> "ModelConfig":
        config = cls(provider=ModelProvider.MOCK, model_name="mock")
        config._mock_default_response = default_response
        config._mock_response_map = response_map or {}
        return config


class ModelInterface(ABC):
    """Abstract interface for language models."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a response to a prompt."""
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        """Generate a streaming response."""
        pass


class MockModel(ModelInterface):
    """Deterministic mock model for testing without a live LLM backend."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.default_response: str = getattr(
            config, "_mock_default_response", "This is a mock response."
        )
        self.response_map: dict[str, str] = getattr(config, "_mock_response_map", {})
        self.calls: list[dict] = []
        logger.debug("MockModel initialized")

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return a deterministic response, checking response_map first."""
        self.calls.append({"prompt": prompt, "system": system})
        for substring, response in self.response_map.items():
            if substring in prompt:
                return response
        return self.default_response

    async def generate_stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        """Yield the response in chunks."""
        response = self.generate(prompt, system)
        chunk_size = max(1, len(response) // 3)
        for i in range(0, len(response), chunk_size):
            yield response[i : i + chunk_size]

    def reset(self) -> None:
        """Clear call history."""
        self.calls.clear()


class OllamaModel(ModelInterface):
    """Ollama local model interface."""

    def __init__(self, config: ModelConfig):
        self.config = config
        logger.debug(f"OllamaModel initialized with {config.model_name}")

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate using Ollama."""
        try:
            import ollama

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            logger.debug(
                f"Ollama request: model={self.config.model_name}, prompt_len={len(prompt)}"
            )
            response = ollama.chat(model=self.config.model_name, messages=messages)
            result = response["message"]["content"]
            logger.debug(f"Ollama response: len={len(result)}")
            return result

        except ImportError:
            logger.error("ollama package not installed")
            return "[Error: ollama package not installed]"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"[Error communicating with Ollama: {e}]"

    async def generate_stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        """Streaming generation - to be implemented."""
        yield self.generate(prompt, system)


class AnthropicModel(ModelInterface):
    """Anthropic Claude model interface."""

    def __init__(self, config: ModelConfig):
        self.config = config
        logger.debug(f"AnthropicModel initialized with {config.model_name}")

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate using Claude."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.config.api_key)

            logger.debug(
                f"Anthropic request: model={self.config.model_name}, prompt_len={len(prompt)}"
            )
            message = client.messages.create(
                model=self.config.model_name,
                max_tokens=1024,
                system=system or "You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            result = message.content[0].text
            logger.debug(f"Anthropic response: len={len(result)}")
            return result

        except ImportError:
            logger.error("anthropic package not installed")
            return "[Error: anthropic package not installed]"
        except Exception as e:
            logger.error(f"Anthropic error: {e}")
            return f"[Error communicating with Anthropic: {e}]"

    async def generate_stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        """Streaming generation - to be implemented."""
        yield self.generate(prompt, system)


def create_model(config: ModelConfig) -> ModelInterface:
    """Factory function to create the appropriate model interface."""
    if config.provider == ModelProvider.MOCK:
        return MockModel(config)
    elif config.provider == ModelProvider.OLLAMA:
        return OllamaModel(config)
    elif config.provider == ModelProvider.ANTHROPIC:
        return AnthropicModel(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


class CognitiveLayer:
    """
    Main cognitive layer interface.

    Handles model selection, context assembly, and response generation.
    """

    def __init__(
        self,
        primary_config: ModelConfig | None = None,
        fallback_config: ModelConfig | None = None,
        learning: "LearningLayer | None" = None,
    ):
        self.primary_config = primary_config or ModelConfig.ollama()
        self.fallback_config = fallback_config
        self.learning = learning

        self.primary: IntelligenceProvider = create_model(self.primary_config)
        self.fallback: IntelligenceProvider | None = (
            create_model(self.fallback_config) if self.fallback_config else None
        )

        logger.info(
            f"CognitiveLayer initialized: primary={self.primary_config.provider.value}, "
            f"model={self.primary_config.model_name}"
        )

    def think(
        self,
        prompt: str,
        context: str | None = None,
        mode: ReasoningMode = ReasoningMode.QUICK,
    ) -> str:
        """
        Generate a thoughtful response.

        Args:
            prompt: The user's input
            context: Relevant context from memory
            mode: Reasoning mode to use

        Returns:
            Generated response
        """
        # Build system prompt
        system = self._build_system_prompt(context, mode)
        logger.debug(
            f"Thinking with mode={mode.value}, context_len={len(context) if context else 0}"
        )

        # Try primary model
        try:
            return self.primary.generate(prompt, system)
        except Exception as e:
            logger.warning(f"Primary model failed: {e}")
            # Fall back if available
            if self.fallback:
                logger.info("Falling back to secondary model")
                return self.fallback.generate(prompt, system)
            raise e

    def _build_system_prompt(
        self,
        context: str | None,
        mode: ReasoningMode,
        tools_schema: str | None = None,
    ) -> str:
        """Build the system prompt based on context, mode, and tools."""
        base = """You are Animus, a personal AI assistant focused on being genuinely helpful.
You are direct, honest, and thoughtful. You remember context from past conversations
and use it to provide more relevant assistance.

You serve one user and are aligned with their interests."""

        # Apply learned preferences for communication style
        if self.learning:
            preferences = self.learning.get_preferences("communication")
            if preferences:
                pref_hints = []
                for pref in preferences:
                    if pref.confidence >= 0.6:
                        pref_hints.append(f"- {pref.value}")
                if pref_hints:
                    base += "\n\nLearned user preferences:\n" + "\n".join(pref_hints)

        if context:
            base += f"\n\nRelevant context from memory:\n{context}"

        if mode == ReasoningMode.DEEP:
            base += (
                "\n\nThink through this carefully, step by step. Consider multiple perspectives."
            )
        elif mode == ReasoningMode.RESEARCH:
            base += "\n\nResearch this topic thoroughly. Use web search if available."

        if tools_schema:
            base += f"""

{tools_schema}

To use a tool, respond with a JSON block in this format:
```tool
{{"tool": "tool_name", "params": {{"param1": "value1"}}}}
```

You can use multiple tools. After tool results are returned, continue your response.
When you have gathered enough information, provide your final answer."""

        return base

    def think_with_tools(
        self,
        prompt: str,
        context: str | None = None,
        mode: ReasoningMode = ReasoningMode.QUICK,
        tools: "ToolRegistry | None" = None,
        max_iterations: int = 5,
        approval_callback: "callable | None" = None,
    ) -> str:
        """
        Generate a response with tool use capability.

        This implements an agentic loop that:
        1. Generates a response (possibly with tool calls)
        2. Parses and executes tool calls
        3. Feeds results back to the model
        4. Repeats until no more tool calls or max iterations

        Args:
            prompt: User's input
            context: Relevant context from memory
            mode: Reasoning mode
            tools: ToolRegistry with available tools
            max_iterations: Maximum tool use iterations
            approval_callback: Function to approve tools that require it
                               (tool_name, params) -> bool

        Returns:
            Final response after tool execution
        """
        if not tools or not tools.list_tools():
            # No tools available, fall back to regular think
            return self.think(prompt, context, mode)

        tools_schema = tools.get_schema_text()
        system = self._build_system_prompt(context, mode, tools_schema)

        messages = [{"role": "user", "content": prompt}]
        final_response = ""

        for iteration in range(max_iterations):
            logger.debug(f"Tool iteration {iteration + 1}/{max_iterations}")

            # Generate response
            full_prompt = self._format_messages(messages)
            response = self.primary.generate(full_prompt, system)

            # Parse tool calls
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # No tool calls, we're done
                final_response = response
                break

            # Execute tools
            tool_results = []
            response_text = self._remove_tool_blocks(response)

            for tool_name, params in tool_calls:
                tool = tools.get(tool_name)
                if not tool:
                    tool_results.append(f"[Error: Unknown tool '{tool_name}']")
                    continue

                # Check approval for sensitive tools
                if tool.requires_approval and approval_callback:
                    if not approval_callback(tool_name, params):
                        tool_results.append(f"[Tool '{tool_name}' was not approved]")
                        continue

                # Execute tool
                result = tools.execute(tool_name, params)
                tool_results.append(result.to_context())
                logger.debug(f"Tool {tool_name} result: success={result.success}")

            # Add to conversation
            if response_text.strip():
                messages.append({"role": "assistant", "content": response_text})

            # Add tool results
            results_text = "\n\n".join(tool_results)
            messages.append({"role": "user", "content": f"Tool results:\n{results_text}"})

        else:
            # Max iterations reached
            logger.warning(f"Max tool iterations ({max_iterations}) reached")
            final_response = response

        return final_response

    def _format_messages(self, messages: list[dict]) -> str:
        """Format message history for the model."""
        lines = []
        for msg in messages:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        return "\n\n".join(lines)

    def _parse_tool_calls(self, response: str) -> list[tuple[str, dict]]:
        """Parse tool calls from model response."""
        tool_calls = []

        # Look for ```tool blocks
        pattern = r"```tool\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match.strip())
                tool_name = data.get("tool")
                params = data.get("params", {})
                if tool_name:
                    tool_calls.append((tool_name, params))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {match[:50]}...")
                continue

        return tool_calls

    def _remove_tool_blocks(self, response: str) -> str:
        """Remove tool call blocks from response."""
        pattern = r"```tool\s*\n?.*?\n?```"
        return re.sub(pattern, "", response, flags=re.DOTALL).strip()

    def brief(
        self,
        memory: "MemoryLayer",
        topic: str | None = None,
        limit: int = 10,
    ) -> str:
        """
        Generate a situation briefing from memory.

        Args:
            memory: MemoryLayer to query for context
            topic: Optional topic to focus on (searches all if None)
            limit: Maximum memories to include

        Returns:
            Briefing text
        """
        # Gather relevant memories
        if topic:
            memories = memory.recall(topic, limit=limit)
        else:
            memories = memory.store.list_all()[:limit]

        if not memories:
            return "No memories available for briefing."

        # Format memories for context
        memory_texts = []
        for mem in memories:
            date_str = mem.created_at.strftime("%Y-%m-%d")
            tags_str = f" [{', '.join(mem.tags)}]" if mem.tags else ""
            memory_texts.append(f"- [{date_str}] {mem.content[:200]}{tags_str}")

        context = "\n".join(memory_texts)

        # Generate briefing
        prompt = f"""Generate a concise situation briefing based on the following memories.
Focus on key facts, recent developments, and actionable items.
{"Topic: " + topic if topic else "General briefing"}

Memories:
{context}

Provide a clear, structured briefing."""

        return self.think(prompt, mode=ReasoningMode.QUICK)
