"""
Animus Cognitive Layer

Handles reasoning, analysis, and response generation.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum

from animus.logging import get_logger

logger = get_logger("cognitive")


class ModelProvider(Enum):
    """Supported model providers."""

    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class ReasoningMode(Enum):
    """Different reasoning modes for different tasks."""

    QUICK = "quick"  # Fast response, shorter context
    DEEP = "deep"  # Extended thinking, full context
    BACKGROUND = "background"  # Async processing


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
    if config.provider == ModelProvider.OLLAMA:
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
    ):
        self.primary_config = primary_config or ModelConfig.ollama()
        self.fallback_config = fallback_config

        self.primary = create_model(self.primary_config)
        self.fallback = create_model(self.fallback_config) if self.fallback_config else None

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

    def _build_system_prompt(self, context: str | None, mode: ReasoningMode) -> str:
        """Build the system prompt based on context and mode."""
        base = """You are Animus, a personal AI assistant focused on being genuinely helpful.
You are direct, honest, and thoughtful. You remember context from past conversations
and use it to provide more relevant assistance.

You serve one user and are aligned with their interests."""

        if context:
            base += f"\n\nRelevant context from memory:\n{context}"

        if mode == ReasoningMode.DEEP:
            base += "\n\nTake your time to think through this carefully."

        return base
