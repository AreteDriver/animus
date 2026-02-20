"""Multi-Provider AI Support.

Provides a unified interface for multiple AI providers (OpenAI, Anthropic, Azure,
AWS Bedrock, Google Vertex AI, Ollama) with automatic fallback, streaming support,
tier-based routing, and provider-agnostic operations.
"""

from .anthropic_provider import AnthropicProvider
from .azure_openai_provider import AzureOpenAIProvider
from .base import (
    CompletionRequest,
    CompletionResponse,
    ModelTier,
    Provider,
    ProviderConfig,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderType,
    RateLimitError,
    StreamChunk,
)
from .bedrock_provider import BedrockProvider
from .hardware import HardwareProfile, detect_hardware
from .manager import (
    ProviderManager,
    get_manager,
    get_provider,
    list_providers,
    reset_manager,
)
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .router import RoutingConfig, RoutingDecision, RoutingMode, TierRouter
from .vertex_provider import VertexProvider

__all__ = [
    # Base classes
    "Provider",
    "ProviderConfig",
    "ProviderType",
    "ModelTier",
    "CompletionRequest",
    "CompletionResponse",
    "StreamChunk",
    "ProviderError",
    "ProviderNotConfiguredError",
    "RateLimitError",
    # Implementations
    "OpenAIProvider",
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "BedrockProvider",
    "VertexProvider",
    "OllamaProvider",
    # Manager
    "ProviderManager",
    "get_provider",
    "get_manager",
    "list_providers",
    "reset_manager",
    # Router
    "TierRouter",
    "RoutingConfig",
    "RoutingMode",
    "RoutingDecision",
    # Hardware
    "detect_hardware",
    "HardwareProfile",
]
