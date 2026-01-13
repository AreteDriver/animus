"""Animus - An exocortex architecture for personal cognitive sovereignty."""

__version__ = "0.1.0"

from animus.cognitive import CognitiveLayer, ModelConfig, ReasoningMode
from animus.config import AnimusConfig
from animus.memory import (
    Conversation,
    Memory,
    MemoryLayer,
    MemorySource,
    MemoryType,
    Procedure,
    SemanticFact,
)

__all__ = [
    "AnimusConfig",
    "CognitiveLayer",
    "Conversation",
    "Memory",
    "MemoryLayer",
    "MemorySource",
    "MemoryType",
    "ModelConfig",
    "Procedure",
    "ReasoningMode",
    "SemanticFact",
    "__version__",
]
