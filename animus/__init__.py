"""Animus - An exocortex architecture for personal cognitive sovereignty."""

__version__ = "0.1.0"

from animus.cognitive import CognitiveLayer, ModelConfig, ReasoningMode
from animus.config import AnimusConfig
from animus.memory import Conversation, Memory, MemoryLayer, MemoryType

__all__ = [
    "AnimusConfig",
    "CognitiveLayer",
    "Conversation",
    "Memory",
    "MemoryLayer",
    "MemoryType",
    "ModelConfig",
    "ReasoningMode",
    "__version__",
]
