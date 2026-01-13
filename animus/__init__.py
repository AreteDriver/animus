"""Animus - An exocortex architecture for personal cognitive sovereignty."""

__version__ = "0.1.0"

from animus.cognitive import CognitiveLayer, ModelConfig, ReasoningMode
from animus.memory import MemoryLayer, MemoryType

__all__ = [
    "CognitiveLayer",
    "MemoryLayer",
    "ModelConfig",
    "ReasoningMode",
    "MemoryType",
    "__version__",
]
