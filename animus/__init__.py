"""Animus - An exocortex architecture for personal cognitive sovereignty."""

__version__ = "0.2.0"

from animus.cognitive import CognitiveLayer, ModelConfig, ReasoningMode, detect_mode
from animus.config import AnimusConfig
from animus.decision import Decision, DecisionFramework
from animus.memory import (
    Conversation,
    Memory,
    MemoryLayer,
    MemorySource,
    MemoryType,
    Procedure,
    SemanticFact,
)
from animus.tasks import Task, TaskStatus, TaskTracker
from animus.tools import Tool, ToolRegistry, ToolResult, create_default_registry

__all__ = [
    "AnimusConfig",
    "CognitiveLayer",
    "Conversation",
    "Decision",
    "DecisionFramework",
    "Memory",
    "MemoryLayer",
    "MemorySource",
    "MemoryType",
    "ModelConfig",
    "Procedure",
    "ReasoningMode",
    "SemanticFact",
    "Task",
    "TaskStatus",
    "TaskTracker",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "__version__",
    "create_default_registry",
    "detect_mode",
]
