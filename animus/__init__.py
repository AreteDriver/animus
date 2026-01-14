"""Animus - An exocortex architecture for personal cognitive sovereignty."""

__version__ = "0.5.0"

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

# Optional imports - available when dependencies are installed
try:
    from animus.api import APIServer
except ImportError:
    APIServer = None  # type: ignore[misc, assignment]

try:
    from animus.voice import VoiceInput, VoiceInterface, VoiceOutput
except ImportError:
    VoiceInput = None  # type: ignore[misc, assignment]
    VoiceInterface = None  # type: ignore[misc, assignment]
    VoiceOutput = None  # type: ignore[misc, assignment]

try:
    from animus.learning import (
        LearningLayer,
        LearningCategory,
        LearnedItem,
        GuardrailManager,
        Guardrail,
    )
except ImportError:
    LearningLayer = None  # type: ignore[misc, assignment]
    LearningCategory = None  # type: ignore[misc, assignment]
    LearnedItem = None  # type: ignore[misc, assignment]
    GuardrailManager = None  # type: ignore[misc, assignment]
    Guardrail = None  # type: ignore[misc, assignment]

__all__ = [
    "APIServer",
    "AnimusConfig",
    "CognitiveLayer",
    "Conversation",
    "Decision",
    "DecisionFramework",
    "Guardrail",
    "GuardrailManager",
    "LearnedItem",
    "LearningCategory",
    "LearningLayer",
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
    "VoiceInput",
    "VoiceInterface",
    "VoiceOutput",
    "__version__",
    "create_default_registry",
    "detect_mode",
]
