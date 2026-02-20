"""Agent memory and context persistence.

This module is a backward-compatibility shim. All classes have been refactored
into focused submodules:

- memory_models.py: MemoryEntry, MessageRole, Message, ContextWindowStats
- agent_memory.py: AgentMemory
- context_window.py: ContextWindow
"""

from .agent_memory import AgentMemory
from .context_window import ContextWindow
from .memory_models import ContextWindowStats, MemoryEntry, Message, MessageRole

__all__ = [
    "MemoryEntry",
    "MessageRole",
    "Message",
    "ContextWindowStats",
    "AgentMemory",
    "ContextWindow",
]
