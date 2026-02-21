"""Memory tools — store memories and set reminders via LLM tool use."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from animus_bootstrap.intelligence.tools.executor import ToolDefinition

logger = logging.getLogger(__name__)

# In-memory stores for stub implementation
_stored_memories: list[dict] = []
_pending_reminders: list[dict] = []


async def _store_memory(content: str, memory_type: str = "semantic") -> str:
    """Stub: store a memory entry."""
    entry = {
        "content": content,
        "memory_type": memory_type,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    _stored_memories.append(entry)
    return f"Stored {memory_type} memory: {content[:100]}"


async def _set_reminder(message: str, delay_minutes: int) -> str:
    """Stub: schedule a reminder."""
    fire_at = datetime.now(UTC) + timedelta(minutes=delay_minutes)
    reminder = {
        "message": message,
        "delay_minutes": delay_minutes,
        "scheduled_for": fire_at.isoformat(),
    }
    _pending_reminders.append(reminder)
    return f"Reminder set for {fire_at.isoformat()}: {message}"


def get_stored_memories() -> list[dict]:
    """Return all stored memories (for testing / inspection)."""
    return list(_stored_memories)


def get_pending_reminders() -> list[dict]:
    """Return all pending reminders (for testing / inspection)."""
    return list(_pending_reminders)


def clear_memory_stores() -> None:
    """Clear all stored memories and reminders."""
    _stored_memories.clear()
    _pending_reminders.clear()


def get_memory_tools() -> list[ToolDefinition]:
    """Return memory tool definitions."""
    return [
        ToolDefinition(
            name="store_memory",
            description="Store a piece of information in memory.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to remember.",
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Memory type: 'episodic', 'semantic', or 'procedural'.",
                        "default": "semantic",
                    },
                },
                "required": ["content"],
            },
            handler=_store_memory,
            category="memory",
        ),
        ToolDefinition(
            name="set_reminder",
            description="Set a reminder to fire after a delay.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The reminder message.",
                    },
                    "delay_minutes": {
                        "type": "integer",
                        "description": "Minutes from now to fire the reminder.",
                    },
                },
                "required": ["message", "delay_minutes"],
            },
            handler=_set_reminder,
            category="memory",
        ),
    ]
