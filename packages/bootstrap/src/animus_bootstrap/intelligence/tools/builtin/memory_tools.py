"""Memory tools — store and recall memories via LLM tool use."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from animus_bootstrap.intelligence.tools.executor import ToolDefinition

logger = logging.getLogger(__name__)

# In-memory stores for fallback when no MemoryManager is wired
_stored_memories: list[dict] = []
_pending_reminders: list[dict] = []

# Live MemoryManager reference — set at runtime
_memory_manager = None


def set_memory_manager(manager: object) -> None:
    """Wire the live MemoryManager for memory tools."""
    global _memory_manager  # noqa: PLW0603
    _memory_manager = manager


async def _store_memory(content: str, memory_type: str = "semantic") -> str:
    """Store a memory entry.

    Delegates to the live MemoryManager if available, otherwise uses
    in-memory fallback list.
    """
    if _memory_manager is not None:
        try:
            backend = _memory_manager._backend
            await backend.store(memory_type, content, {"source": "tool"})
            logger.info("Stored %s memory via backend: %s", memory_type, content[:60])
            return f"Stored {memory_type} memory: {content[:100]}"
        except (OSError, ConnectionError, RuntimeError, ValueError) as exc:
            logger.warning("Backend store failed, using fallback: %s", exc)

    entry = {
        "content": content,
        "memory_type": memory_type,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    _stored_memories.append(entry)
    return f"Stored {memory_type} memory: {content[:100]}"


async def _recall_memory(query: str, memory_type: str = "all", limit: int = 10) -> str:
    """Search memories by query.

    Delegates to the live MemoryManager if available.
    """
    if _memory_manager is not None:
        try:
            results = await _memory_manager.search(query, memory_type=memory_type, limit=limit)
            if not results:
                return f"No memories found for: {query}"
            lines = [f"Found {len(results)} memories:"]
            for r in results:
                content = r.get("content", "")[:200]
                mtype = r.get("memory_type", "unknown")
                lines.append(f"  [{mtype}] {content}")
            return "\n".join(lines)
        except (OSError, ConnectionError, RuntimeError, ValueError) as exc:
            logger.warning("Backend search failed: %s", exc)
            return f"Memory search failed: {exc}"

    # Fallback: search in-memory list
    matches = []
    for m in _stored_memories:
        if memory_type != "all" and m["memory_type"] != memory_type:
            continue
        if query.lower() in m["content"].lower():
            matches.append(m)
    if not matches:
        return f"No memories found for: {query}"
    lines = [f"Found {len(matches[:limit])} memories:"]
    for m in matches[:limit]:
        lines.append(f"  [{m['memory_type']}] {m['content'][:200]}")
    return "\n".join(lines)


async def _set_reminder(message: str, delay_minutes: int) -> str:
    """Schedule a reminder nudge.

    If a ProactiveEngine is available (via timer_ctl), registers a one-shot
    timer. Otherwise stores in the in-memory fallback.
    """
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
            description=(
                "Store a piece of information in long-term memory. "
                "Types: 'episodic' (experiences), 'semantic' (facts), 'procedural' (how-to)."
            ),
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
            name="recall_memory",
            description=(
                "Search long-term memory by query. "
                "Returns matching memories across episodic, semantic, and procedural stores."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant memories.",
                    },
                    "memory_type": {
                        "type": "string",
                        "description": ("Filter: 'all', 'episodic', 'semantic', 'procedural'."),
                        "default": "all",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            handler=_recall_memory,
            category="memory",
        ),
        ToolDefinition(
            name="set_reminder",
            description="Set a reminder to fire after a delay in minutes.",
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
