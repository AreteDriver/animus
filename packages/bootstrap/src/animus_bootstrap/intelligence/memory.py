"""Memory manager — bridges gateway with memory backends."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from animus_bootstrap.intelligence.memory_backends.base import MemoryBackend

logger = logging.getLogger(__name__)


@dataclass
class MemoryContext:
    """Relevant memories injected into LLM prompt."""

    episodic: list[str] = field(default_factory=list)
    semantic: list[str] = field(default_factory=list)
    procedural: list[str] = field(default_factory=list)
    user_prefs: dict[str, str] = field(default_factory=dict)


class MemoryManager:
    """Bridges gateway with memory backends."""

    def __init__(self, backend: MemoryBackend) -> None:
        self._backend = backend

    async def recall(self, query: str, limit: int = 5) -> MemoryContext:
        """Retrieve relevant memories for a query."""
        episodic_results = await self._backend.search(query, memory_type="episodic", limit=limit)
        semantic_results = await self._backend.search(query, memory_type="semantic", limit=limit)
        procedural_results = await self._backend.search(
            query, memory_type="procedural", limit=limit
        )

        # Extract user preferences from backend stats
        stats = await self._backend.get_stats()
        user_prefs: dict[str, str] = stats.get("user_preferences", {})

        return MemoryContext(
            episodic=[r["content"] for r in episodic_results],
            semantic=[r["content"] for r in semantic_results],
            procedural=[r["content"] for r in procedural_results],
            user_prefs=user_prefs,
        )

    async def store_conversation(self, session_id: str, messages: list[dict]) -> None:
        """Store a completed conversation turn as episodic memory."""
        content = json.dumps(messages)
        metadata = {"session_id": session_id, "message_count": len(messages)}
        await self._backend.store("episodic", content, metadata)

    async def store_fact(self, subject: str, predicate: str, obj: str) -> None:
        """Store a knowledge triple in semantic memory."""
        content = f"{subject} {predicate} {obj}"
        metadata = {"subject": subject, "predicate": predicate, "object": obj}
        await self._backend.store("semantic", content, metadata)

    async def store_preference(self, key: str, value: str) -> None:
        """Store a user preference."""
        await self._backend.store(
            "procedural",
            f"User preference: {key} = {value}",
            {"preference_key": key, "preference_value": value},
        )
        # Also store in the dedicated preferences table if backend supports it
        if hasattr(self._backend, "store_preference"):
            await self._backend.store_preference(key, value)

    async def search(self, query: str, memory_type: str = "all", limit: int = 20) -> list[dict]:
        """Full-text search across memory stores."""
        return await self._backend.search(query, memory_type=memory_type, limit=limit)

    async def get_stats(self) -> dict:
        """Return memory statistics (counts per type, storage size)."""
        return await self._backend.get_stats()

    def close(self) -> None:
        """Close the underlying backend."""
        self._backend.close()
