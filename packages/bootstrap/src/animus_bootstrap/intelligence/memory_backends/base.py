"""Memory backend protocol — defines the interface for memory stores."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol for memory storage backends."""

    async def store(self, memory_type: str, content: str, metadata: dict) -> str:
        """Store a memory entry, return its ID."""

    async def search(self, query: str, memory_type: str = "all", limit: int = 5) -> list[dict]:
        """Search memories by query text."""

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""

    async def get_stats(self) -> dict:
        """Return backend statistics."""

    def close(self) -> None:
        """Close the backend and release resources."""
