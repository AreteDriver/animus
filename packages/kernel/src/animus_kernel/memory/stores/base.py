"""Abstract base class for memory storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from animus_kernel.memory.types import Memory, MemoryType, Sensitivity


class MemoryStore(ABC):
    """Abstract base class for memory storage backends."""

    @abstractmethod
    def store(self, memory: Memory) -> None:
        """Store a memory."""
        pass

    @abstractmethod
    def update(self, memory: Memory) -> bool:
        """Update an existing memory."""
        pass

    @abstractmethod
    def retrieve(self, memory_id: str) -> Memory | None:
        """Retrieve a specific memory by ID."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
        allowed_tiers: set[Sensitivity] | None = None,
    ) -> list[Memory]:
        """Search memories with filters.

        When ``allowed_tiers`` is provided, results are restricted to memories
        whose ``sensitivity`` is in the set. When ``None`` (default), no tier
        filter is applied — backward-compatible with pre-Stage-2.B callers.
        """
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        pass

    @abstractmethod
    def list_all(self, memory_type: MemoryType | None = None) -> list[Memory]:
        """List all memories, optionally filtered by type."""
        pass

    @abstractmethod
    def get_all_tags(self) -> dict[str, int]:
        """Get all tags with their counts."""
        pass
