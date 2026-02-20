"""Protocol for memory storage backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from animus.memory import Memory, MemoryType


@runtime_checkable
class MemoryProvider(Protocol):
    """Structural interface for memory storage backends."""

    def store(self, memory: Memory) -> None: ...
    def update(self, memory: Memory) -> bool: ...
    def retrieve(self, memory_id: str) -> Memory | None: ...
    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> list[Memory]: ...
    def delete(self, memory_id: str) -> bool: ...
    def list_all(self, memory_type: MemoryType | None = None) -> list[Memory]: ...
    def get_all_tags(self) -> dict[str, int]: ...
