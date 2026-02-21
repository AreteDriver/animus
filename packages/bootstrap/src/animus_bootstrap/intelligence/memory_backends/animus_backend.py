"""Animus Core memory backend — delegates to the core exocortex memory layer."""

from __future__ import annotations

from typing import Any


class AnimusMemoryBackend:
    """Backend that delegates to animus core's MemoryManager."""

    def __init__(self) -> None:
        try:
            from animus.memory import MemoryManager as CoreMemory  # noqa: F401

            self._core: Any = CoreMemory()
        except ImportError:
            msg = "animus core not installed. pip install animus"
            raise RuntimeError(msg) from None
        raise NotImplementedError("Animus core backend not yet implemented")

    async def store(self, memory_type: str, content: str, metadata: dict) -> str:
        """Store a memory entry, return its ID."""
        raise NotImplementedError

    async def search(self, query: str, memory_type: str = "all", limit: int = 5) -> list[dict]:
        """Search memories via the core memory layer."""
        raise NotImplementedError

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        raise NotImplementedError

    async def get_stats(self) -> dict:
        """Return backend statistics."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the backend."""
