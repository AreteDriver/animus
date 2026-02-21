"""ChromaDB memory backend — vector similarity search (optional dependency)."""

from __future__ import annotations

try:
    import chromadb  # noqa: F401

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class ChromaDBMemoryBackend:
    """ChromaDB-backed memory store with vector similarity search."""

    def __init__(self, persist_directory: str | None = None) -> None:
        if not HAS_CHROMADB:
            msg = "chromadb not installed. pip install animus-bootstrap[chromadb]"
            raise RuntimeError(msg)
        self._persist_directory = persist_directory
        raise NotImplementedError("ChromaDB backend not yet implemented")

    async def store(self, memory_type: str, content: str, metadata: dict) -> str:
        """Store a memory entry, return its ID."""
        raise NotImplementedError

    async def search(self, query: str, memory_type: str = "all", limit: int = 5) -> list[dict]:
        """Search memories by vector similarity."""
        raise NotImplementedError

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        raise NotImplementedError

    async def get_stats(self) -> dict:
        """Return backend statistics."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the backend."""
