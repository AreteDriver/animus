"""ChromaDB memory backend — vector similarity search (optional dependency)."""

from __future__ import annotations

import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)

try:
    import chromadb  # noqa: F401

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class ChromaDBMemoryBackend:
    """ChromaDB-backed memory store with vector similarity search.

    Uses ChromaDB's default embedding function (all-MiniLM-L6-v2).
    Three collections: episodic_memories, semantic_memories, procedural_memories.
    All operations wrapped in ``asyncio.to_thread()`` for async compat.
    """

    VALID_TYPES = ("episodic", "semantic", "procedural")

    def __init__(self, persist_directory: str | None = None) -> None:
        if not HAS_CHROMADB:
            msg = "chromadb not installed. pip install animus-bootstrap[chromadb]"
            raise RuntimeError(msg)

        import chromadb as _chromadb

        if persist_directory:
            self._client = _chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = _chromadb.Client()

        self._collections = {
            t: self._client.get_or_create_collection(name=f"{t}_memories") for t in self.VALID_TYPES
        }
        logger.info(
            "ChromaDB backend initialized: %s",
            persist_directory or "in-memory",
        )

    def _get_collection(self, memory_type: str):  # noqa: ANN202
        """Get collection for a memory type, raising ValueError if invalid."""
        if memory_type not in self.VALID_TYPES:
            msg = f"Invalid memory type: {memory_type}. Must be one of {self.VALID_TYPES}"
            raise ValueError(msg)
        return self._collections[memory_type]

    async def store(self, memory_type: str, content: str, metadata: dict) -> str:
        """Store a memory entry via vector embedding, return its ID."""
        collection = self._get_collection(memory_type)
        memory_id = str(uuid.uuid4())

        def _add():
            collection.add(
                ids=[memory_id],
                documents=[content],
                metadatas=[metadata] if metadata else None,
            )

        await asyncio.to_thread(_add)
        return memory_id

    async def search(self, query: str, memory_type: str = "all", limit: int = 5) -> list[dict]:
        """Search memories by vector similarity."""
        results: list[dict] = []

        types_to_search = list(self.VALID_TYPES) if memory_type == "all" else [memory_type]

        for t in types_to_search:
            if t not in self._collections:
                continue
            collection = self._collections[t]

            query_limit = min(limit, 10)

            def _query(c=collection, lim=query_limit):
                count = c.count()
                if count == 0:
                    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
                actual_limit = min(lim, count)
                return c.query(query_texts=[query], n_results=actual_limit)

            query_result = await asyncio.to_thread(_query)

            ids = query_result.get("ids", [[]])[0]
            docs = query_result.get("documents", [[]])[0]
            metas = query_result.get("metadatas", [[]])[0]
            dists = query_result.get("distances", [[]])[0]

            for i, doc_id in enumerate(ids):
                results.append(
                    {
                        "id": doc_id,
                        "content": docs[i] if i < len(docs) else "",
                        "metadata": metas[i] if i < len(metas) else {},
                        "distance": dists[i] if i < len(dists) else 0.0,
                        "memory_type": t,
                    }
                )

        # Sort by distance (lower = more similar)
        results.sort(key=lambda r: r.get("distance", float("inf")))
        return results[:limit]

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID from all collections."""
        deleted = False
        for collection in self._collections.values():

            def _delete(c=collection):
                try:
                    c.delete(ids=[memory_id])
                    return True
                except Exception:
                    return False

            if await asyncio.to_thread(_delete):
                deleted = True
        return deleted

    async def get_stats(self) -> dict:
        """Return backend statistics."""
        stats: dict = {"backend": "chromadb", "collections": {}}
        for name, collection in self._collections.items():

            def _count(c=collection):
                return c.count()

            count = await asyncio.to_thread(_count)
            stats["collections"][name] = {"count": count}
        stats["total"] = sum(c["count"] for c in stats["collections"].values())
        return stats

    def close(self) -> None:
        """Close the backend (ChromaDB handles cleanup internally)."""
        logger.info("ChromaDB backend closed")
