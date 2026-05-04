"""Tests for backend get_by_id and MemoryManager.get_by_id."""

from __future__ import annotations

import pytest

from animus_bootstrap.intelligence.memory import MemoryManager
from animus_bootstrap.intelligence.memory_backends.sqlite_backend import (
    SQLiteMemoryBackend,
)


@pytest.fixture
def backend(tmp_path) -> SQLiteMemoryBackend:
    b = SQLiteMemoryBackend(tmp_path / "mem.db")
    yield b
    b.close()


class TestSQLiteGetById:
    @pytest.mark.asyncio
    async def test_returns_stored_memory(self, backend: SQLiteMemoryBackend) -> None:
        mid = await backend.store("episodic", "hello world", {"tag": "demo"})
        result = await backend.get_by_id(mid)
        assert result is not None
        assert result["id"] == mid
        assert result["content"] == "hello world"
        assert result["type"] == "episodic"
        assert result["metadata"] == {"tag": "demo"}

    @pytest.mark.asyncio
    async def test_unknown_id_returns_none(self, backend: SQLiteMemoryBackend) -> None:
        assert await backend.get_by_id("does-not-exist") is None


class TestMemoryManagerGetById:
    @pytest.mark.asyncio
    async def test_delegates_to_backend(self, backend: SQLiteMemoryBackend) -> None:
        mid = await backend.store("semantic", "fact", {})
        mm = MemoryManager(backend)
        result = await mm.get_by_id(mid)
        assert result is not None
        assert result["content"] == "fact"

    @pytest.mark.asyncio
    async def test_returns_none_for_backend_without_lookup(self) -> None:
        class NoLookupBackend:
            async def search(self, query, memory_type="all", limit=5):
                return []

            async def store(self, memory_type, content, metadata):
                return "x"

            async def delete(self, memory_id):
                return True

            async def get_stats(self):
                return {}

            def close(self) -> None:
                pass

        mm = MemoryManager(NoLookupBackend())
        assert await mm.get_by_id("any") is None
