"""Tests for ChromaDBMemoryBackend — mocked chromadb dependency."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest


def _run(coro):
    """Run an async coroutine synchronously without closing the global event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


class TestChromaDBBackendInit:
    def test_raises_without_chromadb(self):
        with patch.dict("sys.modules", {"chromadb": None}):
            # Force re-evaluation of HAS_CHROMADB

            from animus_bootstrap.intelligence.memory_backends import chromadb_backend

            original = chromadb_backend.HAS_CHROMADB
            chromadb_backend.HAS_CHROMADB = False
            try:
                with pytest.raises(RuntimeError, match="chromadb not installed"):
                    chromadb_backend.ChromaDBMemoryBackend()
            finally:
                chromadb_backend.HAS_CHROMADB = original

    def test_init_with_mock_chromadb(self):
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_chromadb.Client.return_value = mock_client
        mock_client.get_or_create_collection.return_value = MagicMock()

        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            # Temporarily enable
            import animus_bootstrap.intelligence.memory_backends.chromadb_backend as mod

            original = mod.HAS_CHROMADB
            mod.HAS_CHROMADB = True
            try:
                backend = mod.ChromaDBMemoryBackend()
                assert backend._collections is not None
            finally:
                mod.HAS_CHROMADB = original


class TestChromaDBBackendOperations:
    """Test store/search/delete/stats with mocked chromadb."""

    @pytest.fixture()
    def backend(self):
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_chromadb.Client.return_value = mock_client

        # Create mock collections
        collections = {}
        for t in ("episodic", "semantic", "procedural"):
            col = MagicMock()
            col.count.return_value = 0
            col.add = MagicMock()
            col.query.return_value = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }
            col.delete = MagicMock()
            collections[f"{t}_memories"] = col

        mock_client.get_or_create_collection.side_effect = lambda name: collections[name]

        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            import animus_bootstrap.intelligence.memory_backends.chromadb_backend as mod

            original = mod.HAS_CHROMADB
            mod.HAS_CHROMADB = True
            try:
                b = mod.ChromaDBMemoryBackend()
                b._mock_collections = collections
                yield b
            finally:
                mod.HAS_CHROMADB = original

    def test_store_calls_add(self, backend):
        mem_id = _run(backend.store("episodic", "test content", {"key": "val"}))
        assert isinstance(mem_id, str)
        # Verify collection.add was called
        col = backend._collections["episodic"]
        col.add.assert_called_once()

    def test_store_invalid_type(self, backend):
        with pytest.raises(ValueError, match="Invalid memory type"):
            _run(backend.store("invalid", "content", {}))

    def test_search_empty(self, backend):
        results = _run(backend.search("test query"))
        assert results == []

    def test_search_with_results(self, backend):
        col = backend._collections["episodic"]
        col.count.return_value = 1
        col.query.return_value = {
            "ids": [["id1"]],
            "documents": [["found content"]],
            "metadatas": [[{"key": "val"}]],
            "distances": [[0.5]],
        }
        results = _run(backend.search("test", memory_type="episodic"))
        assert len(results) == 1
        assert results[0]["content"] == "found content"
        assert results[0]["memory_type"] == "episodic"

    def test_delete(self, backend):
        result = _run(backend.delete("some-id"))
        assert result is True

    def test_get_stats(self, backend):
        stats = _run(backend.get_stats())
        assert stats["backend"] == "chromadb"
        assert "collections" in stats
        assert stats["total"] == 0

    def test_close(self, backend):
        backend.close()  # Should not raise


class TestRuntimeFallback:
    """Test that runtime falls back to SQLite when ChromaDB unavailable."""

    def test_chromadb_fallback_in_config(self):
        from animus_bootstrap.config.schema import AnimusConfig

        config = AnimusConfig()
        assert config.intelligence.memory_backend == "sqlite"
