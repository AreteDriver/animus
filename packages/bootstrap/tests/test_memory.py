"""Tests for the memory integration module."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_bootstrap.intelligence import MemoryContext, MemoryManager
from animus_bootstrap.intelligence.memory_backends import MemoryBackend, SQLiteMemoryBackend
from animus_bootstrap.intelligence.memory_backends.base import MemoryBackend as MemoryBackendProto

# ======================================================================
# MemoryContext
# ======================================================================


class TestMemoryContext:
    def test_default_fields(self) -> None:
        ctx = MemoryContext()
        assert ctx.episodic == []
        assert ctx.semantic == []
        assert ctx.procedural == []
        assert ctx.user_prefs == {}

    def test_populated_fields(self) -> None:
        ctx = MemoryContext(
            episodic=["conv1", "conv2"],
            semantic=["fact1"],
            procedural=["how-to-1"],
            user_prefs={"theme": "dark"},
        )
        assert len(ctx.episodic) == 2
        assert ctx.semantic == ["fact1"]
        assert ctx.procedural == ["how-to-1"]
        assert ctx.user_prefs["theme"] == "dark"

    def test_mutable_defaults_are_independent(self) -> None:
        ctx1 = MemoryContext()
        ctx2 = MemoryContext()
        ctx1.episodic.append("x")
        assert ctx2.episodic == []

    def test_user_prefs_independent(self) -> None:
        ctx1 = MemoryContext()
        ctx2 = MemoryContext()
        ctx1.user_prefs["key"] = "val"
        assert "key" not in ctx2.user_prefs


# ======================================================================
# MemoryBackend Protocol
# ======================================================================


class TestMemoryBackendProtocol:
    def test_sqlite_backend_is_memory_backend(self, tmp_path: Path) -> None:
        backend = SQLiteMemoryBackend(tmp_path / "test.db")
        assert isinstance(backend, MemoryBackend)
        backend.close()

    def test_protocol_is_runtime_checkable(self) -> None:
        assert hasattr(MemoryBackendProto, "__protocol_attrs__") or issubclass(
            MemoryBackendProto, MemoryBackend
        )


# ======================================================================
# SQLiteMemoryBackend
# ======================================================================


class TestSQLiteMemoryBackend:
    @pytest.fixture()
    def backend(self, tmp_path: Path) -> SQLiteMemoryBackend:
        b = SQLiteMemoryBackend(tmp_path / "memory.db")
        yield b
        b.close()

    # -- Schema / Init --

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        backend = SQLiteMemoryBackend(tmp_path / "wal.db")
        conn = sqlite3.connect(str(tmp_path / "wal.db"))
        result = conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "wal"
        conn.close()
        backend.close()

    def test_tables_created(self, backend: SQLiteMemoryBackend) -> None:
        cur = backend._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        assert "memories" in tables
        assert "user_preferences" in tables
        assert "memories_fts" in tables

    def test_index_created(self, backend: SQLiteMemoryBackend) -> None:
        cur = backend._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cur.fetchall()}
        assert "idx_memories_type" in indexes

    def test_triggers_created(self, backend: SQLiteMemoryBackend) -> None:
        cur = backend._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
        triggers = {row[0] for row in cur.fetchall()}
        assert "memories_ai" in triggers
        assert "memories_ad" in triggers
        assert "memories_au" in triggers

    # -- Store --

    @pytest.mark.asyncio()
    async def test_store_episodic(self, backend: SQLiteMemoryBackend) -> None:
        memory_id = await backend.store("episodic", "test conversation", {"session": "s1"})
        assert isinstance(memory_id, str)
        assert len(memory_id) == 36  # UUID length

    @pytest.mark.asyncio()
    async def test_store_semantic(self, backend: SQLiteMemoryBackend) -> None:
        memory_id = await backend.store("semantic", "Python is a language", {"topic": "pl"})
        assert isinstance(memory_id, str)

    @pytest.mark.asyncio()
    async def test_store_procedural(self, backend: SQLiteMemoryBackend) -> None:
        memory_id = await backend.store("procedural", "run pytest -v", {"tool": "pytest"})
        assert isinstance(memory_id, str)

    @pytest.mark.asyncio()
    async def test_store_invalid_type_raises(self, backend: SQLiteMemoryBackend) -> None:
        with pytest.raises(ValueError, match="Invalid memory type"):
            await backend.store("invalid_type", "content", {})

    @pytest.mark.asyncio()
    async def test_store_with_metadata(self, backend: SQLiteMemoryBackend) -> None:
        meta = {"key1": "value1", "nested": {"a": 1}}
        memory_id = await backend.store("episodic", "content with meta", meta)

        # Verify metadata stored correctly
        cur = backend._conn.cursor()
        cur.execute("SELECT metadata FROM memories WHERE id = ?", (memory_id,))
        stored_meta = json.loads(cur.fetchone()[0])
        assert stored_meta["key1"] == "value1"
        assert stored_meta["nested"]["a"] == 1

    @pytest.mark.asyncio()
    async def test_store_sets_timestamps(self, backend: SQLiteMemoryBackend) -> None:
        memory_id = await backend.store("episodic", "timestamp test", {})
        cur = backend._conn.cursor()
        cur.execute("SELECT created_at, updated_at FROM memories WHERE id = ?", (memory_id,))
        row = cur.fetchone()
        assert row[0] is not None
        assert row[1] is not None
        assert row[0] == row[1]  # created_at == updated_at on creation

    @pytest.mark.asyncio()
    async def test_store_empty_content(self, backend: SQLiteMemoryBackend) -> None:
        memory_id = await backend.store("episodic", "", {})
        assert isinstance(memory_id, str)

    # -- Search (FTS5) --

    @pytest.mark.asyncio()
    async def test_search_matching(self, backend: SQLiteMemoryBackend) -> None:
        await backend.store("episodic", "Python is great for automation", {})
        await backend.store("episodic", "Rust is great for performance", {})

        results = await backend.search("Python")
        assert len(results) >= 1
        assert any("Python" in r["content"] for r in results)

    @pytest.mark.asyncio()
    async def test_search_no_match(self, backend: SQLiteMemoryBackend) -> None:
        await backend.store("episodic", "Python is great", {})
        results = await backend.search("xyznonexistent")
        assert len(results) == 0

    @pytest.mark.asyncio()
    async def test_search_filtered_by_type(self, backend: SQLiteMemoryBackend) -> None:
        await backend.store("episodic", "conversation about Python", {})
        await backend.store("semantic", "Python is a programming language", {})

        episodic_results = await backend.search("Python", memory_type="episodic")
        semantic_results = await backend.search("Python", memory_type="semantic")

        assert all(r["type"] == "episodic" for r in episodic_results)
        assert all(r["type"] == "semantic" for r in semantic_results)

    @pytest.mark.asyncio()
    async def test_search_all_types(self, backend: SQLiteMemoryBackend) -> None:
        await backend.store("episodic", "conversation about databases", {})
        await backend.store("semantic", "SQLite is a database engine", {})
        await backend.store("procedural", "how to query a database", {})

        results = await backend.search("database", memory_type="all")
        types_found = {r["type"] for r in results}
        assert len(types_found) >= 2  # At least 2 types should match

    @pytest.mark.asyncio()
    async def test_search_with_limit(self, backend: SQLiteMemoryBackend) -> None:
        for i in range(10):
            await backend.store("episodic", f"memory number {i} about testing", {})

        results = await backend.search("testing", limit=3)
        assert len(results) <= 3

    @pytest.mark.asyncio()
    async def test_search_empty_query_returns_recent(self, backend: SQLiteMemoryBackend) -> None:
        await backend.store("episodic", "first memory", {})
        await backend.store("episodic", "second memory", {})

        results = await backend.search("", memory_type="episodic")
        assert len(results) == 2

    @pytest.mark.asyncio()
    async def test_search_empty_query_filtered(self, backend: SQLiteMemoryBackend) -> None:
        await backend.store("episodic", "episodic content", {})
        await backend.store("semantic", "semantic content", {})

        results = await backend.search("", memory_type="episodic")
        assert all(r["type"] == "episodic" for r in results)

    @pytest.mark.asyncio()
    async def test_search_empty_query_all_types(self, backend: SQLiteMemoryBackend) -> None:
        await backend.store("episodic", "episodic content", {})
        await backend.store("semantic", "semantic content", {})

        results = await backend.search("", memory_type="all")
        assert len(results) == 2

    @pytest.mark.asyncio()
    async def test_search_result_fields(self, backend: SQLiteMemoryBackend) -> None:
        await backend.store("episodic", "test content fields", {"key": "val"})
        results = await backend.search("content")
        assert len(results) == 1
        r = results[0]
        assert "id" in r
        assert r["type"] == "episodic"
        assert r["content"] == "test content fields"
        assert r["metadata"] == {"key": "val"}
        assert "created_at" in r
        assert "updated_at" in r

    @pytest.mark.asyncio()
    async def test_search_special_characters(self, backend: SQLiteMemoryBackend) -> None:
        """FTS5 should handle quotes in search query."""
        await backend.store("episodic", 'user said "hello world"', {})
        results = await backend.search("hello")
        assert len(results) >= 1

    # -- Delete --

    @pytest.mark.asyncio()
    async def test_delete_existing(self, backend: SQLiteMemoryBackend) -> None:
        memory_id = await backend.store("episodic", "to be deleted", {})
        deleted = await backend.delete(memory_id)
        assert deleted is True

        # Verify gone from search
        results = await backend.search("deleted")
        assert len(results) == 0

    @pytest.mark.asyncio()
    async def test_delete_nonexistent(self, backend: SQLiteMemoryBackend) -> None:
        deleted = await backend.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio()
    async def test_delete_removes_from_fts(self, backend: SQLiteMemoryBackend) -> None:
        memory_id = await backend.store("episodic", "unique searchable content xyzzy", {})
        results_before = await backend.search("xyzzy")
        assert len(results_before) == 1

        await backend.delete(memory_id)
        results_after = await backend.search("xyzzy")
        assert len(results_after) == 0

    # -- Stats --

    @pytest.mark.asyncio()
    async def test_get_stats_empty(self, backend: SQLiteMemoryBackend) -> None:
        stats = await backend.get_stats()
        assert stats["total_memories"] == 0
        assert stats["by_type"]["episodic"] == 0
        assert stats["by_type"]["semantic"] == 0
        assert stats["by_type"]["procedural"] == 0
        assert stats["preference_count"] == 0
        assert stats["user_preferences"] == {}
        assert stats["db_size_bytes"] >= 0

    @pytest.mark.asyncio()
    async def test_get_stats_with_data(self, backend: SQLiteMemoryBackend) -> None:
        await backend.store("episodic", "ep1", {})
        await backend.store("episodic", "ep2", {})
        await backend.store("semantic", "sem1", {})
        await backend.store("procedural", "proc1", {})

        stats = await backend.get_stats()
        assert stats["total_memories"] == 4
        assert stats["by_type"]["episodic"] == 2
        assert stats["by_type"]["semantic"] == 1
        assert stats["by_type"]["procedural"] == 1

    @pytest.mark.asyncio()
    async def test_get_stats_db_size(self, backend: SQLiteMemoryBackend) -> None:
        await backend.store("episodic", "some content", {})
        stats = await backend.get_stats()
        assert stats["db_size_bytes"] > 0

    @pytest.mark.asyncio()
    async def test_get_stats_with_preferences(self, backend: SQLiteMemoryBackend) -> None:
        await backend.store_preference("theme", "dark")
        await backend.store_preference("lang", "en")

        stats = await backend.get_stats()
        assert stats["preference_count"] == 2
        assert stats["user_preferences"]["theme"] == "dark"
        assert stats["user_preferences"]["lang"] == "en"

    # -- Preferences --

    @pytest.mark.asyncio()
    async def test_store_preference(self, backend: SQLiteMemoryBackend) -> None:
        await backend.store_preference("theme", "dark")
        value = await backend.get_preference("theme")
        assert value == "dark"

    @pytest.mark.asyncio()
    async def test_store_preference_upsert(self, backend: SQLiteMemoryBackend) -> None:
        await backend.store_preference("theme", "light")
        await backend.store_preference("theme", "dark")
        value = await backend.get_preference("theme")
        assert value == "dark"

    @pytest.mark.asyncio()
    async def test_get_preference_nonexistent(self, backend: SQLiteMemoryBackend) -> None:
        value = await backend.get_preference("nonexistent")
        assert value is None

    @pytest.mark.asyncio()
    async def test_get_all_preferences(self, backend: SQLiteMemoryBackend) -> None:
        await backend.store_preference("theme", "dark")
        await backend.store_preference("lang", "en")
        await backend.store_preference("tz", "UTC")

        prefs = await backend.get_all_preferences()
        assert len(prefs) == 3
        assert prefs["theme"] == "dark"
        assert prefs["lang"] == "en"
        assert prefs["tz"] == "UTC"

    @pytest.mark.asyncio()
    async def test_get_all_preferences_empty(self, backend: SQLiteMemoryBackend) -> None:
        prefs = await backend.get_all_preferences()
        assert prefs == {}

    # -- Close --

    def test_close(self, tmp_path: Path) -> None:
        backend = SQLiteMemoryBackend(tmp_path / "close.db")
        backend.close()
        # After close, operations should raise
        with pytest.raises(sqlite3.ProgrammingError):
            backend._conn.execute("SELECT 1")

    # -- In-memory DB --

    @pytest.mark.asyncio()
    async def test_in_memory_db(self) -> None:
        backend = SQLiteMemoryBackend(":memory:")
        memory_id = await backend.store("episodic", "in memory", {})
        assert isinstance(memory_id, str)
        results = await backend.search("memory")
        assert len(results) == 1
        backend.close()


# ======================================================================
# MemoryManager
# ======================================================================


class TestMemoryManager:
    @pytest.fixture()
    def mock_backend(self) -> MagicMock:
        backend = MagicMock()
        backend.search = AsyncMock(return_value=[])
        backend.store = AsyncMock(return_value="mock-id")
        backend.delete = AsyncMock(return_value=True)
        backend.get_stats = AsyncMock(
            return_value={
                "total_memories": 0,
                "by_type": {"episodic": 0, "semantic": 0, "procedural": 0},
                "preference_count": 0,
                "user_preferences": {},
                "db_size_bytes": 0,
            }
        )
        backend.close = MagicMock()
        return backend

    @pytest.fixture()
    def manager(self, mock_backend: MagicMock) -> MemoryManager:
        return MemoryManager(backend=mock_backend)

    # -- recall --

    @pytest.mark.asyncio()
    async def test_recall_returns_memory_context(
        self, manager: MemoryManager, mock_backend: MagicMock
    ) -> None:
        mock_backend.search = AsyncMock(
            side_effect=[
                [{"content": "conv1"}],  # episodic
                [{"content": "fact1"}],  # semantic
                [{"content": "howto1"}],  # procedural
            ]
        )
        mock_backend.get_stats = AsyncMock(return_value={"user_preferences": {"theme": "dark"}})

        ctx = await manager.recall("test query")
        assert isinstance(ctx, MemoryContext)
        assert ctx.episodic == ["conv1"]
        assert ctx.semantic == ["fact1"]
        assert ctx.procedural == ["howto1"]
        assert ctx.user_prefs == {"theme": "dark"}

    @pytest.mark.asyncio()
    async def test_recall_with_empty_results(
        self, manager: MemoryManager, mock_backend: MagicMock
    ) -> None:
        mock_backend.search = AsyncMock(return_value=[])
        mock_backend.get_stats = AsyncMock(return_value={"user_preferences": {}})

        ctx = await manager.recall("empty query")
        assert ctx.episodic == []
        assert ctx.semantic == []
        assert ctx.procedural == []
        assert ctx.user_prefs == {}

    @pytest.mark.asyncio()
    async def test_recall_passes_limit(
        self, manager: MemoryManager, mock_backend: MagicMock
    ) -> None:
        mock_backend.search = AsyncMock(return_value=[])
        mock_backend.get_stats = AsyncMock(return_value={"user_preferences": {}})

        await manager.recall("query", limit=10)
        # Each search call should use limit=10
        for call in mock_backend.search.call_args_list:
            assert call.kwargs.get("limit", call.args[2] if len(call.args) > 2 else 5) == 10

    @pytest.mark.asyncio()
    async def test_recall_searches_all_three_types(
        self, manager: MemoryManager, mock_backend: MagicMock
    ) -> None:
        mock_backend.search = AsyncMock(return_value=[])
        mock_backend.get_stats = AsyncMock(return_value={"user_preferences": {}})

        await manager.recall("query")
        types_searched = [call.kwargs["memory_type"] for call in mock_backend.search.call_args_list]
        assert "episodic" in types_searched
        assert "semantic" in types_searched
        assert "procedural" in types_searched

    # -- store_conversation --

    @pytest.mark.asyncio()
    async def test_store_conversation(
        self, manager: MemoryManager, mock_backend: MagicMock
    ) -> None:
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        await manager.store_conversation("session-1", messages)

        mock_backend.store.assert_awaited_once()
        call_args = mock_backend.store.call_args
        assert call_args.args[0] == "episodic"
        assert "hello" in call_args.args[1]
        assert call_args.args[2]["session_id"] == "session-1"
        assert call_args.args[2]["message_count"] == 2

    @pytest.mark.asyncio()
    async def test_store_conversation_empty_messages(
        self, manager: MemoryManager, mock_backend: MagicMock
    ) -> None:
        await manager.store_conversation("session-2", [])
        mock_backend.store.assert_awaited_once()
        assert mock_backend.store.call_args.args[2]["message_count"] == 0

    # -- store_fact --

    @pytest.mark.asyncio()
    async def test_store_fact(self, manager: MemoryManager, mock_backend: MagicMock) -> None:
        await manager.store_fact("Python", "is", "a programming language")

        mock_backend.store.assert_awaited_once()
        call_args = mock_backend.store.call_args
        assert call_args.args[0] == "semantic"
        assert call_args.args[1] == "Python is a programming language"
        assert call_args.args[2] == {
            "subject": "Python",
            "predicate": "is",
            "object": "a programming language",
        }

    # -- store_preference --

    @pytest.mark.asyncio()
    async def test_store_preference_delegates(
        self, manager: MemoryManager, mock_backend: MagicMock
    ) -> None:
        mock_backend.store_preference = AsyncMock()
        await manager.store_preference("theme", "dark")

        # Should store as procedural memory
        mock_backend.store.assert_awaited_once()
        call_args = mock_backend.store.call_args
        assert call_args.args[0] == "procedural"
        assert "theme" in call_args.args[1]
        assert "dark" in call_args.args[1]

        # Should also call backend's store_preference
        mock_backend.store_preference.assert_awaited_once_with("theme", "dark")

    @pytest.mark.asyncio()
    async def test_store_preference_without_backend_method(
        self, manager: MemoryManager, mock_backend: MagicMock
    ) -> None:
        # Backend without store_preference method should not raise
        if hasattr(mock_backend, "store_preference"):
            del mock_backend.store_preference
        await manager.store_preference("key", "value")
        mock_backend.store.assert_awaited_once()

    # -- search --

    @pytest.mark.asyncio()
    async def test_search_delegates(self, manager: MemoryManager, mock_backend: MagicMock) -> None:
        expected = [{"id": "1", "content": "result"}]
        mock_backend.search = AsyncMock(return_value=expected)

        results = await manager.search("query", memory_type="episodic", limit=10)
        assert results == expected
        mock_backend.search.assert_awaited_once_with("query", memory_type="episodic", limit=10)

    @pytest.mark.asyncio()
    async def test_search_defaults(self, manager: MemoryManager, mock_backend: MagicMock) -> None:
        await manager.search("query")
        mock_backend.search.assert_awaited_once_with("query", memory_type="all", limit=20)

    # -- get_stats --

    @pytest.mark.asyncio()
    async def test_get_stats_delegates(
        self, manager: MemoryManager, mock_backend: MagicMock
    ) -> None:
        expected_stats = {"total_memories": 5}
        mock_backend.get_stats = AsyncMock(return_value=expected_stats)

        stats = await manager.get_stats()
        assert stats == expected_stats
        mock_backend.get_stats.assert_awaited_once()

    # -- close --

    def test_close_delegates(self, manager: MemoryManager, mock_backend: MagicMock) -> None:
        manager.close()
        mock_backend.close.assert_called_once()


# ======================================================================
# MemoryManager + SQLiteMemoryBackend (Integration)
# ======================================================================


class TestMemoryManagerIntegration:
    @pytest.fixture()
    def manager(self, tmp_path: Path) -> MemoryManager:
        backend = SQLiteMemoryBackend(tmp_path / "integration.db")
        mgr = MemoryManager(backend=backend)
        yield mgr
        mgr.close()

    @pytest.mark.asyncio()
    async def test_store_and_recall_conversation(self, manager: MemoryManager) -> None:
        messages = [{"role": "user", "content": "What is Python?"}]
        await manager.store_conversation("s1", messages)

        ctx = await manager.recall("Python")
        assert len(ctx.episodic) >= 1
        assert "Python" in ctx.episodic[0]

    @pytest.mark.asyncio()
    async def test_store_and_recall_fact(self, manager: MemoryManager) -> None:
        await manager.store_fact("SQLite", "supports", "FTS5")
        ctx = await manager.recall("SQLite")
        assert len(ctx.semantic) >= 1
        assert "SQLite" in ctx.semantic[0]

    @pytest.mark.asyncio()
    async def test_store_preference_and_recall(self, manager: MemoryManager) -> None:
        await manager.store_preference("editor", "neovim")
        ctx = await manager.recall("editor")
        assert len(ctx.procedural) >= 1

    @pytest.mark.asyncio()
    async def test_full_lifecycle(self, manager: MemoryManager) -> None:
        # Store various memories
        await manager.store_conversation("s1", [{"role": "user", "content": "deploy help"}])
        await manager.store_fact("Docker", "uses", "containers")
        await manager.store_preference("lang", "Python")

        # Search across all
        results = await manager.search("deploy")
        assert len(results) >= 1

        # Stats
        stats = await manager.get_stats()
        assert stats["total_memories"] >= 3

    @pytest.mark.asyncio()
    async def test_recall_with_no_data(self, manager: MemoryManager) -> None:
        ctx = await manager.recall("anything")
        assert ctx.episodic == []
        assert ctx.semantic == []
        assert ctx.procedural == []


# ======================================================================
# ChromaDB Backend
# ======================================================================


class TestChromaDBBackend:
    def test_raises_without_chromadb(self) -> None:
        with patch.dict("sys.modules", {"chromadb": None}):
            # Need to reimport to trigger the import guard
            import importlib

            import animus_bootstrap.intelligence.memory_backends.chromadb_backend as mod

            importlib.reload(mod)
            assert mod.HAS_CHROMADB is False
            with pytest.raises(RuntimeError, match="chromadb not installed"):
                mod.ChromaDBMemoryBackend()


# ======================================================================
# Animus Backend
# ======================================================================


class TestAnimusBackend:
    def test_raises_without_animus(self) -> None:
        with patch.dict("sys.modules", {"animus": None, "animus.memory": None}):
            from animus_bootstrap.intelligence.memory_backends.animus_backend import (
                AnimusMemoryBackend,
            )

            with pytest.raises(RuntimeError, match="animus core not installed"):
                AnimusMemoryBackend()


# ======================================================================
# Module exports
# ======================================================================


class TestExports:
    def test_intelligence_init_exports(self) -> None:
        import sys

        intel = sys.modules["animus_bootstrap.intelligence"]
        assert hasattr(intel, "MemoryContext")
        assert hasattr(intel, "MemoryManager")

    def test_backends_init_exports(self) -> None:
        import sys

        backends = sys.modules["animus_bootstrap.intelligence.memory_backends"]
        assert hasattr(backends, "MemoryBackend")
        assert hasattr(backends, "SQLiteMemoryBackend")
