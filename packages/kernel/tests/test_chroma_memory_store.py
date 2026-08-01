"""Tests for ChromaMemoryStore (conditional — skipped if chromadb unavailable).

Covers:
- store / retrieve / update / delete
- search with vector similarity (mocked or real)
- list_all and get_all_tags
- allowed_tiers filtering
- _build_chroma_metadata serialization
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from animus_kernel.memory.types import Memory, MemoryTier, MemoryType, Sensitivity

pytestmark = pytest.mark.skipif(
    pytest.importorskip("chromadb", reason="chromadb not installed") is None,
    reason="chromadb not installed",
)

from animus_kernel.memory.stores.chroma import ChromaMemoryStore  # noqa: E402


@pytest.fixture
def chroma_store(tmp_path: Path) -> ChromaMemoryStore:
    return ChromaMemoryStore(data_dir=tmp_path, collection_name="test_memories")


def _make_memory(content: str = "hello", **kwargs) -> Memory:
    return Memory.create(content=content, memory_type=MemoryType.SEMANTIC, **kwargs)


class TestChromaStoreCRUD:
    def test_store_and_retrieve(self, chroma_store: ChromaMemoryStore):
        mem = _make_memory("chroma test")
        chroma_store.store(mem)
        retrieved = chroma_store.retrieve(mem.id)
        assert retrieved is not None
        assert retrieved.content == "chroma test"

    def test_retrieve_missing_returns_none(self, chroma_store: ChromaMemoryStore):
        assert chroma_store.retrieve("nonexistent") is None

    def test_update_existing(self, chroma_store: ChromaMemoryStore):
        mem = _make_memory("original")
        chroma_store.store(mem)
        mem.content = "updated"
        ok = chroma_store.update(mem)
        assert ok is True
        retrieved = chroma_store.retrieve(mem.id)
        assert retrieved.content == "updated"

    def test_update_missing_returns_false(self, chroma_store: ChromaMemoryStore):
        mem = _make_memory("orphan")
        ok = chroma_store.update(mem)
        assert ok is False

    def test_delete_existing(self, chroma_store: ChromaMemoryStore):
        mem = _make_memory("to delete")
        chroma_store.store(mem)
        ok = chroma_store.delete(mem.id)
        assert ok is True
        assert chroma_store.retrieve(mem.id) is None

    def test_delete_missing_returns_false(self, chroma_store: ChromaMemoryStore):
        assert chroma_store.delete("nonexistent") is False


class TestChromaStoreSearch:
    def test_search_returns_results(self, chroma_store: ChromaMemoryStore):
        chroma_store.store(_make_memory("apple pie recipe"))
        chroma_store.store(_make_memory("banana bread recipe"))
        chroma_store.store(_make_memory("cherry tart"))

        results = chroma_store.search("recipe", limit=10)
        # Semantic search should find the recipe memories
        assert len(results) >= 1

    def test_search_by_memory_type(self, chroma_store: ChromaMemoryStore):
        chroma_store.store(_make_memory("semantic memory", memory_type=MemoryType.SEMANTIC))
        chroma_store.store(_make_memory("episodic memory", memory_type=MemoryType.EPISODIC))
        results = chroma_store.search("memory", memory_type=MemoryType.EPISODIC, limit=10)
        assert len(results) >= 1
        assert results[0].memory_type == MemoryType.EPISODIC

    def test_search_by_source(self, chroma_store: ChromaMemoryStore):
        chroma_store.store(_make_memory("learned", source="learned"))
        chroma_store.store(_make_memory("stated", source="stated"))
        results = chroma_store.search("learned", source="learned", limit=10)
        assert len(results) >= 1

    def test_search_min_confidence(self, chroma_store: ChromaMemoryStore):
        chroma_store.store(_make_memory("high", confidence=0.9))
        chroma_store.store(_make_memory("low", confidence=0.3))
        results = chroma_store.search("high", min_confidence=0.5, limit=10)
        assert all(r.confidence >= 0.5 for r in results)

    def test_search_allowed_tiers(self, chroma_store: ChromaMemoryStore):
        chroma_store.store(_make_memory("secret", sensitivity=Sensitivity.SECRET))
        chroma_store.store(_make_memory("public", sensitivity=Sensitivity.PUBLIC))
        results = chroma_store.search("secret", allowed_tiers={Sensitivity.PUBLIC}, limit=10)
        assert all(r.sensitivity == Sensitivity.PUBLIC for r in results)

    def test_search_no_match(self, chroma_store: ChromaMemoryStore):
        results = chroma_store.search("xyzabc123nonexistent", limit=10)
        assert results == []


class TestChromaStoreListAndTags:
    def test_list_all(self, chroma_store: ChromaMemoryStore):
        chroma_store.store(_make_memory("one"))
        chroma_store.store(_make_memory("two"))
        assert len(chroma_store.list_all()) == 2

    def test_list_all_by_type(self, chroma_store: ChromaMemoryStore):
        chroma_store.store(_make_memory("s", memory_type=MemoryType.SEMANTIC))
        chroma_store.store(_make_memory("e", memory_type=MemoryType.EPISODIC))
        results = chroma_store.list_all(memory_type=MemoryType.EPISODIC)
        assert len(results) == 1
        assert results[0].memory_type == MemoryType.EPISODIC

    def test_get_all_tags(self, chroma_store: ChromaMemoryStore):
        m1 = _make_memory("a", tags=["foo", "bar"])
        m2 = _make_memory("b", tags=["foo"])
        chroma_store.store(m1)
        chroma_store.store(m2)
        tags = chroma_store.get_all_tags()
        assert tags["foo"] == 2
        assert tags["bar"] == 1


class TestChromaMetadata:
    def test_build_chroma_metadata_serializes_all_fields(self, chroma_store: ChromaMemoryStore):
        mem = Memory.create(
            content="test",
            memory_type=MemoryType.SEMANTIC,
            tags=["tag1"],
            source="stated",
            confidence=0.8,
            subtype="fact",
            version=2,
            parent_id="parent-123",
            change_summary="updated",
            provenance="sync",
            sensitivity=Sensitivity.CONFIDENTIAL,
            tier=MemoryTier.HOT,
            access_count=5,
            last_accessed=datetime(2024, 1, 1, 12, 0),
        )
        meta = chroma_store._build_chroma_metadata(mem)
        assert meta["memory_type"] == "semantic"
        assert meta["tags"] == '["tag1"]'
        assert meta["source"] == "stated"
        assert meta["confidence"] == 0.8
        assert meta["version"] == "2"
        assert meta["subtype"] == "fact"
        assert meta["parent_id"] == "parent-123"
        assert meta["change_summary"] == "updated"
        assert meta["provenance"] == "sync"
        assert meta["sensitivity"] == "confidential"
        assert meta["tier"] == "hot"
        assert meta["access_count"] == 5
        assert meta["last_accessed"] == "2024-01-01T12:00:00"

    def test_build_chroma_metadata_omits_optional_fields(self, chroma_store: ChromaMemoryStore):
        mem = Memory.create(content="minimal")
        meta = chroma_store._build_chroma_metadata(mem)
        assert "subtype" not in meta
        assert "parent_id" not in meta
        assert "change_summary" not in meta

    def test_load_metadata_reconstructs_memory(self, chroma_store: ChromaMemoryStore):
        mem = Memory.create(
            content="reload",
            tags=["reload"],
            sensitivity=Sensitivity.PERSONAL,
            tier=MemoryTier.COLD,
        )
        chroma_store.store(mem)
        # Re-init should load metadata
        store2 = ChromaMemoryStore(
            data_dir=chroma_store.data_dir.parent,
            collection_name="test_memories",
        )
        retrieved = store2.retrieve(mem.id)
        assert retrieved is not None
        assert retrieved.content == "reload"
        assert retrieved.sensitivity == Sensitivity.PERSONAL
        assert retrieved.tier == MemoryTier.COLD


class TestChromaPrewarm:
    def test_prewarm_returns_bool(self):
        result = ChromaMemoryStore.prewarm()
        assert isinstance(result, bool)


class TestChromaBM25:
    def test_bm25_search_returns_ids(self, chroma_store: ChromaMemoryStore):
        chroma_store.store(_make_memory("apple pie"))
        chroma_store.store(_make_memory("banana bread"))
        ids = chroma_store._bm25_search("apple", limit=5)
        # BM25 requires rank_bm25 package; if not present, returns []
        assert isinstance(ids, list)

    def test_rebuild_bm25_no_crash_on_empty(self, chroma_store: ChromaMemoryStore):
        chroma_store._rebuild_bm25()
        assert chroma_store._bm25_dirty is False


class TestChromaPersistence:
    def test_data_survives_reinit(self, tmp_path: Path):
        store1 = ChromaMemoryStore(data_dir=tmp_path, collection_name="persist")
        mem = _make_memory("persistent")
        store1.store(mem)

        store2 = ChromaMemoryStore(data_dir=tmp_path, collection_name="persist")
        retrieved = store2.retrieve(mem.id)
        assert retrieved is not None
        assert retrieved.content == "persistent"

    def test_different_collections_isolate_data(self, tmp_path: Path):
        store1 = ChromaMemoryStore(data_dir=tmp_path, collection_name="c1")
        store2 = ChromaMemoryStore(data_dir=tmp_path, collection_name="c2")
        mem = _make_memory("isolated")
        store1.store(mem)
        assert store2.retrieve(mem.id) is None
