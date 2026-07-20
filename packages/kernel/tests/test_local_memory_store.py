"""Tests for LocalMemoryStore (JSON file-backed fallback store).

Covers:
- store / retrieve / update / delete cycle
- search with substring matching and filters
- list_all and get_all_tags
- persistence across re-initialization
- allowed_tiers filtering
"""

from __future__ import annotations

from pathlib import Path

import pytest

from animus_kernel.memory.stores.local import LocalMemoryStore
from animus_kernel.memory.types import Memory, MemoryType, Sensitivity


@pytest.fixture
def local_store(tmp_path: Path) -> LocalMemoryStore:
    return LocalMemoryStore(data_dir=tmp_path)


def _make_memory(content: str = "hello", memory_type: MemoryType = MemoryType.SEMANTIC, **kwargs) -> Memory:
    return Memory.create(content=content, memory_type=memory_type, **kwargs)


class TestLocalStoreCRUD:
    def test_store_and_retrieve(self, local_store: LocalMemoryStore):
        mem = _make_memory("test content")
        local_store.store(mem)
        retrieved = local_store.retrieve(mem.id)
        assert retrieved is not None
        assert retrieved.content == "test content"
        assert retrieved.id == mem.id

    def test_retrieve_missing_returns_none(self, local_store: LocalMemoryStore):
        assert local_store.retrieve("nonexistent") is None

    def test_update_existing(self, local_store: LocalMemoryStore):
        mem = _make_memory("original")
        local_store.store(mem)
        mem.content = "updated"
        ok = local_store.update(mem)
        assert ok is True
        retrieved = local_store.retrieve(mem.id)
        assert retrieved.content == "updated"

    def test_update_missing_returns_false(self, local_store: LocalMemoryStore):
        mem = _make_memory("orphan")
        ok = local_store.update(mem)
        assert ok is False

    def test_delete_existing(self, local_store: LocalMemoryStore):
        mem = _make_memory("to delete")
        local_store.store(mem)
        ok = local_store.delete(mem.id)
        assert ok is True
        assert local_store.retrieve(mem.id) is None

    def test_delete_missing_returns_false(self, local_store: LocalMemoryStore):
        assert local_store.delete("nonexistent") is False


class TestLocalStoreSearch:
    def test_substring_search(self, local_store: LocalMemoryStore):
        local_store.store(_make_memory("apple pie"))
        local_store.store(_make_memory("banana bread"))
        local_store.store(_make_memory("cherry"))

        results = local_store.search("pie")
        assert len(results) == 1
        assert results[0].content == "apple pie"

    def test_search_limit(self, local_store: LocalMemoryStore):
        for i in range(5):
            local_store.store(_make_memory(f"item {i}"))
        results = local_store.search("item", limit=3)
        assert len(results) == 3

    def test_search_by_memory_type(self, local_store: LocalMemoryStore):
        local_store.store(_make_memory("semantic memory", memory_type=MemoryType.SEMANTIC))
        local_store.store(_make_memory("episodic memory", memory_type=MemoryType.EPISODIC))
        results = local_store.search("memory", memory_type=MemoryType.EPISODIC)
        assert len(results) == 1
        assert results[0].memory_type == MemoryType.EPISODIC

    def test_search_by_tags(self, local_store: LocalMemoryStore):
        m1 = _make_memory("a", tags=["foo", "bar"])
        m2 = _make_memory("b", tags=["foo"])
        local_store.store(m1)
        local_store.store(m2)
        results = local_store.search("a", tags=["foo", "bar"])
        assert len(results) == 1
        assert results[0].content == "a"

    def test_search_by_source(self, local_store: LocalMemoryStore):
        m1 = _make_memory("learned", source="learned")
        m2 = _make_memory("stated", source="stated")
        local_store.store(m1)
        local_store.store(m2)
        results = local_store.search("learned", source="learned")
        assert len(results) == 1

    def test_search_min_confidence(self, local_store: LocalMemoryStore):
        m1 = _make_memory("high", confidence=0.9)
        m2 = _make_memory("low", confidence=0.3)
        local_store.store(m1)
        local_store.store(m2)
        results = local_store.search("h", min_confidence=0.5)
        assert len(results) == 1
        assert results[0].content == "high"

    def test_search_allowed_tiers(self, local_store: LocalMemoryStore):
        m1 = _make_memory("secret", sensitivity=Sensitivity.SECRET)
        m2 = _make_memory("public", sensitivity=Sensitivity.PUBLIC)
        local_store.store(m1)
        local_store.store(m2)
        results = local_store.search("c", allowed_tiers={Sensitivity.PUBLIC})
        assert len(results) == 1
        assert results[0].sensitivity == Sensitivity.PUBLIC

    def test_search_no_match(self, local_store: LocalMemoryStore):
        local_store.store(_make_memory("xyz"))
        results = local_store.search("abc")
        assert results == []


class TestLocalStoreListAndTags:
    def test_list_all(self, local_store: LocalMemoryStore):
        local_store.store(_make_memory("one"))
        local_store.store(_make_memory("two"))
        assert len(local_store.list_all()) == 2

    def test_list_all_by_type(self, local_store: LocalMemoryStore):
        local_store.store(_make_memory("s", memory_type=MemoryType.SEMANTIC))
        local_store.store(_make_memory("e", memory_type=MemoryType.EPISODIC))
        results = local_store.list_all(memory_type=MemoryType.EPISODIC)
        assert len(results) == 1
        assert results[0].memory_type == MemoryType.EPISODIC

    def test_get_all_tags(self, local_store: LocalMemoryStore):
        m1 = _make_memory("a", tags=["foo", "bar"])
        m2 = _make_memory("b", tags=["foo"])
        local_store.store(m1)
        local_store.store(m2)
        tags = local_store.get_all_tags()
        assert tags["foo"] == 2
        assert tags["bar"] == 1

    def test_get_all_tags_empty(self, local_store: LocalMemoryStore):
        assert local_store.get_all_tags() == {}


class TestLocalStorePersistence:
    def test_data_survives_reinit(self, tmp_path: Path):
        store1 = LocalMemoryStore(data_dir=tmp_path)
        mem = _make_memory("persistent")
        store1.store(mem)

        store2 = LocalMemoryStore(data_dir=tmp_path)
        retrieved = store2.retrieve(mem.id)
        assert retrieved is not None
        assert retrieved.content == "persistent"

    def test_atomic_write(self, tmp_path: Path):
        store = LocalMemoryStore(data_dir=tmp_path)
        store.store(_make_memory("atomic"))
        # tmp file should not exist after successful write
        assert not (tmp_path / "memories.json.tmp").exists()

    def test_empty_store_load(self, tmp_path: Path):
        store = LocalMemoryStore(data_dir=tmp_path)
        assert store.list_all() == []

    def test_malformed_json_not_present(self, tmp_path: Path):
        # If JSON is malformed, load should propagate the error
        (tmp_path / "memories.json").write_text("not json")
        with pytest.raises(Exception):
            LocalMemoryStore(data_dir=tmp_path)
