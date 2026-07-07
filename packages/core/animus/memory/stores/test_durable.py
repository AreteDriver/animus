"""Tests for :class:`DurableMemoryStore`.

Uses SQLite in-memory via DurableObjectStore so no PostgreSQL server is required.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from animus.memory.stores.durable import (
    DurableMemoryStore,
    _memory_to_record,
    _record_to_memory,
    _security_to_sensitivity,
    _sensitivity_to_security,
    _storage_to_tier,
    _tier_to_storage,
)
from animus.memory.types import Memory, MemoryTier, MemoryType, Sensitivity

pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    s = DurableMemoryStore(database_url=url, workspace_id="ws-test")
    s.create_tables()
    yield s


@pytest.fixture
def sample_memory():
    return Memory.create(
        content="Test memory content",
        memory_type=MemoryType.SEMANTIC,
        tags=["test", "durable"],
        confidence=0.95,
    )


class TestRoundTrip:
    """Memory survives store → retrieve."""

    def test_store_and_retrieve(self, store, sample_memory):
        store.store(sample_memory)
        retrieved = store.retrieve(sample_memory.id)
        assert retrieved is not None
        assert retrieved.id == sample_memory.id
        assert retrieved.content == sample_memory.content
        assert retrieved.tags == sample_memory.tags
        assert retrieved.confidence == pytest.approx(0.95)

    def test_retrieve_missing(self, store):
        assert store.retrieve("nonexistent-id") is None

    def test_store_generates_ledger(self, store, sample_memory):
        store.store(sample_memory)
        events = store.get_ledger_events(sample_memory.id)
        assert len(events) == 1
        assert events[0]["event_type"] == "created"
        assert store.verify_integrity(events[0]["event_id"]) is True


class TestUpdate:
    """Updates are ledgered and versioned."""

    def test_update(self, store, sample_memory):
        store.store(sample_memory)
        sample_memory.content = "Updated content"
        assert store.update(sample_memory) is True

        retrieved = store.retrieve(sample_memory.id)
        assert retrieved.content == "Updated content"

        events = store.get_ledger_events(sample_memory.id)
        assert any(e["event_type"] == "updated" for e in events)

    def test_update_missing(self, store, sample_memory):
        assert store.update(sample_memory) is False


class TestDelete:
    """Deletion is soft (versioned history preserved)."""

    def test_delete(self, store, sample_memory):
        store.store(sample_memory)
        assert store.delete(sample_memory.id) is True
        assert store.retrieve(sample_memory.id) is None

        # But history exists
        events = store.get_ledger_events(sample_memory.id)
        assert any(e["event_type"] == "deleted" for e in events)

    def test_delete_missing(self, store):
        assert store.delete("nonexistent") is False


class TestSearch:
    """Substring search with filters."""

    def test_search_content(self, store):
        m1 = Memory.create(content="hello world", tags=["greeting"])
        m2 = Memory.create(content="goodbye world", tags=["farewell"])
        store.store(m1)
        store.store(m2)

        results = store.search("hello")
        assert len(results) == 1
        assert results[0].id == m1.id

    def test_search_tags(self, store):
        m1 = Memory.create(content="alpha", tags=["important"])
        m2 = Memory.create(content="beta", tags=["other"])
        store.store(m1)
        store.store(m2)

        results = store.search("a", tags=["important"])
        assert len(results) == 1
        assert results[0].id == m1.id

    def test_search_memory_type(self, store):
        m1 = Memory.create(content="fact", memory_type=MemoryType.SEMANTIC)
        m2 = Memory.create(content="chat", memory_type=MemoryType.EPISODIC)
        store.store(m1)
        store.store(m2)

        results = store.search("a", memory_type=MemoryType.EPISODIC)
        assert len(results) == 1
        assert results[0].id == m2.id

    def test_search_sensitivity(self, store):
        m1 = Memory.create(content="public info", sensitivity=Sensitivity.PUBLIC)
        m2 = Memory.create(content="secret", sensitivity=Sensitivity.SECRET)
        store.store(m1)
        store.store(m2)

        results = store.search("info", allowed_tiers={Sensitivity.PUBLIC})
        assert len(results) == 1
        assert results[0].id == m1.id

    def test_search_limit(self, store):
        for i in range(5):
            store.store(Memory.create(content=f"item {i}"))
        results = store.search("item", limit=2)
        assert len(results) == 2


class TestListAll:
    """Bulk listing."""

    def test_list_all(self, store):
        store.store(Memory.create(content="a"))
        store.store(Memory.create(content="b"))
        assert len(store.list_all()) == 2

    def test_list_all_by_type(self, store):
        store.store(Memory.create(content="fact", memory_type=MemoryType.SEMANTIC))
        store.store(Memory.create(content="chat", memory_type=MemoryType.EPISODIC))
        assert len(store.list_all(MemoryType.SEMANTIC)) == 1


class TestTags:
    """Tag aggregation."""

    def test_get_all_tags(self, store):
        store.store(Memory.create(content="a", tags=["x", "y"]))
        store.store(Memory.create(content="b", tags=["x"]))
        tags = store.get_all_tags()
        assert tags["x"] == 2
        assert tags["y"] == 1


class TestMapping:
    """Conversion between Memory and ObjectRecord."""

    def test_memory_to_record_and_back(self, sample_memory):
        record = _memory_to_record(sample_memory)
        # object_id is schema-compliant; original UUID lives in payload
        assert record.object_id.startswith("mem-")
        assert record.payload["memory_id"] == sample_memory.id
        assert record.payload["content"] == sample_memory.content
        assert record.tags == sample_memory.tags
        assert record.artifact_type == "memory"

        memory = _record_to_memory(record)
        assert memory.id == sample_memory.id
        assert memory.content == sample_memory.content
        assert memory.tags == sample_memory.tags

    def test_tier_mapping(self):
        assert _tier_to_storage(MemoryTier.HOT) == "hot"
        assert _tier_to_storage(MemoryTier.WARM) == "warm"
        assert _tier_to_storage(MemoryTier.COLD) == "cold"
        assert _storage_to_tier("hot") == MemoryTier.HOT
        assert _storage_to_tier("warm") == MemoryTier.WARM
        assert _storage_to_tier("cold") == MemoryTier.COLD

    def test_sensitivity_mapping(self):
        assert _sensitivity_to_security(Sensitivity.PUBLIC) == "public"
        assert _sensitivity_to_security(Sensitivity.PERSONAL) == "internal"
        assert _sensitivity_to_security(Sensitivity.CONFIDENTIAL) == "confidential"
        assert _sensitivity_to_security(Sensitivity.SECRET) == "restricted"
        assert _security_to_sensitivity("public") == Sensitivity.PUBLIC
        assert _security_to_sensitivity("internal") == Sensitivity.PERSONAL
        assert _security_to_sensitivity("confidential") == Sensitivity.CONFIDENTIAL
        assert _security_to_sensitivity("restricted") == Sensitivity.SECRET


class TestBitemporalExtras:
    """DurableObjectStore-specific features exposed through Memory."""

    def test_as_of_valid_time(self, store, sample_memory):
        store.store(sample_memory)
        # Far future should still see the memory
        future = datetime.now(timezone.utc)
        result = store.as_of_valid_time(sample_memory.id, future)
        assert result is not None
        assert result.content == sample_memory.content

    def test_as_of_transaction_time(self, store, sample_memory):
        store.store(sample_memory)
        future = datetime.now(timezone.utc)
        result = store.as_of_transaction_time(sample_memory.id, future)
        assert result is not None
        assert result.content == sample_memory.content

    def test_outbox_entries(self, store, sample_memory):
        store.store(sample_memory)
        entries = store.claim_outbox_entries("worker-1", limit=10)
        assert len(entries) >= 1
        assert entries[0]["topic"] == "object.created"

        # Acknowledge
        entry_id = entries[0]["entry_id"]
        assert store.acknowledge_outbox_entry(entry_id) is True
