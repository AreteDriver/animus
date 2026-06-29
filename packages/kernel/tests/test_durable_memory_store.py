"""Tests for DurableMemoryStore using an in-memory SQLite database.

These tests verify that the durable store correctly implements the
MemoryStore interface and writes events to the ledger, without requiring
a live PostgreSQL instance.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from animus_kernel.memory.stores.durable import DurableMemoryStore
from animus_kernel.memory.types import Memory, MemoryType, Sensitivity

# Use SQLite in-memory for tests — SQLAlchemy abstracts the dialect.
_TEST_DB_URL = "sqlite:///:memory:"


@pytest.fixture
def store(tmp_path: Path):
    """Create a DurableMemoryStore backed by an in-memory SQLite database."""
    # Ensure the env var doesn't leak from the host
    old_env = os.environ.get("ANIMUS_DATABASE_URL")
    os.environ["ANIMUS_DATABASE_URL"] = _TEST_DB_URL

    ds = DurableMemoryStore(owner_id="test-owner", workspace_id="test-ws")

    # Create tables manually (Alembic isn't run in test)
    from animus_kernel.memory.stores.durable import (
        _EventLedgerRow,
        _ObjectRegistryRow,
    )

    _ObjectRegistryRow.metadata.create_all(ds._engine)

    yield ds

    # Cleanup
    if old_env is None:
        os.environ.pop("ANIMUS_DATABASE_URL", None)
    else:
        os.environ["ANIMUS_DATABASE_URL"] = old_env


def _make_memory(content: str = "hello world") -> Memory:
    return Memory.create(
        content=content,
        memory_type=MemoryType.SEMANTIC,
        tags=["test"],
        source="stated",
        confidence=1.0,
    )


def test_store_creates_registry_row(store: DurableMemoryStore):
    mem = _make_memory("store test")
    store.store(mem)

    retrieved = store.retrieve(mem.id)
    assert retrieved is not None
    assert retrieved.content == "store test"
    assert retrieved.memory_type == MemoryType.SEMANTIC


def test_store_writes_event_to_ledger(store: DurableMemoryStore):
    mem = _make_memory("ledger test")
    store.store(mem)

    # Verify event was written
    from animus_kernel.memory.stores.durable import _EventLedgerRow

    with store._session_factory() as session:
        events = session.query(_EventLedgerRow).all()
        assert len(events) >= 1
        assert any(e.event_kind == "memory.stored" for e in events)


def test_update_increments_version(store: DurableMemoryStore):
    mem = _make_memory("version test")
    store.store(mem)

    mem.content = "updated content"
    ok = store.update(mem)
    assert ok is True

    retrieved = store.retrieve(mem.id)
    assert retrieved is not None
    assert retrieved.content == "updated content"
    assert retrieved.version == 2


def test_delete_marks_superseded(store: DurableMemoryStore):
    mem = _make_memory("delete test")
    store.store(mem)

    ok = store.delete(mem.id)
    assert ok is True

    retrieved = store.retrieve(mem.id)
    assert retrieved is None


def test_search_finds_matching_content(store: DurableMemoryStore):
    store.store(_make_memory("apple pie recipe"))
    store.store(_make_memory("banana bread recipe"))
    store.store(_make_memory("unrelated"))

    results = store.search("recipe", limit=10)
    assert len(results) == 2
    contents = {r.content for r in results}
    assert contents == {"apple pie recipe", "banana bread recipe"}


def test_search_respects_tier_filter(store: DurableMemoryStore):
    mem = _make_memory("secret")
    mem.sensitivity = Sensitivity.CONFIDENTIAL
    store.store(mem)

    public = store.search("secret", allowed_tiers={Sensitivity.PUBLIC})
    assert len(public) == 0

    confidential = store.search("secret", allowed_tiers={Sensitivity.CONFIDENTIAL})
    assert len(confidential) == 1


def test_list_all_returns_non_deleted(store: DurableMemoryStore):
    m1 = _make_memory("one")
    m2 = _make_memory("two")
    store.store(m1)
    store.store(m2)
    store.delete(m1.id)

    all_memories = store.list_all()
    assert len(all_memories) == 1
    assert all_memories[0].content == "two"


def test_get_all_tags_counts(store: DurableMemoryStore):
    m1 = _make_memory("a")
    m1.tags = ["foo", "bar"]
    m2 = _make_memory("b")
    m2.tags = ["foo"]
    store.store(m1)
    store.store(m2)

    tags = store.get_all_tags()
    assert tags.get("foo") == 2
    assert tags.get("bar") == 1
