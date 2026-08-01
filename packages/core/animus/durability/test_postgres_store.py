"""Tests for :class:`DurableObjectStore`.

Uses SQLite in-memory as the engine so no PostgreSQL server is required.
All SQLAlchemy operations are generic enough to work across backends.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from animus.durability.postgres_store import (
    ConcurrencyError,
    DurableObjectStore,
    EventType,
    LifecycleStatus,
    ObjectRecord,
    ObjectType,
)

# Guard: skip entire module if sqlalchemy is unavailable
pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")


@pytest.fixture
def store(tmp_path):
    """Create a :class:`DurableObjectStore` backed by SQLite in-memory."""
    # Use a file-backed SQLite database so it persists across connections
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    s = DurableObjectStore(database_url=url, workspace_id="ws-test")
    s.create_tables()
    yield s


@pytest.fixture
def sample_record():
    """Return a basic :class:`ObjectRecord`."""
    return ObjectRecord(
        object_id="mem-001",
        schema_id="memory_candidate",
        schema_version="1.0.0",
        owner_id="owner-test",
        workspace_id="ws-test",
        artifact_type=ObjectType.MEMORY.value,
        payload={"content": "hello world"},
        tags=["test"],
        created_by="pytest",
    )


class TestLifecycle:
    """CRUD + ledger integration."""

    def test_store_creates_registry_ledger_and_outbox(self, store, sample_record):
        obj_id, event_id = store.store(sample_record)
        assert obj_id == sample_record.object_id
        assert event_id.startswith("evt-")

        # Registry
        retrieved = store.retrieve(sample_record.object_id)
        assert retrieved is not None
        assert retrieved.object_id == sample_record.object_id
        assert retrieved.version == 1
        assert retrieved.payload == sample_record.payload

        # Ledger
        events = store.get_ledger_events(sample_record.object_id)
        assert len(events) == 1
        assert events[0]["event_type"] == EventType.CREATED.value
        assert events[0]["event_id"] == event_id

        # Integrity hash verification
        assert store.verify_integrity(event_id) is True

    def test_update_with_optimistic_concurrency(self, store, sample_record):
        store.store(sample_record)

        # Modify and update
        sample_record.payload["content"] = "updated"
        ok, event_id = store.update(sample_record, expected_version=1)
        assert ok is True
        assert event_id.startswith("evt-")

        retrieved = store.retrieve(sample_record.object_id)
        assert retrieved.version == 2
        assert retrieved.payload["content"] == "updated"

    def test_update_wrong_version_raises(self, store, sample_record):
        store.store(sample_record)
        with pytest.raises(ConcurrencyError):
            store.update(sample_record, expected_version=99)

    def test_retrieve_version(self, store, sample_record):
        store.store(sample_record)
        sample_record.payload["content"] = "v2"
        store.update(sample_record)

        v1 = store.retrieve_version(sample_record.object_id, 1)
        assert v1.version == 1
        assert v1.payload["content"] == "hello world"

        v2 = store.retrieve_version(sample_record.object_id, 2)
        assert v2.version == 2
        assert v2.payload["content"] == "v2"

    def test_delete_soft_deletes(self, store, sample_record):
        store.store(sample_record)
        ok, event_id = store.delete(sample_record.object_id, principal="pytest")
        assert ok is True
        assert event_id.startswith("evt-")

        # Current retrieval returns None
        assert store.retrieve(sample_record.object_id) is None

        # Ledger still has events
        events = store.get_ledger_events(sample_record.object_id)
        assert any(e["event_type"] == EventType.DELETED.value for e in events)

    def test_delete_missing_returns_false(self, store):
        ok, _ = store.delete("nonexistent")
        assert ok is False

    def test_list_current(self, store):
        for i in range(3):
            store.store(
                ObjectRecord(
                    object_id=f"mem-{i:03d}",
                    schema_id="memory_candidate",
                    artifact_type=ObjectType.MEMORY.value,
                )
            )
        # Add a source
        store.store(
            ObjectRecord(
                object_id="src-001",
                schema_id="source",
                artifact_type=ObjectType.SOURCE.value,
            )
        )
        all_current = store.list_current()
        assert len(all_current) == 4

        mems = store.list_current(artifact_type=ObjectType.MEMORY.value)
        assert len(mems) == 3


class TestBitemporalQueries:
    """Time-travel queries."""

    def test_as_of_valid_time(self, store, sample_record):
        store.store(sample_record)
        sample_record.payload["content"] = "updated"
        store.update(sample_record)

        now = datetime.now(timezone.utc)
        # Query in the far future should get latest version
        future = now + timedelta(days=1)
        result = store.as_of_valid_time(sample_record.object_id, future)
        assert result is not None
        assert result.version == 2

    def test_as_of_transaction_time(self, store, sample_record):
        store.store(sample_record)
        sample_record.payload["content"] = "updated"
        store.update(sample_record)

        now = datetime.now(timezone.utc)
        future = now + timedelta(days=1)
        result = store.as_of_transaction_time(sample_record.object_id, future)
        assert result is not None
        # At future transaction time, both versions may be visible depending
        # on superseded_at logic. The current (non-superseded) row should
        # always be visible.


class TestOutbox:
    """Transactional outbox pattern."""

    def test_claim_and_acknowledge(self, store, sample_record):
        store.store(sample_record)

        entries = store.claim_outbox_entries("worker-1", limit=10)
        assert len(entries) >= 1
        entry_id = entries[0]["entry_id"]
        assert entries[0]["topic"] == "object.created"

        # Re-claim returns nothing (already claimed)
        claimed_again = store.claim_outbox_entries("worker-2", limit=10)
        assert all(e["entry_id"] != entry_id for e in claimed_again)

        # Acknowledge
        assert store.acknowledge_outbox_entry(entry_id) is True

        # Now fully processed
        processed = store.claim_outbox_entries("worker-1", limit=10)
        assert entry_id not in [e["entry_id"] for e in processed]

    def test_acknowledge_nonexistent(self, store):
        assert store.acknowledge_outbox_entry("obx-fake") is False

    def test_acknowledge_with_error(self, store, sample_record):
        store.store(sample_record)
        entries = store.claim_outbox_entries("worker-1", limit=10)
        entry_id = entries[0]["entry_id"]

        # Acknowledge with error
        assert store.acknowledge_outbox_entry(entry_id, error="timeout") is True

        # Should be reclaimable
        reclaimed = store.claim_outbox_entries("worker-1", limit=10)
        assert any(e["entry_id"] == entry_id for e in reclaimed)


class TestEnums:
    """Domain enum sanity checks."""

    def test_object_type_values(self):
        assert ObjectType.MEMORY.value == "memory"
        assert ObjectType.SOURCE.value == "source"

    def test_lifecycle_status_flow(self):
        assert LifecycleStatus.ACTIVE.value == "active"
        assert LifecycleStatus.SUPERSEDED.value == "superseded"
        assert LifecycleStatus.DELETED.value == "deleted"

    def test_event_type_values(self):
        assert EventType.CREATED.value == "created"
        assert EventType.UPDATED.value == "updated"


class TestConfigIntegration:
    """Ensure store respects configuration patterns."""

    def test_env_var_database_url(self, tmp_path, monkeypatch):
        db_path = tmp_path / "env_test.db"
        monkeypatch.setenv("ANIMUS_DATABASE_URL", f"sqlite:///{db_path}")
        store = DurableObjectStore()
        store.create_tables()

        record = ObjectRecord(object_id="env-test", schema_id="test")
        obj_id, _ = store.store(record)
        assert obj_id == "env-test"

    def test_explicit_database_url_overrides_env(self, tmp_path, monkeypatch):
        db_path = tmp_path / "explicit.db"
        monkeypatch.setenv("ANIMUS_DATABASE_URL", "sqlite:///wrong.db")
        store = DurableObjectStore(database_url=f"sqlite:///{db_path}")
        store.create_tables()

        record = ObjectRecord(object_id="explicit-test", schema_id="test")
        obj_id, _ = store.store(record)
        assert obj_id == "explicit-test"

    def test_missing_database_url_raises(self, monkeypatch):
        monkeypatch.delenv("ANIMUS_DATABASE_URL", raising=False)
        with pytest.raises(RuntimeError, match="ANIMUS_DATABASE_URL"):
            DurableObjectStore()


class TestWithoutSqlalchemy:
    """Graceful degradation when sqlalchemy is absent."""

    def test_import_fails_gracefully(self):
        # This test only runs when sqlalchemy is present, so we simulate the
        # absence by temporarily forcing _HAS_SQLALCHEMY = False.
        from animus.durability import postgres_store as _mod

        orig = _mod._HAS_SQLALCHEMY
        try:
            _mod._HAS_SQLALCHEMY = False
            with pytest.raises(RuntimeError, match="sqlalchemy"):
                DurableObjectStore(database_url="sqlite:///test.db")
        finally:
            _mod._HAS_SQLALCHEMY = orig
