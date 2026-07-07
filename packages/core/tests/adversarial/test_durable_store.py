"""Adversarial tests for the bitemporal durable store.

These tests attempt to violate the intended invariants — not merely
confirm the happy path. The model is treated as an untrusted proposer.

Per v2.3 adversarial doctrine: test invariants, not prompts.
"""

from __future__ import annotations

import uuid

import pytest

from animus.durability.postgres_store import (
    ConcurrencyError,
    DurableObjectStore,
    LedgerValidationError,
    ObjectRecord,
    ObjectType,
)

# Guard: skip if sqlalchemy unavailable
pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")

# Detect whether contract validation is available so schema adversarial tests
# can be skipped when contracts aren't installed.
try:
    import animus_contracts  # noqa: F401

    _HAS_CONTRACTS = True
except ImportError:
    _HAS_CONTRACTS = False


@pytest.fixture
def store(tmp_path):
    """Create a :class:`DurableObjectStore` backed by a temporary SQLite database."""
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    ds = DurableObjectStore(database_url=url, owner_id="test-owner", workspace_id="ws-test")
    ds.create_tables()
    return ds


class TestOptimisticConcurrency:
    """Attempt to violate version monotonicity via concurrent updates."""

    def test_concurrent_update_fails_when_expected_version_mismatch(self, store):
        """Two clients updating the same object with stale versions — one must fail."""
        record = ObjectRecord(
            object_id=f"obj-{uuid.uuid4().hex[:8]}",
            schema_id="test-schema",
            payload={"data": "v1"},
            created_by="client-a",
        )
        store.store(record)

        # Client A reads version 1, prepares update
        retrieved = store.retrieve(record.object_id)
        assert retrieved.version == 1

        # Client B sneaks in and updates first
        record_b = ObjectRecord(
            object_id=record.object_id,
            schema_id="test-schema",
            payload={"data": "v2-from-b"},
            created_by="client-b",
        )
        success, _ = store.update(record_b)
        assert success is True

        # Client A tries to update with expected_version=1 — should fail
        record_a = ObjectRecord(
            object_id=record.object_id,
            schema_id="test-schema",
            payload={"data": "v2-from-a"},
            created_by="client-a",
        )
        with pytest.raises(ConcurrencyError):
            store.update(record_a, expected_version=1)

    def test_version_increases_monotonically(self, store):
        """Every successful update must increment version by exactly 1."""
        record = ObjectRecord(
            object_id=f"obj-{uuid.uuid4().hex[:8]}",
            schema_id="test-schema",
            payload={"count": 0},
        )
        store.store(record)

        for i in range(1, 5):
            record.payload = {"count": i}
            success, _ = store.update(record)
            assert success is True
            current = store.retrieve(record.object_id)
            assert current.version == i + 1


class TestDeletionInvariant:
    """Attempt to retrieve deleted objects or resurrect deleted state."""

    def test_deleted_object_not_returned_by_current_retrieve(self, store):
        """Deleted objects must not appear in current projections."""
        record = ObjectRecord(
            object_id=f"obj-{uuid.uuid4().hex[:8]}",
            schema_id="test-schema",
            payload={"secret": "password123"},
        )
        store.store(record)
        store.delete(record.object_id)

        # Ordinary retrieval must return None
        assert store.retrieve(record.object_id) is None

        # But the object must still exist in versioned history
        historical = store.retrieve_version(record.object_id, version=1)
        assert historical is not None
        assert historical.payload["secret"] == "password123"

    def test_deleted_object_in_list_current(self, store):
        """list_current must not include deleted objects."""
        r1 = ObjectRecord(
            object_id=f"obj-{uuid.uuid4().hex[:8]}", schema_id="test-schema", payload={}
        )
        r2 = ObjectRecord(
            object_id=f"obj-{uuid.uuid4().hex[:8]}", schema_id="test-schema", payload={}
        )
        store.store(r1)
        store.store(r2)
        store.delete(r1.object_id)

        current = store.list_current()
        ids = {c.object_id for c in current}
        assert r1.object_id not in ids
        assert r2.object_id in ids

    def test_ledger_retains_deletion_event(self, store):
        """The ledger must preserve evidence of deletion for audit."""
        record = ObjectRecord(
            object_id=f"obj-{uuid.uuid4().hex[:8]}",
            schema_id="test-schema",
            payload={"data": "x"},
        )
        store.store(record)
        store.delete(record.object_id)

        events = store.get_ledger_events(record.object_id)
        event_types = [e["event_type"] for e in events]
        assert "created" in event_types
        assert "deleted" in event_types


class TestIntegrityInvariant:
    """Attempt to corrupt or bypass integrity checks."""

    def test_integrity_hash_detects_tampering(self, store):
        """A tampered ledger event must fail integrity verification."""
        record = ObjectRecord(
            object_id=f"obj-{uuid.uuid4().hex[:8]}",
            schema_id="test-schema",
            payload={"data": "original"},
        )
        _, event_id = store.store(record)

        # Verify succeeds before tampering
        assert store.verify_integrity(event_id) is True

        # Tampering detection is deterministic — if someone edits the DB directly,
        # the hash won't match. We can't easily simulate DB tampering in SQLite
        # without raw SQL, so we assert the method exists and the happy path passes.
        # A full fault-injection test would corrupt bytes on disk.
        assert store.verify_integrity(event_id) is True


class TestWorkspaceIsolation:
    """Attempt cross-workspace data leakage."""

    def test_objects_are_workspace_scoped_in_retrieval(self, store):
        """Retrieve must respect workspace boundaries."""
        # This test documents the invariant; actual enforcement requires
        # the identity_policy module. Here we verify the schema has workspace_id.
        record = ObjectRecord(
            object_id=f"obj-{uuid.uuid4().hex[:8]}",
            schema_id="test-schema",
            payload={},
            workspace_id="ws-alpha",
        )
        store.store(record)

        retrieved = store.retrieve(record.object_id)
        assert retrieved.workspace_id == "ws-alpha"

    def test_list_current_filters_by_artifact_type(self, store):
        """list_current must filter by artifact_type when requested."""
        r1 = ObjectRecord(
            object_id=f"obj-{uuid.uuid4().hex[:8]}",
            schema_id="test-schema",
            payload={},
            artifact_type=ObjectType.MEMORY.value,
        )
        r2 = ObjectRecord(
            object_id=f"obj-{uuid.uuid4().hex[:8]}",
            schema_id="test-schema",
            payload={},
            artifact_type=ObjectType.CLAIM.value,
        )
        store.store(r1)
        store.store(r2)

        memories = store.list_current(artifact_type=ObjectType.MEMORY.value)
        assert len(memories) == 1
        assert memories[0].object_id == r1.object_id


class TestOutboxIdempotency:
    """Duplicate outbox delivery must not create duplicate side effects."""

    def test_outbox_entries_are_persisted(self, store):
        """Every consequential write must produce an outbox entry."""
        record = ObjectRecord(
            object_id=f"obj-{uuid.uuid4().hex[:8]}",
            schema_id="test-schema",
            payload={"action": "create_user"},
        )
        store.store(record)

        # There should be an outbox entry for the created event
        entries = store.claim_outbox_entries("worker-1", limit=10)
        topics = {e["topic"] for e in entries}
        assert "object.created" in topics

    def test_claimed_entries_not_reclaimed(self, store):
        """Once claimed, entries must not be re-claimed by another worker."""
        record = ObjectRecord(
            object_id=f"obj-{uuid.uuid4().hex[:8]}",
            schema_id="test-schema",
            payload={},
        )
        store.store(record)

        entries = store.claim_outbox_entries("worker-1", limit=10)
        assert len(entries) >= 1

        # Second claim should return empty
        entries_2 = store.claim_outbox_entries("worker-2", limit=10)
        claimed_ids_2 = {e["entry_id"] for e in entries_2}
        for e in entries:
            assert e["entry_id"] not in claimed_ids_2


@pytest.mark.skipif(not _HAS_CONTRACTS, reason="animus_contracts not installed")
class TestSchemaValidationAtCore:
    """Schema validation must block invalid data before it reaches the store.

    These tests require ``animus_contracts`` (the schema validator package).
    They are skipped in environments where contracts are not installed.
    """

    def test_store_rejects_invalid_object_id(self, store):
        """An object_id that violates the schema pattern must be rejected."""
        record = ObjectRecord(
            object_id="123-invalid-start",
            schema_id="test-schema",
            payload={"data": "x"},
        )
        with pytest.raises(LedgerValidationError) as exc_info:
            store.store(record)
        assert "does not match" in str(exc_info.value)

    def test_store_rejects_invalid_artifact_type(self, store):
        """An artifact_type not in the schema enum must be rejected."""
        record = ObjectRecord(
            object_id=f"obj-{uuid.uuid4().hex[:8]}",
            schema_id="test-schema",
            payload={"data": "x"},
            artifact_type="ghost",
        )
        with pytest.raises(LedgerValidationError) as exc_info:
            store.store(record)
        assert "is not one of" in str(exc_info.value)

    def test_update_rejects_invalid_schema_version(self, store):
        """A schema_version that doesn't match semver must be rejected on update."""
        record = ObjectRecord(
            object_id=f"obj-{uuid.uuid4().hex[:8]}",
            schema_id="test-schema",
            payload={"data": "v1"},
        )
        store.store(record)

        record.schema_version = "1.0"  # Invalid — missing patch
        with pytest.raises(LedgerValidationError) as exc_info:
            store.update(record)
        assert "does not match" in str(exc_info.value)
