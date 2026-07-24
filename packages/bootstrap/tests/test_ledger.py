"""Tests for the Cognitive Event Ledger.

Covers: model validation, integrity chain computation, SQLite store operations,
atomicity, and chain verification.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from animus_bootstrap.ledger import (
    EventType,
    IntegrityChain,
    LedgerEntry,
    LedgerEvent,
    LedgerStore,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_event(
    event_id: str = "evt-test-001",
    event_type: EventType = EventType.created,
    object_id: str = "obj-test",
    object_version: int = 1,
    principal: str = "user-alice",
    workspace_id: str = "ws-test",
    payload: dict | None = None,
    integrity_hash: str | None = None,
    parent_event_id: str | None = None,
) -> LedgerEvent:
    """Build a minimal valid LedgerEvent."""
    payload = payload or {"msg": "hello"}
    return LedgerEvent(
        event_id=event_id,
        event_type=event_type,
        object_id=object_id,
        object_version=object_version,
        principal=principal,
        workspace_id=workspace_id,
        payload=payload,
        integrity_hash=integrity_hash or IntegrityChain.hash_payload(payload),
        parent_event_id=parent_event_id,
    )


@pytest.fixture()
def tmp_store() -> LedgerStore:
    """Provide a temporary LedgerStore backed by a fresh SQLite DB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Path(tmpdir) / "ledger.db"
        yield LedgerStore(db_path=db)


# ------------------------------------------------------------------
# Model Validation
# ------------------------------------------------------------------


class TestLedgerEventValidation:
    """Pydantic field constraints."""

    def test_minimal_valid_event(self) -> None:
        event = _make_event()
        assert event.event_id == "evt-test-001"
        assert event.event_type == EventType.created

    def test_event_id_pattern_rejects_bad_prefix(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(event_id="bad-id")

    def test_event_id_pattern_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(event_id="")

    def test_object_version_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(object_version=0)

    def test_principal_min_length(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(principal="ab")

    def test_workspace_id_pattern(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(workspace_id="bad-ws")

    def test_integrity_hash_must_be_64_hex(self) -> None:
        with pytest.raises(ValidationError):
            _make_event(integrity_hash="not-a-hash")

    def test_auto_tx_time(self) -> None:
        before = datetime.now(UTC)
        event = _make_event()
        after = datetime.now(UTC)
        assert before <= event.tx_time <= after

    def test_parent_event_id_optional(self) -> None:
        event = _make_event(parent_event_id=None)
        assert event.parent_event_id is None
        event2 = _make_event(parent_event_id="evt-parent-001")
        assert event2.parent_event_id == "evt-parent-001"

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            LedgerEvent(
                event_id="evt-test-001",
                event_type=EventType.created,
                object_id="obj-test",
                object_version=1,
                principal="user-alice",
                workspace_id="ws-test",
                payload={},
                integrity_hash="a" * 64,
                unexpected_field="boom",
            )


# ------------------------------------------------------------------
# Integrity Chain
# ------------------------------------------------------------------


class TestIntegrityChain:
    """Hash computation and verification."""

    def test_hash_payload_deterministic(self) -> None:
        p = {"a": 1, "b": 2}
        h1 = IntegrityChain.hash_payload(p)
        h2 = IntegrityChain.hash_payload(p)
        assert h1 == h2
        assert len(h1) == 64

    def test_hash_payload_different_for_different_data(self) -> None:
        h1 = IntegrityChain.hash_payload({"a": 1})
        h2 = IntegrityChain.hash_payload({"a": 2})
        assert h1 != h2

    def test_compute_chain_hash_uses_genesis_for_first(self) -> None:
        event = _make_event()
        chain = IntegrityChain.compute_chain_hash(event, None)
        expected = IntegrityChain.compute_chain_hash(event, IntegrityChain.GENESIS_HASH)
        assert chain == expected

    def test_chain_hash_changes_with_previous(self) -> None:
        event = _make_event()
        h1 = IntegrityChain.compute_chain_hash(event, None)
        h2 = IntegrityChain.compute_chain_hash(event, "a" * 64)
        assert h1 != h2

    def test_verify_single_entry(self) -> None:
        event = _make_event()
        chain = IntegrityChain.compute_chain_hash(event, None)
        entry = LedgerEntry(**event.model_dump(mode="json"), chain_hash=chain)
        assert IntegrityChain.verify([entry]) is True

    def test_verify_two_linked_entries(self) -> None:
        e1 = _make_event(event_id="evt-001")
        c1 = IntegrityChain.compute_chain_hash(e1, None)
        entry1 = LedgerEntry(**e1.model_dump(mode="json"), chain_hash=c1)

        e2 = _make_event(event_id="evt-002", parent_event_id="evt-001")
        c2 = IntegrityChain.compute_chain_hash(e2, c1)
        entry2 = LedgerEntry(**e2.model_dump(mode="json"), chain_hash=c2)

        assert IntegrityChain.verify([entry1, entry2]) is True

    def test_verify_detects_tampered_hash(self) -> None:
        event = _make_event()
        entry = LedgerEntry(**event.model_dump(mode="json"), chain_hash="0" * 64)
        assert IntegrityChain.verify([entry]) is False

    def test_verify_detects_tampered_middle(self) -> None:
        e1 = _make_event(event_id="evt-001")
        c1 = IntegrityChain.compute_chain_hash(e1, None)
        entry1 = LedgerEntry(**e1.model_dump(mode="json"), chain_hash=c1)

        e2 = _make_event(event_id="evt-002", parent_event_id="evt-001")
        c2 = IntegrityChain.compute_chain_hash(e2, c1)
        entry2 = LedgerEntry(**e2.model_dump(mode="json"), chain_hash=c2)

        # Tamper entry2's payload without updating hash
        entry2_tampered = entry2.model_copy(update={"payload": {"tampered": True}})
        assert IntegrityChain.verify([entry1, entry2_tampered]) is False

    def test_verify_empty_list(self) -> None:
        assert IntegrityChain.verify([]) is True


# ------------------------------------------------------------------
# LedgerStore — Append and Query
# ------------------------------------------------------------------


class TestLedgerStoreAppend:
    """Atomic append operations."""

    def test_append_returns_entry(self, tmp_store: LedgerStore) -> None:
        event = _make_event()
        entry = tmp_store.append(event)
        assert entry.db_id is not None
        assert entry.chain_hash is not None
        assert len(entry.chain_hash) == 64

    def test_append_computes_integrity_hash(self, tmp_store: LedgerStore) -> None:
        event = _make_event()
        entry = tmp_store.append(event)
        assert entry.chain_hash == IntegrityChain.compute_chain_hash(event, None)

    def test_append_sequence_links_hashes(self, tmp_store: LedgerStore) -> None:
        e1 = _make_event(event_id="evt-seq-001")
        entry1 = tmp_store.append(e1)

        e2 = _make_event(event_id="evt-seq-002", parent_event_id="evt-seq-001")
        entry2 = tmp_store.append(e2)

        expected = IntegrityChain.compute_chain_hash(e2, entry1.chain_hash)
        assert entry2.chain_hash == expected

    def test_append_duplicate_event_id_raises(self, tmp_store: LedgerStore) -> None:
        event = _make_event(event_id="evt-dup")
        tmp_store.append(event)
        with pytest.raises(sqlite3.IntegrityError):
            tmp_store.append(event)

    def test_count_increments(self, tmp_store: LedgerStore) -> None:
        assert tmp_store.count() == 0
        tmp_store.append(_make_event(event_id="evt-c-001"))
        assert tmp_store.count() == 1
        tmp_store.append(_make_event(event_id="evt-c-002"))
        assert tmp_store.count() == 2

    def test_get_last_event_id(self, tmp_store: LedgerStore) -> None:
        assert tmp_store.get_last_event_id() is None
        tmp_store.append(_make_event(event_id="evt-last-001"))
        assert tmp_store.get_last_event_id() == "evt-last-001"
        tmp_store.append(_make_event(event_id="evt-last-002"))
        assert tmp_store.get_last_event_id() == "evt-last-002"


class TestLedgerStoreQuery:
    """Filtering and retrieval."""

    def test_get_by_event_id(self, tmp_store: LedgerStore) -> None:
        event = _make_event(event_id="evt-find")
        tmp_store.append(event)
        found = tmp_store.get_by_event_id("evt-find")
        assert found is not None
        assert found.event_id == "evt-find"

    def test_get_by_event_id_missing(self, tmp_store: LedgerStore) -> None:
        assert tmp_store.get_by_event_id("evt-missing") is None

    def test_query_by_object_id(self, tmp_store: LedgerStore) -> None:
        tmp_store.append(_make_event(event_id="evt-obj-001", object_id="obj-a"))
        tmp_store.append(_make_event(event_id="evt-obj-002", object_id="obj-a"))
        tmp_store.append(_make_event(event_id="evt-obj-003", object_id="obj-b"))
        results = tmp_store.query(object_id="obj-a")
        assert len(results) == 2
        assert all(e.object_id == "obj-a" for e in results)

    def test_query_by_event_type(self, tmp_store: LedgerStore) -> None:
        tmp_store.append(_make_event(event_id="evt-t-001", event_type=EventType.created))
        tmp_store.append(_make_event(event_id="evt-t-002", event_type=EventType.updated))
        results = tmp_store.query(event_type=EventType.created)
        assert len(results) == 1
        assert results[0].event_type == EventType.created

    def test_query_by_workspace(self, tmp_store: LedgerStore) -> None:
        tmp_store.append(_make_event(event_id="evt-w-001", workspace_id="ws-alpha"))
        tmp_store.append(_make_event(event_id="evt-w-002", workspace_id="ws-beta"))
        results = tmp_store.query(workspace_id="ws-alpha")
        assert len(results) == 1

    def test_query_limit_and_offset(self, tmp_store: LedgerStore) -> None:
        for i in range(5):
            tmp_store.append(_make_event(event_id=f"evt-lo-{i}"))
        results = tmp_store.query(limit=2)
        assert len(results) == 2
        # DESC order — newest first
        assert results[0].event_id == "evt-lo-4"
        assert results[1].event_id == "evt-lo-3"

    def test_query_combined_filters(self, tmp_store: LedgerStore) -> None:
        tmp_store.append(
            _make_event(
                event_id="evt-cf-001",
                object_id="obj-x",
                event_type=EventType.created,
                workspace_id="ws-prod",
            )
        )
        tmp_store.append(
            _make_event(
                event_id="evt-cf-002",
                object_id="obj-x",
                event_type=EventType.updated,
                workspace_id="ws-prod",
            )
        )
        results = tmp_store.query(
            object_id="obj-x",
            event_type=EventType.created,
            workspace_id="ws-prod",
        )
        assert len(results) == 1
        assert results[0].event_id == "evt-cf-001"

    def test_get_chain_chronological(self, tmp_store: LedgerStore) -> None:
        for i in range(3):
            tmp_store.append(_make_event(event_id=f"evt-ch-{i}", object_id="obj-chain"))
        chain = tmp_store.get_chain("obj-chain")
        assert len(chain) == 3
        # ASC order by id
        assert chain[0].event_id == "evt-ch-0"
        assert chain[1].event_id == "evt-ch-1"
        assert chain[2].event_id == "evt-ch-2"

    def test_count_by_object_id(self, tmp_store: LedgerStore) -> None:
        tmp_store.append(_make_event(event_id="evt-co-001", object_id="obj-1"))
        tmp_store.append(_make_event(event_id="evt-co-002", object_id="obj-1"))
        tmp_store.append(_make_event(event_id="evt-co-003", object_id="obj-2"))
        assert tmp_store.count(object_id="obj-1") == 2
        assert tmp_store.count(object_id="obj-2") == 1
        assert tmp_store.count() == 3


class TestLedgerStoreIntegrity:
    """Chain verification after persistence."""

    def test_verify_chain_empty(self, tmp_store: LedgerStore) -> None:
        assert tmp_store.verify_chain() is True

    def test_verify_chain_after_appends(self, tmp_store: LedgerStore) -> None:
        for i in range(3):
            tmp_store.append(_make_event(event_id=f"evt-v-{i}"))
        assert tmp_store.verify_chain() is True

    def test_verify_chain_by_object(self, tmp_store: LedgerStore) -> None:
        tmp_store.append(_make_event(event_id="evt-vo-001", object_id="obj-a"))
        tmp_store.append(_make_event(event_id="evt-vo-002", object_id="obj-b"))
        tmp_store.append(_make_event(event_id="evt-vo-003", object_id="obj-a"))
        assert tmp_store.verify_chain("obj-a") is True
        assert tmp_store.verify_chain("obj-b") is True

    def test_verify_chain_fails_on_tamper(self, tmp_store: LedgerStore) -> None:
        tmp_store.append(_make_event(event_id="evt-tamp-001"))
        # Directly tamper the DB
        import sqlite3
        with sqlite3.connect(str(tmp_store._db_path)) as conn:
            conn.execute(
                "UPDATE ledger_events SET payload = ? WHERE event_id = ?",
                (json.dumps({"tampered": True}), "evt-tamp-001"),
            )
            conn.commit()
        assert tmp_store.verify_chain() is False


class TestLedgerStoreReload:
    """Persistence survives store restart."""

    def test_reload_from_db(self, tmp_store: LedgerStore) -> None:
        tmp_store.append(_make_event(event_id="evt-rel-001"))
        tmp_store.append(_make_event(event_id="evt-rel-002"))

        store2 = LedgerStore(db_path=tmp_store._db_path)
        assert store2.count() == 2
        found = store2.get_by_event_id("evt-rel-001")
        assert found is not None
        assert found.event_id == "evt-rel-001"

    def test_reload_chain_still_verifies(self, tmp_store: LedgerStore) -> None:
        for i in range(4):
            tmp_store.append(_make_event(event_id=f"evt-rc-{i}"))
        store2 = LedgerStore(db_path=tmp_store._db_path)
        assert store2.verify_chain() is True


class TestLedgerStoreEdgeCases:
    """Boundary and error conditions."""

    def test_no_db_path_raises_on_append(self) -> None:
        store = LedgerStore(db_path=None)
        with pytest.raises(RuntimeError):
            store.append(_make_event())

    def test_no_db_path_returns_empty_queries(self) -> None:
        store = LedgerStore(db_path=None)
        assert store.query() == []
        assert store.count() == 0
        assert store.get_by_event_id("x") is None
        assert store.verify_chain() is True

    def test_large_payload(self, tmp_store: LedgerStore) -> None:
        payload = {"data": "x" * 100_000}
        event = _make_event(event_id="evt-big", payload=payload)
        entry = tmp_store.append(event)
        assert entry.integrity_hash == IntegrityChain.hash_payload(payload)

    def test_unicode_in_fields(self, tmp_store: LedgerStore) -> None:
        event = _make_event(
            event_id="evt-unicode-001",
            principal="ユーザー",
            workspace_id="ws-unicode-test",
            payload={"msg": "🚀"},
        )
        tmp_store.append(event)
        found = tmp_store.get_by_event_id("evt-unicode-001")
        assert found is not None
        assert found.principal == "ユーザー"
        assert found.payload["msg"] == "🚀"

    def test_all_event_types(self, tmp_store: LedgerStore) -> None:
        for i, et in enumerate(EventType):
            tmp_store.append(_make_event(event_id=f"evt-all-{i}", event_type=et))
        assert tmp_store.count() == len(EventType)
        for et in EventType:
            assert tmp_store.query(event_type=et)
