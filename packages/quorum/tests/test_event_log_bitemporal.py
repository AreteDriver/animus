"""Tests for EventLog bitemporal-lite (valid_from + recorded_at)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from animus_quorum.event_log import EventLog, EventType


def _iso(year: int, month: int, day: int) -> str:
    return datetime(year, month, day, tzinfo=timezone.utc).isoformat()


class TestRecordDefaults:
    def test_recorded_at_defaults_to_now(self) -> None:
        log = EventLog(":memory:")
        before = datetime.now(timezone.utc).isoformat()
        event = log.record(EventType.INTENT_PUBLISHED, "agent-1")
        after = datetime.now(timezone.utc).isoformat()
        assert event.recorded_at is not None
        assert before <= event.recorded_at <= after

    def test_valid_from_defaults_to_recorded_at(self) -> None:
        log = EventLog(":memory:")
        event = log.record(EventType.INTENT_PUBLISHED, "agent-1")
        assert event.valid_from == event.recorded_at

    def test_timestamp_aliases_recorded_at_on_new_writes(self) -> None:
        log = EventLog(":memory:")
        event = log.record(EventType.INTENT_PUBLISHED, "agent-1")
        assert event.timestamp == event.recorded_at

    def test_explicit_valid_from_persisted(self) -> None:
        log = EventLog(":memory:")
        event = log.record(
            EventType.INTENT_PUBLISHED,
            "agent-1",
            valid_from=_iso(2025, 1, 1),
        )
        assert event.valid_from == _iso(2025, 1, 1)
        # recorded_at should still be roughly now
        assert event.recorded_at != _iso(2025, 1, 1)

    def test_explicit_recorded_at_persisted(self) -> None:
        log = EventLog(":memory:")
        event = log.record(
            EventType.INTENT_PUBLISHED,
            "agent-1",
            recorded_at=_iso(2025, 6, 15),
        )
        assert event.recorded_at == _iso(2025, 6, 15)
        assert event.valid_from == _iso(2025, 6, 15)

    def test_legacy_timestamp_still_works(self) -> None:
        log = EventLog(":memory:")
        event = log.record(
            EventType.INTENT_PUBLISHED,
            "agent-1",
            timestamp=_iso(2024, 12, 1),
        )
        assert event.timestamp == _iso(2024, 12, 1)
        assert event.recorded_at == _iso(2024, 12, 1)
        assert event.valid_from == _iso(2024, 12, 1)


class TestQueryByValidFrom:
    def test_query_by_valid_from_range(self) -> None:
        log = EventLog(":memory:")
        log.record(
            EventType.INTENT_PUBLISHED,
            "agent-1",
            valid_from=_iso(2024, 1, 1),
            recorded_at=_iso(2026, 1, 1),
        )
        log.record(
            EventType.INTENT_PUBLISHED,
            "agent-2",
            valid_from=_iso(2025, 6, 1),
            recorded_at=_iso(2026, 1, 1),
        )
        log.record(
            EventType.INTENT_PUBLISHED,
            "agent-3",
            valid_from=_iso(2026, 3, 1),
            recorded_at=_iso(2026, 3, 1),
        )
        in_2025 = log.query(
            valid_from_since=_iso(2025, 1, 1),
            valid_from_until=_iso(2025, 12, 31),
        )
        agents = [e.agent_id for e in in_2025]
        assert agents == ["agent-2"]

    def test_query_by_recorded_at_back_compat(self) -> None:
        log = EventLog(":memory:")
        log.record(
            EventType.INTENT_PUBLISHED,
            "old",
            valid_from=_iso(2024, 1, 1),
            recorded_at=_iso(2024, 1, 1),
        )
        log.record(
            EventType.INTENT_PUBLISHED,
            "recent",
            recorded_at=_iso(2026, 5, 10),
        )
        recent_only = log.query(since=_iso(2026, 1, 1))
        agents = [e.agent_id for e in recent_only]
        assert agents == ["recent"]

    def test_query_combined_world_and_record_time(self) -> None:
        log = EventLog(":memory:")
        # valid in 2024, observed in 2026
        log.record(
            EventType.INTENT_PUBLISHED,
            "backfilled",
            valid_from=_iso(2024, 5, 1),
            recorded_at=_iso(2026, 5, 1),
        )
        # valid + observed both in 2024
        log.record(
            EventType.INTENT_PUBLISHED,
            "live-2024",
            valid_from=_iso(2024, 5, 1),
            recorded_at=_iso(2024, 5, 1),
        )
        # Want 2024 facts that we just observed in 2026
        results = log.query(
            valid_from_since=_iso(2024, 1, 1),
            valid_from_until=_iso(2024, 12, 31),
            since=_iso(2026, 1, 1),
        )
        agents = [e.agent_id for e in results]
        assert agents == ["backfilled"]


class TestMigration:
    def _make_legacy_db(self, db_path: Path) -> None:
        """Create a pre-bitemporal database row directly."""
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE coordination_events ("
            "event_id TEXT PRIMARY KEY, "
            "event_type TEXT NOT NULL, "
            "agent_id TEXT NOT NULL, "
            "timestamp TEXT NOT NULL, "
            "payload TEXT NOT NULL, "
            "correlation_id TEXT)"
        )
        conn.execute(
            "INSERT INTO coordination_events "
            "(event_id, event_type, agent_id, timestamp, "
            "payload, correlation_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "legacy-1",
                "intent_published",
                "agent-x",
                _iso(2024, 6, 1),
                json.dumps({"k": "v"}),
                "corr-x",
            ),
        )
        conn.commit()
        conn.close()

    def test_migration_idempotent_on_fresh_schema(self, tmp_path: Path) -> None:
        db = tmp_path / "fresh.db"
        log1 = EventLog(str(db))
        log1.close()
        # Reopen — _migrate runs again on existing schema, no error
        log2 = EventLog(str(db))
        log2.record(EventType.INTENT_PUBLISHED, "agent-1")
        events = log2.query()
        assert len(events) == 1
        log2.close()

    def test_migration_backfills_legacy_rows(self, tmp_path: Path) -> None:
        db = tmp_path / "legacy.db"
        self._make_legacy_db(db)
        log = EventLog(str(db))
        events = log.query()
        assert len(events) == 1
        ev = events[0]
        assert ev.event_id == "legacy-1"
        assert ev.timestamp == _iso(2024, 6, 1)
        # Backfill: both bitemporal columns should equal timestamp
        assert ev.valid_from == _iso(2024, 6, 1)
        assert ev.recorded_at == _iso(2024, 6, 1)
        log.close()

    def test_migration_does_not_overwrite_existing_columns(self, tmp_path: Path) -> None:
        db = tmp_path / "fresh.db"
        log = EventLog(str(db))
        # Record with explicit valid_from in the past
        log.record(
            EventType.INTENT_PUBLISHED,
            "agent-1",
            valid_from=_iso(2020, 1, 1),
            recorded_at=_iso(2026, 1, 1),
        )
        log.close()
        # Reopen — _migrate runs again; existing valid_from must
        # not be overwritten by the backfill UPDATE
        reopened = EventLog(str(db))
        events = reopened.query()
        assert events[0].valid_from == _iso(2020, 1, 1)
        assert events[0].recorded_at == _iso(2026, 1, 1)
        reopened.close()


class TestRowToEvent:
    def test_event_round_trip_through_query(self) -> None:
        log = EventLog(":memory:")
        original = log.record(
            EventType.VOTE_CAST,
            "voter-7",
            payload={"weight": 0.8},
            correlation_id="req-42",
            valid_from=_iso(2026, 4, 1),
        )
        retrieved = log.query(correlation_id="req-42")
        assert len(retrieved) == 1
        ev = retrieved[0]
        assert ev.event_id == original.event_id
        assert ev.event_type == EventType.VOTE_CAST
        assert ev.agent_id == "voter-7"
        assert ev.payload == {"weight": 0.8}
        assert ev.correlation_id == "req-42"
        assert ev.valid_from == _iso(2026, 4, 1)
        assert ev.recorded_at == original.recorded_at
