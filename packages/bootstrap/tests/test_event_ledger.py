"""Tests for the EventLedger operational event log."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from animus_bootstrap.intelligence.event_ledger import EventLedger


class TestRecordAndQuery:
    """Basic record/query operations."""

    def test_record_adds_event(self) -> None:
        ledger = EventLedger()
        ledger.record("tool_execution", "executor", {"tool": "test"})
        events = ledger.query()
        assert len(events) == 1
        assert events[0]["type"] == "tool_execution"
        assert events[0]["source"] == "executor"
        assert events[0]["payload"]["tool"] == "test"
        assert "timestamp" in events[0]

    def test_query_filters_by_type(self) -> None:
        ledger = EventLedger()
        ledger.record("tool_execution", "a")
        ledger.record("memory_recall", "b")
        ledger.record("tool_execution", "c")
        tool_events = ledger.query(event_type="tool_execution")
        assert len(tool_events) == 2
        assert all(e["type"] == "tool_execution" for e in tool_events)

    def test_query_filters_by_source(self) -> None:
        ledger = EventLedger()
        ledger.record("x", "source_a")
        ledger.record("x", "source_b")
        events = ledger.query(source="source_a")
        assert len(events) == 1
        assert events[0]["source"] == "source_a"

    def test_query_limit(self) -> None:
        ledger = EventLedger()
        for i in range(10):
            ledger.record("x", "s", {"i": i})
        events = ledger.query(limit=3)
        assert len(events) == 3
        # Newest first
        assert events[0]["payload"]["i"] == 9

    def test_query_returns_newest_first(self) -> None:
        ledger = EventLedger()
        ledger.record("a", "s")
        time.sleep(0.01)
        ledger.record("b", "s")
        events = ledger.query()
        assert events[0]["type"] == "b"
        assert events[1]["type"] == "a"


class TestRingBufferEviction:
    """In-memory ring buffer eviction."""

    def test_eviction_at_max(self) -> None:
        ledger = EventLedger(max_events=5)
        for i in range(7):
            ledger.record("x", "s", {"i": i})
        events = ledger.query()
        assert len(events) == 5
        # Oldest 2 evicted
        assert events[-1]["payload"]["i"] == 2
        assert events[0]["payload"]["i"] == 6

    def test_no_eviction_below_max(self) -> None:
        ledger = EventLedger(max_events=100)
        for i in range(10):
            ledger.record("x", "s", {"i": i})
        assert len(ledger.query()) == 10


class TestStats:
    """Aggregate statistics."""

    def test_empty_stats(self) -> None:
        stats = EventLedger().get_stats()
        assert stats["total"] == 0
        assert stats["by_type"] == {}
        assert stats["last_event_time"] is None
        assert stats["events_per_min"] == 0.0

    def test_stats_counts(self) -> None:
        ledger = EventLedger()
        ledger.record("tool_execution", "a")
        ledger.record("tool_execution", "a")
        ledger.record("memory_recall", "b")
        stats = ledger.get_stats()
        assert stats["total"] == 3
        assert stats["by_type"]["tool_execution"] == 2
        assert stats["by_type"]["memory_recall"] == 1
        assert stats["by_source"]["a"] == 2
        assert stats["by_source"]["b"] == 1
        assert stats["last_event_time"] is not None

    def test_events_per_min(self) -> None:
        ledger = EventLedger()
        for _ in range(12):
            ledger.record("x", "s")
        stats = ledger.get_stats()
        # All within 5 minutes
        assert stats["events_per_min"] > 0

    def test_recent_errors(self) -> None:
        ledger = EventLedger()
        ledger.record("error", "a", {"msg": "boom"})
        ledger.record("tool_execution", "b")
        errors = ledger.get_recent_errors(limit=5)
        assert len(errors) == 1
        assert errors[0]["type"] == "error"


class TestPersistence:
    """SQLite persistence across restarts."""

    def test_persistence_creates_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "events.db"
            ledger = EventLedger(db_path=db)
            ledger.record("tool_execution", "a", {"x": 1})
            assert db.exists()

    def test_reload_from_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "events.db"
            ledger1 = EventLedger(db_path=db)
            ledger1.record("tool_execution", "a", {"x": 1})
            ledger1.record("memory_recall", "b", {"y": 2})

            ledger2 = EventLedger(db_path=db)
            events = ledger2.query()
            assert len(events) == 2
            types = {e["type"] for e in events}
            assert types == {"tool_execution", "memory_recall"}

    def test_persistence_survives_ring_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "events.db"
            ledger1 = EventLedger(max_events=2, db_path=db)
            ledger1.record("a", "s", {"i": 1})
            ledger1.record("a", "s", {"i": 2})
            ledger1.record("a", "s", {"i": 3})
            # In-memory only has 2
            assert len(ledger1.query()) == 2
            # But DB has all 3
            ledger2 = EventLedger(max_events=10, db_path=db)
            assert len(ledger2.query()) == 3

    def test_persistence_no_crash_on_error(self) -> None:
        ledger = EventLedger(db_path=Path("/nonexistent/path/events.db"))
        # Should not raise
        ledger.record("x", "s")
        assert len(ledger.query()) == 1
