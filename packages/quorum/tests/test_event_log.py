"""Tests for the coordination event log."""

from __future__ import annotations

from convergent.event_log import CoordinationEvent, EventLog, EventType, event_timeline


class TestEventLog:
    """Tests for EventLog CRUD operations."""

    def test_record_and_query(self) -> None:
        log = EventLog(":memory:")
        event = log.record(
            EventType.INTENT_PUBLISHED,
            agent_id="a1",
            payload={"intent": "build_auth"},
        )
        assert event.event_type == EventType.INTENT_PUBLISHED
        assert event.agent_id == "a1"
        assert event.payload == {"intent": "build_auth"}
        assert event.event_id  # non-empty UUID

        events = log.query()
        assert len(events) == 1
        assert events[0].event_id == event.event_id

    def test_record_with_correlation_id(self) -> None:
        log = EventLog(":memory:")
        log.record(EventType.VOTE_CAST, agent_id="a1", correlation_id="task-42")
        log.record(EventType.DECISION_MADE, agent_id="system", correlation_id="task-42")
        log.record(EventType.MARKER_LEFT, agent_id="a2", correlation_id="task-99")

        events = log.query(correlation_id="task-42")
        assert len(events) == 2
        assert all(e.correlation_id == "task-42" for e in events)

    def test_record_default_payload(self) -> None:
        log = EventLog(":memory:")
        event = log.record(EventType.SIGNAL_SENT, agent_id="a1")
        assert event.payload == {}

    def test_record_custom_timestamp(self) -> None:
        log = EventLog(":memory:")
        ts = "2025-06-15T12:00:00Z"
        event = log.record(EventType.SCORE_UPDATED, agent_id="a1", timestamp=ts)
        assert event.timestamp == ts

    def test_query_by_type(self) -> None:
        log = EventLog(":memory:")
        log.record(EventType.INTENT_PUBLISHED, agent_id="a1")
        log.record(EventType.VOTE_CAST, agent_id="a1")
        log.record(EventType.INTENT_PUBLISHED, agent_id="a2")

        events = log.query(event_type=EventType.INTENT_PUBLISHED)
        assert len(events) == 2
        assert all(e.event_type == EventType.INTENT_PUBLISHED for e in events)

    def test_query_by_agent(self) -> None:
        log = EventLog(":memory:")
        log.record(EventType.MARKER_LEFT, agent_id="a1")
        log.record(EventType.MARKER_LEFT, agent_id="a2")
        log.record(EventType.SCORE_UPDATED, agent_id="a1")

        events = log.query(agent_id="a1")
        assert len(events) == 2
        assert all(e.agent_id == "a1" for e in events)

    def test_query_time_range(self) -> None:
        log = EventLog(":memory:")
        log.record(EventType.VOTE_CAST, agent_id="a1", timestamp="2025-01-01T00:00:00Z")
        log.record(EventType.VOTE_CAST, agent_id="a1", timestamp="2025-06-15T00:00:00Z")
        log.record(EventType.VOTE_CAST, agent_id="a1", timestamp="2025-12-31T00:00:00Z")

        events = log.query(since="2025-06-01T00:00:00Z", until="2025-07-01T00:00:00Z")
        assert len(events) == 1
        assert events[0].timestamp == "2025-06-15T00:00:00Z"

    def test_query_limit(self) -> None:
        log = EventLog(":memory:")
        for i in range(10):
            log.record(EventType.SIGNAL_SENT, agent_id=f"a{i}")

        events = log.query(limit=3)
        assert len(events) == 3

    def test_query_combined_filters(self) -> None:
        log = EventLog(":memory:")
        log.record(EventType.VOTE_CAST, agent_id="a1", correlation_id="t1")
        log.record(EventType.VOTE_CAST, agent_id="a2", correlation_id="t1")
        log.record(EventType.DECISION_MADE, agent_id="a1", correlation_id="t1")

        events = log.query(event_type=EventType.VOTE_CAST, agent_id="a1")
        assert len(events) == 1

    def test_query_order_ascending(self) -> None:
        log = EventLog(":memory:")
        log.record(EventType.VOTE_CAST, agent_id="a1", timestamp="2025-01-03T00:00:00Z")
        log.record(EventType.VOTE_CAST, agent_id="a1", timestamp="2025-01-01T00:00:00Z")
        log.record(EventType.VOTE_CAST, agent_id="a1", timestamp="2025-01-02T00:00:00Z")

        events = log.query()
        assert events[0].timestamp < events[1].timestamp < events[2].timestamp


class TestEventLogCount:
    """Tests for event counting."""

    def test_count_all(self) -> None:
        log = EventLog(":memory:")
        assert log.count() == 0
        log.record(EventType.VOTE_CAST, agent_id="a1")
        log.record(EventType.MARKER_LEFT, agent_id="a2")
        assert log.count() == 2

    def test_count_by_type(self) -> None:
        log = EventLog(":memory:")
        log.record(EventType.VOTE_CAST, agent_id="a1")
        log.record(EventType.VOTE_CAST, agent_id="a2")
        log.record(EventType.MARKER_LEFT, agent_id="a1")

        assert log.count(EventType.VOTE_CAST) == 2
        assert log.count(EventType.MARKER_LEFT) == 1
        assert log.count(EventType.DECISION_MADE) == 0


class TestEventLogClose:
    """Tests for lifecycle management."""

    def test_close(self) -> None:
        log = EventLog(":memory:")
        log.record(EventType.VOTE_CAST, agent_id="a1")
        log.close()
        # After close, queries should fail
        import sqlite3

        try:
            log.query()
            raise AssertionError("Should have raised after close")
        except sqlite3.ProgrammingError:
            pass


class TestCoordinationEvent:
    """Tests for the CoordinationEvent dataclass."""

    def test_frozen(self) -> None:
        event = CoordinationEvent(
            event_id="e1",
            event_type=EventType.VOTE_CAST,
            agent_id="a1",
            timestamp="2025-01-01T00:00:00Z",
            payload={"key": "value"},
        )
        assert event.event_id == "e1"
        assert event.correlation_id is None


class TestEventType:
    """Tests for EventType enum."""

    def test_all_types(self) -> None:
        assert len(EventType) == 10

    def test_roundtrip(self) -> None:
        for t in EventType:
            assert EventType(t.value) == t


class TestEventTimeline:
    """Tests for the timeline renderer."""

    def test_empty_timeline(self) -> None:
        assert event_timeline([]) == "(no events)"

    def test_basic_timeline(self) -> None:
        log = EventLog(":memory:")
        log.record(EventType.INTENT_PUBLISHED, agent_id="a1", timestamp="2025-01-01T00:00:00Z")
        log.record(EventType.VOTE_CAST, agent_id="a2", timestamp="2025-01-01T01:00:00Z")

        events = log.query()
        output = event_timeline(events)

        assert "Coordination Event Timeline" in output
        assert "intent_published" in output
        assert "vote_cast" in output
        assert "a1" in output
        assert "a2" in output
        assert "Total: 2 events" in output

    def test_timeline_with_correlation(self) -> None:
        log = EventLog(":memory:")
        log.record(
            EventType.DECISION_MADE,
            agent_id="system",
            correlation_id="task-42",
            timestamp="2025-01-01T00:00:00Z",
        )

        events = log.query()
        output = event_timeline(events)
        assert "[task-42]" in output

    def test_timeline_with_payload(self) -> None:
        log = EventLog(":memory:")
        log.record(
            EventType.SCORE_UPDATED,
            agent_id="a1",
            payload={"score": 0.85},
            timestamp="2025-01-01T00:00:00Z",
        )

        events = log.query()
        output = event_timeline(events)
        assert "0.85" in output

    def test_timeline_truncates_microseconds(self) -> None:
        log = EventLog(":memory:")
        log.record(
            EventType.SIGNAL_SENT,
            agent_id="a1",
            timestamp="2025-01-01T12:30:45.123456Z",
        )

        events = log.query()
        output = event_timeline(events)
        # Should truncate to seconds
        assert "12:30:45Z" in output
        assert "123456" not in output
