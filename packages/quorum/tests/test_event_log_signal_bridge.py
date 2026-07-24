"""Tests for EventLog → SignalBus bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from animus_quorum.event_log import EventLog, EventType
from animus_quorum.protocol import Signal
from animus_quorum.signal_bus import SignalBus


class _RecordingBus:
    """Minimal SignalBus stand-in that records publish() calls.

    Avoids spinning up the file-system backend for each test.
    Implements only the surface EventLog.publish_signal touches.
    """

    def __init__(self, raise_on_publish: bool = False) -> None:
        self.published: list[Signal] = []
        self._raise = raise_on_publish

    def publish(self, signal: Signal) -> None:
        if self._raise:
            raise RuntimeError("simulated bus failure")
        self.published.append(signal)


class TestNoBus:
    def test_record_with_no_bus_does_not_publish(self) -> None:
        log = EventLog(":memory:")
        ev = log.record(EventType.INTENT_PUBLISHED, "agent-1")
        assert ev.event_id  # sanity: record still works


class TestPublishSignal:
    def test_record_publishes_one_signal(self) -> None:
        bus = _RecordingBus()
        log = EventLog(":memory:", signal_bus=bus)
        log.record(EventType.INTENT_PUBLISHED, "agent-1")
        assert len(bus.published) == 1

    def test_signal_type_format(self) -> None:
        bus = _RecordingBus()
        log = EventLog(":memory:", signal_bus=bus)
        log.record(EventType.MARKER_LEFT, "agent-1")
        log.record(EventType.VOTE_CAST, "agent-2")
        types = [s.signal_type for s in bus.published]
        assert types == ["event.marker_left", "event.vote_cast"]

    def test_signal_source_is_event_agent(self) -> None:
        bus = _RecordingBus()
        log = EventLog(":memory:", signal_bus=bus)
        log.record(EventType.INTENT_PUBLISHED, "agent-7")
        assert bus.published[0].source_agent == "agent-7"

    def test_signal_target_is_none_broadcast(self) -> None:
        bus = _RecordingBus()
        log = EventLog(":memory:", signal_bus=bus)
        log.record(EventType.INTENT_PUBLISHED, "agent-1")
        assert bus.published[0].target_agent is None

    def test_signal_payload_round_trips_event_fields(self) -> None:
        bus = _RecordingBus()
        log = EventLog(":memory:", signal_bus=bus)
        ev = log.record(
            EventType.SCORE_UPDATED,
            "agent-1",
            payload={"intent_id": "i-1", "new_score": 0.7},
            correlation_id="i-1",
        )
        envelope = json.loads(bus.published[0].payload)
        assert envelope["event_id"] == ev.event_id
        assert envelope["event_type"] == "score_updated"
        assert envelope["valid_from"] == ev.valid_from
        assert envelope["recorded_at"] == ev.recorded_at
        assert envelope["correlation_id"] == "i-1"
        assert envelope["payload"] == {
            "intent_id": "i-1",
            "new_score": 0.7,
        }

    def test_signal_timestamp_uses_recorded_at(self) -> None:
        bus = _RecordingBus()
        log = EventLog(":memory:", signal_bus=bus)
        ev = log.record(EventType.INTENT_PUBLISHED, "agent-1")
        assert bus.published[0].timestamp == ev.recorded_at


class TestBusFailureDoesNotBreakRecord:
    def test_publish_failure_swallowed(self) -> None:
        bus = _RecordingBus(raise_on_publish=True)
        log = EventLog(":memory:", signal_bus=bus)
        # Must not raise
        ev = log.record(EventType.INTENT_PUBLISHED, "agent-1")
        assert ev.event_id

    def test_record_persists_even_when_bus_raises(self) -> None:
        bus = _RecordingBus(raise_on_publish=True)
        log = EventLog(":memory:", signal_bus=bus)
        log.record(EventType.INTENT_PUBLISHED, "agent-1")
        # Verify the row landed in SQLite despite the bus failing
        events = log.query()
        assert len(events) == 1
        assert events[0].agent_id == "agent-1"

    def test_failure_logged_as_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        bus = _RecordingBus(raise_on_publish=True)
        log = EventLog(":memory:", signal_bus=bus)
        with caplog.at_level("WARNING"):
            log.record(EventType.INTENT_PUBLISHED, "agent-1")
        assert any("signal bridge failed" in rec.message for rec in caplog.records)


class TestRealSignalBus:
    """One end-to-end sanity check using the real filesystem bus."""

    def test_real_bus_receives_event_signal(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        received: list[Signal] = []

        def cb(sig: Signal) -> None:
            received.append(sig)

        bus.subscribe("event.intent_published", cb)
        log = EventLog(":memory:", signal_bus=bus)
        log.record(EventType.INTENT_PUBLISHED, "agent-1")
        # SignalBus dispatches via poll_once() (or its background
        # poll loop). Call it explicitly to flush.
        bus.poll_once()
        assert len(received) == 1
        assert received[0].signal_type == "event.intent_published"
