"""Tests for convergent.signal_bus."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from convergent.protocol import Signal
from convergent.signal_bus import SignalBus


def _signal(
    signal_type: str = "task_complete",
    source: str = "agent-1",
    target: str | None = None,
    payload: str = "",
) -> Signal:
    """Helper to create signals with sane defaults."""
    return Signal(
        signal_type=signal_type,
        source_agent=source,
        target_agent=target,
        payload=payload,
    )


class TestPublish:
    def test_publish_creates_file(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        bus.publish(_signal())
        files = list((tmp_path / "signals").glob("*.json"))
        assert len(files) == 1

    def test_publish_multiple_creates_multiple_files(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        bus.publish(_signal(source="agent-1"))
        bus.publish(_signal(source="agent-2"))
        files = list((tmp_path / "signals").glob("*.json"))
        assert len(files) == 2

    def test_published_file_is_valid_json(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        sig = _signal(payload='{"key": "value"}')
        bus.publish(sig)
        files = list((tmp_path / "signals").glob("*.json"))
        restored = Signal.from_json(files[0].read_text(encoding="utf-8"))
        assert restored.signal_type == sig.signal_type
        assert restored.source_agent == sig.source_agent
        assert restored.payload == sig.payload

    def test_publish_creates_directory(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "nested" / "signals")
        bus.publish(_signal())
        assert (tmp_path / "nested" / "signals").is_dir()


class TestSubscribeAndPoll:
    def test_subscribe_receives_signal(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        received: list[Signal] = []
        bus.subscribe("task_complete", received.append)
        bus.publish(_signal(signal_type="task_complete"))
        signals = bus.poll_once()
        assert len(signals) == 1
        assert len(received) == 1
        assert received[0].signal_type == "task_complete"

    def test_subscribe_ignores_other_types(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        received: list[Signal] = []
        bus.subscribe("blocked", received.append)
        bus.publish(_signal(signal_type="task_complete"))
        bus.poll_once()
        assert len(received) == 0

    def test_multiple_subscribers(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        received_a: list[Signal] = []
        received_b: list[Signal] = []
        bus.subscribe("conflict", received_a.append)
        bus.subscribe("conflict", received_b.append)
        bus.publish(_signal(signal_type="conflict"))
        bus.poll_once()
        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_poll_once_idempotent(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        received: list[Signal] = []
        bus.subscribe("task_complete", received.append)
        bus.publish(_signal())
        bus.poll_once()
        bus.poll_once()  # Second poll should not re-deliver
        assert len(received) == 1

    def test_poll_returns_new_signals(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        bus.publish(_signal(source="agent-1"))
        first = bus.poll_once()
        bus.publish(_signal(source="agent-2"))
        second = bus.poll_once()
        assert len(first) == 1
        assert len(second) == 1
        assert first[0].source_agent == "agent-1"
        assert second[0].source_agent == "agent-2"


class TestTargetedSignals:
    def test_targeted_signal_reaches_target(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        received: list[Signal] = []
        bus.subscribe("blocked", received.append, agent_id="agent-2")
        bus.publish(_signal(signal_type="blocked", source="agent-1", target="agent-2"))
        bus.poll_once()
        assert len(received) == 1

    def test_targeted_signal_skips_non_target(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        received: list[Signal] = []
        bus.subscribe("blocked", received.append, agent_id="agent-3")
        bus.publish(_signal(signal_type="blocked", source="agent-1", target="agent-2"))
        bus.poll_once()
        assert len(received) == 0

    def test_broadcast_reaches_all_subscribers(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        received_a: list[Signal] = []
        received_b: list[Signal] = []
        bus.subscribe("task_complete", received_a.append, agent_id="agent-1")
        bus.subscribe("task_complete", received_b.append, agent_id="agent-2")
        bus.publish(_signal(signal_type="task_complete", target=None))  # broadcast
        bus.poll_once()
        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_subscriber_without_agent_id_receives_targeted(self, tmp_path: Path) -> None:
        """A subscriber with no agent_id filter receives all signals."""
        bus = SignalBus(tmp_path / "signals")
        received: list[Signal] = []
        bus.subscribe("blocked", received.append)  # No agent_id filter
        bus.publish(_signal(signal_type="blocked", target="agent-2"))
        bus.poll_once()
        assert len(received) == 1


class TestUnsubscribe:
    def test_unsubscribe_removes_callback(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        received: list[Signal] = []
        cb = received.append  # Store reference — bound methods differ per access
        bus.subscribe("task_complete", cb)
        assert bus.unsubscribe("task_complete", cb) is True
        bus.publish(_signal())
        bus.poll_once()
        assert len(received) == 0

    def test_unsubscribe_nonexistent_returns_false(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        assert bus.unsubscribe("task_complete", lambda s: None) is False

    def test_unsubscribe_wrong_type_returns_false(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        received: list[Signal] = []
        bus.subscribe("blocked", received.append)
        assert bus.unsubscribe("task_complete", received.append) is False


class TestGetSignals:
    def test_get_all_signals(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        bus.publish(_signal(signal_type="blocked"))
        bus.publish(_signal(signal_type="task_complete"))
        signals = bus.get_signals()
        assert len(signals) == 2

    def test_get_signals_by_type(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        bus.publish(_signal(signal_type="blocked"))
        bus.publish(_signal(signal_type="task_complete"))
        signals = bus.get_signals(signal_type="blocked")
        assert len(signals) == 1
        assert signals[0].signal_type == "blocked"

    def test_get_signals_by_source(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        bus.publish(_signal(source="agent-1"))
        bus.publish(_signal(source="agent-2"))
        signals = bus.get_signals(source_agent="agent-1")
        assert len(signals) == 1
        assert signals[0].source_agent == "agent-1"

    def test_get_signals_since(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        old_signal = Signal(
            signal_type="blocked",
            source_agent="agent-1",
            timestamp=old_ts,
        )
        bus.publish(old_signal)
        bus.publish(_signal(signal_type="task_complete"))  # fresh
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        signals = bus.get_signals(since=cutoff)
        assert len(signals) == 1
        assert signals[0].signal_type == "task_complete"

    def test_get_signals_empty_directory(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        assert bus.get_signals() == []


class TestCleanup:
    def test_cleanup_expired_removes_old_signals(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        old_signal = Signal(
            signal_type="blocked",
            source_agent="agent-1",
            timestamp=old_ts,
        )
        bus.publish(old_signal)
        bus.publish(_signal())  # fresh
        deleted = bus.cleanup_expired(max_age_seconds=3600)
        assert deleted == 1
        remaining = bus.get_signals()
        assert len(remaining) == 1

    def test_cleanup_no_expired(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        bus.publish(_signal())
        deleted = bus.cleanup_expired(max_age_seconds=3600)
        assert deleted == 0

    def test_clear_removes_all(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        bus.publish(_signal(source="agent-1"))
        bus.publish(_signal(source="agent-2"))
        deleted = bus.clear()
        assert deleted == 2
        assert bus.get_signals() == []

    def test_clear_resets_processed(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        received: list[Signal] = []
        bus.subscribe("task_complete", received.append)
        bus.publish(_signal())
        bus.poll_once()
        assert len(received) == 1

        bus.clear()
        bus.publish(_signal())
        bus.poll_once()
        assert len(received) == 2  # Re-delivered after clear


class TestPolling:
    def test_start_and_stop_polling(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals", poll_interval=0.05)
        bus.start_polling()
        assert bus.is_polling is True
        bus.stop_polling()
        assert bus.is_polling is False

    def test_polling_dispatches_signals(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals", poll_interval=0.05)
        received: list[Signal] = []
        bus.subscribe("task_complete", received.append)
        bus.start_polling()
        try:
            bus.publish(_signal())
            # Give polling thread time to pick it up
            time.sleep(0.2)
            assert len(received) >= 1
        finally:
            bus.stop_polling()

    def test_double_start_raises(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals", poll_interval=0.05)
        bus.start_polling()
        try:
            with pytest.raises(RuntimeError, match="already active"):
                bus.start_polling()
        finally:
            bus.stop_polling()

    def test_stop_without_start(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")
        bus.stop_polling()  # Should not raise


class TestMalformedSignals:
    def test_malformed_file_skipped_on_poll(self, tmp_path: Path) -> None:
        signals_dir = tmp_path / "signals"
        signals_dir.mkdir()
        (signals_dir / "bad.json").write_text("not json", encoding="utf-8")
        bus = SignalBus(signals_dir)
        received: list[Signal] = []
        bus.subscribe("task_complete", received.append)
        signals = bus.poll_once()
        assert len(signals) == 0
        assert len(received) == 0

    def test_malformed_file_skipped_on_get(self, tmp_path: Path) -> None:
        signals_dir = tmp_path / "signals"
        signals_dir.mkdir()
        (signals_dir / "bad.json").write_text("{invalid", encoding="utf-8")
        bus = SignalBus(signals_dir)
        signals = bus.get_signals()
        assert len(signals) == 0


class TestSubscriberErrors:
    def test_callback_error_does_not_crash(self, tmp_path: Path) -> None:
        bus = SignalBus(tmp_path / "signals")

        def bad_callback(s: Signal) -> None:
            raise ValueError("oops")

        received: list[Signal] = []
        bus.subscribe("task_complete", bad_callback)
        bus.subscribe("task_complete", received.append)
        bus.publish(_signal())
        bus.poll_once()
        # Second subscriber still receives despite first erroring
        assert len(received) == 1


class TestSignalBusWithSQLiteBackend:
    """Test SignalBus wired to SQLiteSignalBackend."""

    def test_publish_and_poll(self) -> None:
        from convergent.sqlite_signal_backend import SQLiteSignalBackend

        backend = SQLiteSignalBackend(":memory:")
        bus = SignalBus(backend=backend)
        received: list[Signal] = []
        bus.subscribe("task_complete", received.append)
        bus.publish(_signal())
        signals = bus.poll_once()
        assert len(signals) == 1
        assert len(received) == 1
        backend.close()

    def test_poll_idempotent(self) -> None:
        from convergent.sqlite_signal_backend import SQLiteSignalBackend

        backend = SQLiteSignalBackend(":memory:")
        bus = SignalBus(backend=backend)
        received: list[Signal] = []
        bus.subscribe("task_complete", received.append)
        bus.publish(_signal())
        bus.poll_once()
        bus.poll_once()
        assert len(received) == 1
        backend.close()

    def test_get_signals_delegates(self) -> None:
        from convergent.sqlite_signal_backend import SQLiteSignalBackend

        backend = SQLiteSignalBackend(":memory:")
        bus = SignalBus(backend=backend)
        bus.publish(_signal(signal_type="blocked"))
        bus.publish(_signal(signal_type="task_complete"))
        signals = bus.get_signals(signal_type="blocked")
        assert len(signals) == 1
        backend.close()

    def test_cleanup_delegates(self) -> None:
        from convergent.sqlite_signal_backend import SQLiteSignalBackend

        backend = SQLiteSignalBackend(":memory:")
        bus = SignalBus(backend=backend)
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        bus.publish(Signal(signal_type="old", source_agent="a", timestamp=old_ts))
        bus.publish(_signal())
        deleted = bus.cleanup_expired(max_age_seconds=3600)
        assert deleted == 1
        backend.close()

    def test_clear_delegates(self) -> None:
        from convergent.sqlite_signal_backend import SQLiteSignalBackend

        backend = SQLiteSignalBackend(":memory:")
        bus = SignalBus(backend=backend)
        bus.publish(_signal())
        deleted = bus.clear()
        assert deleted == 1
        assert bus.get_signals() == []
        backend.close()

    def test_close_stops_polling_and_closes_backend(self) -> None:
        from convergent.sqlite_signal_backend import SQLiteSignalBackend

        backend = SQLiteSignalBackend(":memory:")
        bus = SignalBus(backend=backend, poll_interval=0.05)
        bus.start_polling()
        bus.close()
        assert not bus.is_polling


class TestMultiConsumer:
    """Test two SignalBus instances with different consumer_ids on same backend."""

    def test_independent_consumption(self) -> None:
        from convergent.sqlite_signal_backend import SQLiteSignalBackend

        backend = SQLiteSignalBackend(":memory:")
        bus_a = SignalBus(backend=backend, consumer_id="consumer-a")
        bus_b = SignalBus(backend=backend, consumer_id="consumer-b")

        received_a: list[Signal] = []
        received_b: list[Signal] = []
        bus_a.subscribe("task_complete", received_a.append)
        bus_b.subscribe("task_complete", received_b.append)

        bus_a.publish(_signal())

        signals_a = bus_a.poll_once()
        signals_b = bus_b.poll_once()

        assert len(signals_a) == 1
        assert len(signals_b) == 1
        assert len(received_a) == 1
        assert len(received_b) == 1
        backend.close()

    def test_one_consumer_processes_other_still_sees(self) -> None:
        from convergent.sqlite_signal_backend import SQLiteSignalBackend

        backend = SQLiteSignalBackend(":memory:")
        bus_a = SignalBus(backend=backend, consumer_id="consumer-a")
        bus_b = SignalBus(backend=backend, consumer_id="consumer-b")

        bus_a.publish(_signal())
        # consumer-a processes
        bus_a.poll_once()
        # consumer-a sees nothing new
        assert len(bus_a.poll_once()) == 0
        # consumer-b still sees it
        assert len(bus_b.poll_once()) == 1
        backend.close()

    def test_file_backed_multi_consumer(self, tmp_path: Path) -> None:
        from convergent.sqlite_signal_backend import SQLiteSignalBackend

        db_path = str(tmp_path / "shared.db")
        backend = SQLiteSignalBackend(db_path)
        bus_a = SignalBus(backend=backend, consumer_id="a")
        bus_b = SignalBus(backend=backend, consumer_id="b")

        bus_a.publish(_signal(source="agent-1"))
        bus_a.publish(_signal(source="agent-2"))

        a_signals = bus_a.poll_once()
        b_signals = bus_b.poll_once()
        assert len(a_signals) == 2
        assert len(b_signals) == 2

        # Both now have empty queues
        assert len(bus_a.poll_once()) == 0
        assert len(bus_b.poll_once()) == 0
        backend.close()

    def test_targeted_signal_dispatch_with_sqlite(self) -> None:
        from convergent.sqlite_signal_backend import SQLiteSignalBackend

        backend = SQLiteSignalBackend(":memory:")
        bus = SignalBus(backend=backend, consumer_id="c")

        received_target: list[Signal] = []
        received_other: list[Signal] = []
        bus.subscribe("blocked", received_target.append, agent_id="agent-2")
        bus.subscribe("blocked", received_other.append, agent_id="agent-3")

        bus.publish(_signal(signal_type="blocked", target="agent-2"))
        bus.poll_once()

        assert len(received_target) == 1
        assert len(received_other) == 0
        backend.close()


class TestPollLoopExceptionHandling:
    def test_poll_loop_survives_exception(self, tmp_path: Path) -> None:
        """Exception in poll_once doesn't kill the polling thread."""
        from unittest.mock import MagicMock

        backend = MagicMock()
        backend.get_unprocessed.side_effect = RuntimeError("transient")
        bus = SignalBus(backend=backend, poll_interval=0.02)
        bus.start_polling()
        time.sleep(0.1)  # Let it poll a few times
        assert bus.is_polling  # Still running despite exceptions
        bus.stop_polling()


class TestPublicAPI:
    def test_import_from_convergent(self) -> None:
        import convergent

        assert hasattr(convergent, "SignalBus")

    def test_all_exports_listed(self) -> None:
        import convergent

        assert "SignalBus" in convergent.__all__
