"""Tests for convergent.sqlite_signal_backend — cross-process signal storage."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from convergent.protocol import Signal
from convergent.signal_backend import SignalBackend
from convergent.sqlite_signal_backend import SQLiteSignalBackend


def _signal(
    signal_type: str = "task_complete",
    source: str = "agent-1",
    target: str | None = None,
    payload: str = "",
) -> Signal:
    return Signal(
        signal_type=signal_type,
        source_agent=source,
        target_agent=target,
        payload=payload,
    )


class TestProtocolCompliance:
    def test_implements_signal_backend(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        assert isinstance(backend, SignalBackend)
        backend.close()


class TestStoreSignal:
    def test_store_and_retrieve(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        sig = _signal(payload='{"key": "value"}')
        backend.store_signal(sig)
        signals = backend.get_signals()
        assert len(signals) == 1
        assert signals[0].signal_type == sig.signal_type
        assert signals[0].source_agent == sig.source_agent
        assert signals[0].payload == sig.payload
        backend.close()

    def test_store_multiple(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal(source="agent-1"))
        backend.store_signal(_signal(source="agent-2"))
        signals = backend.get_signals()
        assert len(signals) == 2
        backend.close()

    def test_store_with_target(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal(target="agent-2"))
        signals = backend.get_signals()
        assert signals[0].target_agent == "agent-2"
        backend.close()

    def test_store_broadcast(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal(target=None))
        signals = backend.get_signals()
        assert signals[0].target_agent is None
        backend.close()


class TestGetUnprocessed:
    def test_returns_new_signals(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal())
        unprocessed = backend.get_unprocessed("consumer-1")
        assert len(unprocessed) == 1
        assert isinstance(unprocessed[0], tuple)
        backend_id, signal = unprocessed[0]
        assert backend_id == "1"  # First row ID
        assert signal.signal_type == "task_complete"
        backend.close()

    def test_mark_processed_prevents_redelivery(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal())
        first = backend.get_unprocessed("consumer-1")
        backend.mark_processed("consumer-1", [first[0][0]])
        second = backend.get_unprocessed("consumer-1")
        assert len(second) == 0
        backend.close()

    def test_different_consumers_independent(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal())
        c1 = backend.get_unprocessed("consumer-1")
        backend.mark_processed("consumer-1", [c1[0][0]])

        # consumer-2 still sees the signal
        c2 = backend.get_unprocessed("consumer-2")
        assert len(c2) == 1
        backend.close()

    def test_multi_consumer_both_process(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal())
        backend.store_signal(_signal(source="agent-2"))

        c1 = backend.get_unprocessed("consumer-1")
        c2 = backend.get_unprocessed("consumer-2")
        assert len(c1) == 2
        assert len(c2) == 2

        backend.mark_processed("consumer-1", [c1[0][0]])
        remaining = backend.get_unprocessed("consumer-1")
        assert len(remaining) == 1
        # consumer-2 unaffected
        c2_again = backend.get_unprocessed("consumer-2")
        assert len(c2_again) == 2
        backend.close()

    def test_order_by_id(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal(source="first"))
        backend.store_signal(_signal(source="second"))
        unprocessed = backend.get_unprocessed("c")
        assert unprocessed[0][1].source_agent == "first"
        assert unprocessed[1][1].source_agent == "second"
        backend.close()

    def test_mark_processed_duplicate_ignored(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal())
        first = backend.get_unprocessed("c")
        sid = first[0][0]
        backend.mark_processed("c", [sid])
        # Second mark should not raise (INSERT OR IGNORE)
        backend.mark_processed("c", [sid])
        backend.close()


class TestGetSignals:
    def test_get_all(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal(signal_type="blocked"))
        backend.store_signal(_signal(signal_type="task_complete"))
        signals = backend.get_signals()
        assert len(signals) == 2
        backend.close()

    def test_filter_by_type(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal(signal_type="blocked"))
        backend.store_signal(_signal(signal_type="task_complete"))
        signals = backend.get_signals(signal_type="blocked")
        assert len(signals) == 1
        assert signals[0].signal_type == "blocked"
        backend.close()

    def test_filter_by_source(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal(source="agent-1"))
        backend.store_signal(_signal(source="agent-2"))
        signals = backend.get_signals(source_agent="agent-1")
        assert len(signals) == 1
        assert signals[0].source_agent == "agent-1"
        backend.close()

    def test_filter_since(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        old_signal = Signal(signal_type="blocked", source_agent="a", timestamp=old_ts)
        backend.store_signal(old_signal)
        backend.store_signal(_signal(signal_type="task_complete"))  # fresh
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        signals = backend.get_signals(since=cutoff)
        assert len(signals) == 1
        assert signals[0].signal_type == "task_complete"
        backend.close()

    def test_combined_filters(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal(signal_type="blocked", source="agent-1"))
        backend.store_signal(_signal(signal_type="blocked", source="agent-2"))
        backend.store_signal(_signal(signal_type="task_complete", source="agent-1"))
        signals = backend.get_signals(signal_type="blocked", source_agent="agent-1")
        assert len(signals) == 1
        backend.close()

    def test_empty_database(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        assert backend.get_signals() == []
        backend.close()


class TestCleanupExpired:
    def test_removes_old_signals(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        old_signal = Signal(signal_type="blocked", source_agent="a", timestamp=old_ts)
        backend.store_signal(old_signal)
        backend.store_signal(_signal())  # fresh
        deleted = backend.cleanup_expired(max_age_seconds=3600)
        assert deleted == 1
        remaining = backend.get_signals()
        assert len(remaining) == 1
        backend.close()

    def test_cascades_to_consumers(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        old_signal = Signal(signal_type="blocked", source_agent="a", timestamp=old_ts)
        backend.store_signal(old_signal)

        # Process it first
        unprocessed = backend.get_unprocessed("c")
        backend.mark_processed("c", [unprocessed[0][0]])

        # Cleanup should remove both signal and consumer record
        deleted = backend.cleanup_expired(max_age_seconds=3600)
        assert deleted == 1

        # Verify consumer record also gone
        cursor = backend._conn.execute("SELECT COUNT(*) as cnt FROM signal_consumers")
        assert cursor.fetchone()["cnt"] == 0
        backend.close()

    def test_no_expired(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal())
        assert backend.cleanup_expired(max_age_seconds=3600) == 0
        backend.close()


class TestClear:
    def test_removes_all(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal(source="agent-1"))
        backend.store_signal(_signal(source="agent-2"))
        deleted = backend.clear()
        assert deleted == 2
        assert backend.get_signals() == []
        backend.close()

    def test_clear_also_removes_consumer_records(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.store_signal(_signal())
        unprocessed = backend.get_unprocessed("c")
        backend.mark_processed("c", [unprocessed[0][0]])
        backend.clear()
        cursor = backend._conn.execute("SELECT COUNT(*) as cnt FROM signal_consumers")
        assert cursor.fetchone()["cnt"] == 0
        backend.close()

    def test_clear_empty_database(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        assert backend.clear() == 0
        backend.close()


class TestClose:
    def test_close(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        backend.close()
        with pytest.raises(sqlite3.ProgrammingError):
            backend.store_signal(_signal())


class TestFilePersistence:
    def test_persist_and_reload(self, tmp_path: object) -> None:
        import pathlib

        db_path = str(pathlib.Path(str(tmp_path)) / "signals.db")
        b1 = SQLiteSignalBackend(db_path)
        b1.store_signal(_signal(source="agent-1"))
        b1.store_signal(_signal(source="agent-2"))
        b1.mark_processed("consumer-1", ["1"])
        b1.close()

        b2 = SQLiteSignalBackend(db_path)
        all_signals = b2.get_signals()
        assert len(all_signals) == 2
        # consumer-1 processing state persisted
        unprocessed = b2.get_unprocessed("consumer-1")
        assert len(unprocessed) == 1
        assert unprocessed[0][1].source_agent == "agent-2"
        b2.close()

    def test_cross_connection_visibility(self, tmp_path: object) -> None:
        """Two connections to the same DB see each other's signals."""
        import pathlib

        db_path = str(pathlib.Path(str(tmp_path)) / "shared.db")
        b1 = SQLiteSignalBackend(db_path)
        b2 = SQLiteSignalBackend(db_path)

        b1.store_signal(_signal(source="from-b1"))
        signals_from_b2 = b2.get_signals()
        assert len(signals_from_b2) == 1
        assert signals_from_b2[0].source_agent == "from-b1"

        b1.close()
        b2.close()


class TestProperties:
    def test_db_path(self) -> None:
        backend = SQLiteSignalBackend(":memory:")
        assert backend.db_path == ":memory:"
        backend.close()


class TestPublicAPI:
    def test_import_from_convergent(self) -> None:
        import convergent

        assert hasattr(convergent, "SQLiteSignalBackend")

    def test_all_exports_listed(self) -> None:
        import convergent

        assert "SQLiteSignalBackend" in convergent.__all__
