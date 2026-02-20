"""Tests for convergent.signal_backend — protocol and FilesystemSignalBackend."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from convergent.protocol import Signal
from convergent.signal_backend import FilesystemSignalBackend, SignalBackend


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


class TestSignalBackendProtocol:
    def test_filesystem_backend_implements_protocol(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        assert isinstance(backend, SignalBackend)

    def test_protocol_is_runtime_checkable(self) -> None:
        assert hasattr(SignalBackend, "__protocol_attrs__") or hasattr(
            SignalBackend, "__abstractmethods__"
        )


class TestFilesystemStore:
    def test_store_creates_file(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        backend.store_signal(_signal())
        files = list((tmp_path / "signals").glob("*.json"))
        assert len(files) == 1

    def test_store_multiple(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        backend.store_signal(_signal(source="agent-1"))
        backend.store_signal(_signal(source="agent-2"))
        files = list((tmp_path / "signals").glob("*.json"))
        assert len(files) == 2

    def test_store_creates_directory(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "nested" / "signals")
        backend.store_signal(_signal())
        assert (tmp_path / "nested" / "signals").is_dir()


class TestFilesystemGetUnprocessed:
    def test_returns_new_signals(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        backend.store_signal(_signal())
        unprocessed = backend.get_unprocessed("consumer-1")
        assert len(unprocessed) == 1
        assert isinstance(unprocessed[0], tuple)
        assert isinstance(unprocessed[0][0], str)  # backend_id
        assert isinstance(unprocessed[0][1], Signal)

    def test_marks_processed_prevents_redelivery(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        backend.store_signal(_signal())
        first = backend.get_unprocessed("consumer-1")
        backend.mark_processed("consumer-1", [first[0][0]])
        second = backend.get_unprocessed("consumer-1")
        assert len(second) == 0

    def test_different_consumers_independent(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        backend.store_signal(_signal())
        c1 = backend.get_unprocessed("consumer-1")
        backend.mark_processed("consumer-1", [c1[0][0]])
        c2 = backend.get_unprocessed("consumer-2")
        assert len(c2) == 1

    def test_malformed_file_skipped(self, tmp_path: Path) -> None:
        signals_dir = tmp_path / "signals"
        signals_dir.mkdir()
        (signals_dir / "bad.json").write_text("not json", encoding="utf-8")
        backend = FilesystemSignalBackend(signals_dir)
        unprocessed = backend.get_unprocessed("consumer-1")
        assert len(unprocessed) == 0

    def test_malformed_file_not_reprocessed(self, tmp_path: Path) -> None:
        signals_dir = tmp_path / "signals"
        signals_dir.mkdir()
        (signals_dir / "bad.json").write_text("{invalid", encoding="utf-8")
        backend = FilesystemSignalBackend(signals_dir)
        backend.get_unprocessed("consumer-1")
        # Second call should also return empty (malformed marked as processed)
        unprocessed = backend.get_unprocessed("consumer-1")
        assert len(unprocessed) == 0


class TestFilesystemGetSignals:
    def test_get_all_signals(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        backend.store_signal(_signal(signal_type="blocked"))
        backend.store_signal(_signal(signal_type="task_complete"))
        signals = backend.get_signals()
        assert len(signals) == 2

    def test_filter_by_type(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        backend.store_signal(_signal(signal_type="blocked"))
        backend.store_signal(_signal(signal_type="task_complete"))
        signals = backend.get_signals(signal_type="blocked")
        assert len(signals) == 1
        assert signals[0].signal_type == "blocked"

    def test_filter_by_source(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        backend.store_signal(_signal(source="agent-1"))
        backend.store_signal(_signal(source="agent-2"))
        signals = backend.get_signals(source_agent="agent-1")
        assert len(signals) == 1

    def test_filter_since(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        old_signal = Signal(signal_type="blocked", source_agent="agent-1", timestamp=old_ts)
        backend.store_signal(old_signal)
        backend.store_signal(_signal(signal_type="task_complete"))
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        signals = backend.get_signals(since=cutoff)
        assert len(signals) == 1
        assert signals[0].signal_type == "task_complete"

    def test_empty_directory(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        assert backend.get_signals() == []


class TestFilesystemCleanup:
    def test_cleanup_expired_removes_old(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        old_signal = Signal(signal_type="blocked", source_agent="agent-1", timestamp=old_ts)
        backend.store_signal(old_signal)
        backend.store_signal(_signal())  # fresh
        deleted = backend.cleanup_expired(max_age_seconds=3600)
        assert deleted == 1
        remaining = backend.get_signals()
        assert len(remaining) == 1

    def test_cleanup_no_expired(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        backend.store_signal(_signal())
        assert backend.cleanup_expired(max_age_seconds=3600) == 0


class TestFilesystemClear:
    def test_clear_removes_all(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        backend.store_signal(_signal(source="agent-1"))
        backend.store_signal(_signal(source="agent-2"))
        deleted = backend.clear()
        assert deleted == 2
        assert backend.get_signals() == []

    def test_clear_resets_processed(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        backend.store_signal(_signal())
        first = backend.get_unprocessed("consumer-1")
        backend.mark_processed("consumer-1", [first[0][0]])
        backend.clear()
        backend.store_signal(_signal())
        second = backend.get_unprocessed("consumer-1")
        assert len(second) == 1


class TestFilesystemClose:
    def test_close_is_noop(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        backend.close()  # Should not raise


class TestFilesystemProperties:
    def test_signals_dir(self, tmp_path: Path) -> None:
        backend = FilesystemSignalBackend(tmp_path / "signals")
        assert backend.signals_dir == tmp_path / "signals"


class TestSignalBusWithExplicitBackend:
    """Test SignalBus with explicit backend parameter."""

    def test_explicit_backend(self, tmp_path: Path) -> None:
        from convergent.signal_bus import SignalBus

        backend = FilesystemSignalBackend(tmp_path / "signals")
        bus = SignalBus(backend=backend)
        bus.publish(_signal())
        signals = bus.poll_once()
        assert len(signals) == 1

    def test_no_args_raises(self) -> None:
        from convergent.signal_bus import SignalBus

        with pytest.raises(ValueError, match="Either signals_dir or backend"):
            SignalBus()

    def test_backend_property(self, tmp_path: Path) -> None:
        from convergent.signal_bus import SignalBus

        backend = FilesystemSignalBackend(tmp_path / "signals")
        bus = SignalBus(backend=backend)
        assert bus.backend is backend

    def test_consumer_id_property(self, tmp_path: Path) -> None:
        from convergent.signal_bus import SignalBus

        bus = SignalBus(signals_dir=tmp_path / "signals", consumer_id="my-consumer")
        assert bus.consumer_id == "my-consumer"

    def test_close_stops_polling_and_closes_backend(self, tmp_path: Path) -> None:
        from convergent.signal_bus import SignalBus

        backend = FilesystemSignalBackend(tmp_path / "signals")
        bus = SignalBus(backend=backend, poll_interval=0.05)
        bus.start_polling()
        assert bus.is_polling
        bus.close()
        assert not bus.is_polling


class TestFilesystemEdgeCases:
    """Edge case tests for coverage push."""

    def test_mark_processed_without_prior_get(self, tmp_path: Path) -> None:
        """mark_processed creates consumer set if get_unprocessed wasn't called first."""
        backend = FilesystemSignalBackend(tmp_path / "signals")
        backend.store_signal(_signal())
        # Call mark_processed directly — consumer set doesn't exist yet
        backend.mark_processed("new-consumer", ["some-file.json"])
        # Now get_unprocessed should still return the signal (different ID)
        unprocessed = backend.get_unprocessed("new-consumer")
        assert len(unprocessed) == 1

    def test_get_signals_naive_timestamp(self, tmp_path: Path) -> None:
        """Signal with naive timestamp is treated as UTC in get_signals."""
        backend = FilesystemSignalBackend(tmp_path / "signals")
        # Create signal with naive timestamp (strip +00:00 from UTC)
        utc_now = datetime.now(timezone.utc)
        naive_ts = utc_now.strftime("%Y-%m-%dT%H:%M:%S.%f")
        sig = Signal(signal_type="test", source_agent="a1", timestamp=naive_ts)
        backend.store_signal(sig)
        cutoff = utc_now - timedelta(hours=1)
        signals = backend.get_signals(since=cutoff)
        assert len(signals) == 1

    def test_cleanup_expired_naive_timestamp(self, tmp_path: Path) -> None:
        """Signal with naive old timestamp gets cleaned up."""
        backend = FilesystemSignalBackend(tmp_path / "signals")
        old_utc = datetime.now(timezone.utc) - timedelta(hours=2)
        old_ts = old_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")  # naive
        sig = Signal(signal_type="test", source_agent="a1", timestamp=old_ts)
        backend.store_signal(sig)
        deleted = backend.cleanup_expired(max_age_seconds=3600)
        assert deleted == 1

    def test_cleanup_expired_removes_from_processed(self, tmp_path: Path) -> None:
        """Cleanup removes expired signal from consumer processed sets."""
        backend = FilesystemSignalBackend(tmp_path / "signals")
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        sig = Signal(signal_type="test", source_agent="a1", timestamp=old_ts)
        backend.store_signal(sig)
        # Mark as processed by a consumer
        unprocessed = backend.get_unprocessed("consumer-1")
        assert len(unprocessed) == 1
        backend.mark_processed("consumer-1", [unprocessed[0][0]])
        # Cleanup should remove from processed set too
        deleted = backend.cleanup_expired(max_age_seconds=3600)
        assert deleted == 1

    def test_cleanup_malformed_file_skipped(self, tmp_path: Path) -> None:
        """Malformed JSON files don't break cleanup."""
        signals_dir = tmp_path / "signals"
        signals_dir.mkdir()
        (signals_dir / "bad.json").write_text("not json", encoding="utf-8")
        backend = FilesystemSignalBackend(signals_dir)
        deleted = backend.cleanup_expired(max_age_seconds=0)
        assert deleted == 0
        # File should still exist
        assert (signals_dir / "bad.json").exists()

    def test_clear_with_unreadable_file(self, tmp_path: Path) -> None:
        """OSError during clear is handled gracefully."""
        import os

        if os.getuid() == 0:
            pytest.skip("Root bypasses filesystem permission checks")

        signals_dir = tmp_path / "signals"
        signals_dir.mkdir()
        # Create a file we can't delete (read-only directory)
        (signals_dir / "test.json").write_text("{}", encoding="utf-8")
        backend = FilesystemSignalBackend(signals_dir)
        # Make directory read-only to cause OSError on unlink
        os.chmod(signals_dir, 0o555)
        try:
            deleted = backend.clear()
            assert deleted == 0  # Couldn't delete
        finally:
            os.chmod(signals_dir, 0o755)


class TestPathTraversalPrevention:
    def test_slash_in_signal_type_sanitized(self, tmp_path: Path) -> None:
        """signal_type with path separators must not escape signals directory."""
        backend = FilesystemSignalBackend(tmp_path / "signals")
        sig = Signal(signal_type="../../etc/malicious", source_agent="agent-1")
        backend.store_signal(sig)
        # File should be inside signals_dir, not escaped
        files = list((tmp_path / "signals").glob("*.json"))
        assert len(files) == 1
        assert files[0].parent == tmp_path / "signals"

    def test_slash_in_source_agent_sanitized(self, tmp_path: Path) -> None:
        """source_agent with path separators must not escape signals directory."""
        backend = FilesystemSignalBackend(tmp_path / "signals")
        sig = Signal(signal_type="test", source_agent="../../../tmp/evil")
        backend.store_signal(sig)
        files = list((tmp_path / "signals").glob("*.json"))
        assert len(files) == 1
        assert files[0].parent == tmp_path / "signals"

    def test_dot_segments_sanitized(self, tmp_path: Path) -> None:
        """Dots in signal fields are sanitized to prevent traversal."""
        backend = FilesystemSignalBackend(tmp_path / "signals")
        sig = Signal(signal_type="...", source_agent="a..b")
        backend.store_signal(sig)
        files = list((tmp_path / "signals").glob("*.json"))
        assert len(files) == 1

    def test_null_byte_sanitized(self, tmp_path: Path) -> None:
        """Null bytes in signal fields are sanitized."""
        backend = FilesystemSignalBackend(tmp_path / "signals")
        sig = Signal(signal_type="test\x00evil", source_agent="agent-1")
        backend.store_signal(sig)
        files = list((tmp_path / "signals").glob("*.json"))
        assert len(files) == 1


class TestPublicAPI:
    def test_import_signal_backend(self) -> None:
        import convergent

        assert hasattr(convergent, "SignalBackend")
        assert hasattr(convergent, "FilesystemSignalBackend")

    def test_all_exports_listed(self) -> None:
        import convergent

        assert "SignalBackend" in convergent.__all__
        assert "FilesystemSignalBackend" in convergent.__all__
