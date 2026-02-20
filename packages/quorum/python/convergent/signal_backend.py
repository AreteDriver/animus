"""Pluggable signal backend protocol and filesystem implementation.

Defines the ``SignalBackend`` protocol that all signal bus backends must
implement. Includes ``FilesystemSignalBackend``, extracted from the
original ``SignalBus`` — filesystem-backed JSON file storage.

New backends (SQLite, Redis, NATS) implement the same protocol.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from convergent.protocol import Signal

logger = logging.getLogger(__name__)

# Characters unsafe in filenames — replaced with underscores
_UNSAFE_FILENAME_RE = __import__("re").compile(r"[/\\.\x00]")


def _sanitize_filename_component(value: str) -> str:
    """Replace path-separator and null characters to prevent path traversal."""
    return _UNSAFE_FILENAME_RE.sub("_", value)


@runtime_checkable
class SignalBackend(Protocol):
    """Protocol for signal bus storage backends.

    Each backend stores signals and tracks per-consumer processing state.
    ``get_unprocessed`` returns ``(backend_id, Signal)`` tuples so the
    caller can later call ``mark_processed`` with the backend IDs.
    This keeps ``Signal`` frozen and pure — no ``id`` field mutation.
    """

    def store_signal(self, signal: Signal) -> None:
        """Store a signal in the backend."""
        ...

    def get_unprocessed(self, consumer_id: str) -> list[tuple[str, Signal]]:
        """Return unprocessed signals for a consumer.

        Returns:
            List of (backend_id, Signal) tuples. The backend_id is
            opaque to the caller — it's whatever the backend needs
            to identify the signal for ``mark_processed``.
        """
        ...

    def mark_processed(self, consumer_id: str, signal_ids: list[str]) -> None:
        """Mark signals as processed by a consumer.

        Args:
            consumer_id: The consumer that processed the signals.
            signal_ids: Backend IDs returned by ``get_unprocessed``.
        """
        ...

    def get_signals(
        self,
        signal_type: str | None = None,
        since: datetime | None = None,
        source_agent: str | None = None,
    ) -> list[Signal]:
        """Query signals with optional filters.

        Args:
            signal_type: Filter by signal type. None for all.
            since: Only return signals after this time. None for all.
            source_agent: Filter by source agent. None for all.

        Returns:
            List of matching signals, ordered by timestamp.
        """
        ...

    def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        """Remove signals older than max_age_seconds.

        Returns:
            Count of signals removed.
        """
        ...

    def clear(self) -> int:
        """Remove all signals and reset state.

        Returns:
            Count of signals removed.
        """
        ...

    def close(self) -> None:
        """Release resources (close files, connections, etc.)."""
        ...


class FilesystemSignalBackend:
    """Signal backend backed by JSON files on the filesystem.

    Extracted from the original ``SignalBus`` implementation. Signals are
    stored as individual JSON files in a directory. Processing state is
    tracked per-process with in-memory sets (NOT cross-process safe).

    For cross-process signal consumption, use ``SQLiteSignalBackend``.

    Args:
        signals_dir: Directory to store signal files.
    """

    def __init__(self, signals_dir: Path) -> None:
        self._signals_dir = Path(signals_dir)
        self._signals_dir.mkdir(parents=True, exist_ok=True)
        # Per-consumer tracking (in-memory, per-process only)
        self._processed: dict[str, set[str]] = {}

    def store_signal(self, signal: Signal) -> None:
        """Write a signal as a JSON file to the signals directory."""
        safe_ts = signal.timestamp.replace(":", "-").replace("+", "p")
        safe_type = _sanitize_filename_component(signal.signal_type)
        safe_agent = _sanitize_filename_component(signal.source_agent)
        filename = f"{safe_ts}_{safe_type}_{safe_agent}.json"
        filepath = self._signals_dir / filename
        # Guard against path traversal: ensure resolved path stays within signals_dir
        if not filepath.resolve().parent == self._signals_dir.resolve():
            raise ValueError(f"Signal filename resolves outside signals directory: {filename}")
        filepath.write_text(signal.to_json(), encoding="utf-8")
        logger.info(
            "Stored signal %s from %s (target=%s)",
            signal.signal_type,
            signal.source_agent,
            signal.target_agent or "broadcast",
        )

    def get_unprocessed(self, consumer_id: str) -> list[tuple[str, Signal]]:
        """Return signals not yet processed by this consumer.

        Returns:
            List of (filename, Signal) tuples.
        """
        if consumer_id not in self._processed:
            self._processed[consumer_id] = set()
        processed = self._processed[consumer_id]

        results: list[tuple[str, Signal]] = []
        for filepath in sorted(self._signals_dir.glob("*.json")):
            fname = filepath.name
            if fname in processed:
                continue
            try:
                raw = filepath.read_text(encoding="utf-8")
                signal = Signal.from_json(raw)
            except (json.JSONDecodeError, TypeError, KeyError):
                logger.warning("Skipping malformed signal file: %s", fname)
                processed.add(fname)
                continue
            results.append((fname, signal))
        return results

    def mark_processed(self, consumer_id: str, signal_ids: list[str]) -> None:
        """Mark filenames as processed by this consumer."""
        if consumer_id not in self._processed:
            self._processed[consumer_id] = set()
        self._processed[consumer_id].update(signal_ids)

    def get_signals(
        self,
        signal_type: str | None = None,
        since: datetime | None = None,
        source_agent: str | None = None,
    ) -> list[Signal]:
        """Read signals from the directory, optionally filtered."""
        signals = []
        for filepath in sorted(self._signals_dir.glob("*.json")):
            try:
                raw = filepath.read_text(encoding="utf-8")
                signal = Signal.from_json(raw)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

            if signal_type is not None and signal.signal_type != signal_type:
                continue
            if source_agent is not None and signal.source_agent != source_agent:
                continue
            if since is not None:
                sig_time = datetime.fromisoformat(signal.timestamp)
                if sig_time.tzinfo is None:
                    sig_time = sig_time.replace(tzinfo=timezone.utc)
                if sig_time <= since:
                    continue

            signals.append(signal)
        return signals

    def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        """Remove signal files older than max_age_seconds."""
        now = datetime.now(timezone.utc)
        deleted = 0

        for filepath in self._signals_dir.glob("*.json"):
            try:
                raw = filepath.read_text(encoding="utf-8")
                signal = Signal.from_json(raw)
                sig_time = datetime.fromisoformat(signal.timestamp)
                if sig_time.tzinfo is None:
                    sig_time = sig_time.replace(tzinfo=timezone.utc)
                age = (now - sig_time).total_seconds()
                if age > max_age_seconds:
                    filepath.unlink()
                    # Remove from all consumer processed sets
                    for processed in self._processed.values():
                        processed.discard(filepath.name)
                    deleted += 1
            except (json.JSONDecodeError, TypeError, KeyError, OSError):
                continue

        if deleted:
            logger.info("Cleaned up %d expired signals", deleted)
        return deleted

    def clear(self) -> int:
        """Remove all signal files and reset processed sets."""
        deleted = 0
        for filepath in self._signals_dir.glob("*.json"):
            try:
                filepath.unlink()
                deleted += 1
            except OSError:
                continue
        self._processed.clear()
        return deleted

    def close(self) -> None:
        """No-op for filesystem backend."""

    @property
    def signals_dir(self) -> Path:
        """The directory where signals are stored."""
        return self._signals_dir
