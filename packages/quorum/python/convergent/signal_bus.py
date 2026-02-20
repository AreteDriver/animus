"""Lightweight pub/sub signal bus for inter-agent communication.

Orchestrates subscriber management, polling, and dispatch. Delegates
storage to a pluggable ``SignalBackend``. Defaults to
``FilesystemSignalBackend`` for backward compatibility.

Signals are how agents say: "I'm blocked on X", "I finished Y, you can
start Z", "I found a conflict in file W."
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from convergent.protocol import Signal
from convergent.signal_backend import FilesystemSignalBackend, SignalBackend

logger = logging.getLogger(__name__)

# Type alias for signal callbacks
_Callback = Callable[[Signal], None]


class _SubscriberEntry:
    """Internal subscriber entry with optional agent filter.

    Attributes:
        callback: The callback function.
        agent_id: If set, only deliver targeted signals to this agent.
    """

    __slots__ = ("callback", "agent_id")

    def __init__(self, callback: _Callback, agent_id: str | None = None) -> None:
        self.callback = callback
        self.agent_id = agent_id


class SignalBus:
    """Pub/sub signal bus with pluggable backend.

    Signals are stored via a ``SignalBackend``. Subscribers register
    callbacks for specific signal types. A polling thread reads new signals
    and dispatches to matching subscribers.

    Args:
        signals_dir: Directory for filesystem backend. Ignored if ``backend``
            is provided. Kept for backward compatibility.
        poll_interval: Seconds between polling cycles.
        backend: Explicit backend. If None, creates a
            ``FilesystemSignalBackend`` from ``signals_dir``.
        consumer_id: Identifier for this consumer. Different consumers
            on the same backend track processing state independently.
    """

    def __init__(
        self,
        signals_dir: Path | None = None,
        poll_interval: float = 1.0,
        backend: SignalBackend | None = None,
        consumer_id: str = "default",
    ) -> None:
        if backend is not None:
            self._backend = backend
        elif signals_dir is not None:
            self._backend = FilesystemSignalBackend(signals_dir)
        else:
            raise ValueError("Either signals_dir or backend must be provided")

        self._consumer_id = consumer_id
        self._poll_interval = poll_interval
        self._subscribers: dict[str, list[_SubscriberEntry]] = {}
        self._lock = threading.Lock()
        self._polling = False
        self._poll_thread: threading.Thread | None = None

    @property
    def backend(self) -> SignalBackend:
        """Access the underlying storage backend."""
        return self._backend

    @property
    def consumer_id(self) -> str:
        """This bus's consumer identity."""
        return self._consumer_id

    def publish(self, signal: Signal) -> None:
        """Publish a signal to the bus.

        Args:
            signal: The signal to publish.
        """
        self._backend.store_signal(signal)
        logger.info(
            "Published signal %s from %s (target=%s)",
            signal.signal_type,
            signal.source_agent,
            signal.target_agent or "broadcast",
        )

    def subscribe(
        self,
        signal_type: str,
        callback: _Callback,
        agent_id: str | None = None,
    ) -> None:
        """Subscribe to signals of a specific type.

        Args:
            signal_type: The signal type to listen for.
            callback: Function called with matching Signal objects.
            agent_id: If set, only receive signals targeted to this agent
                or broadcast signals. If None, receive all matching signals.
        """
        with self._lock:
            if signal_type not in self._subscribers:
                self._subscribers[signal_type] = []
            self._subscribers[signal_type].append(
                _SubscriberEntry(callback=callback, agent_id=agent_id)
            )
        logger.debug(
            "Subscribed to %s (agent_id=%s)",
            signal_type,
            agent_id or "any",
        )

    def unsubscribe(self, signal_type: str, callback: _Callback) -> bool:
        """Remove a callback subscription.

        Args:
            signal_type: The signal type to unsubscribe from.
            callback: The callback to remove.

        Returns:
            True if the callback was found and removed, False otherwise.
        """
        with self._lock:
            entries = self._subscribers.get(signal_type, [])
            for i, entry in enumerate(entries):
                if entry.callback is callback:
                    entries.pop(i)
                    return True
        return False

    def start_polling(self) -> None:
        """Start the background polling thread.

        Reads new signal files and dispatches to matching subscribers.
        Call stop_polling() to stop.

        Raises:
            RuntimeError: If polling is already active.
        """
        if self._polling:
            raise RuntimeError("Polling is already active")
        self._polling = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="signal-bus-poll"
        )
        self._poll_thread.start()
        logger.info("Signal bus polling started (interval=%.1fs)", self._poll_interval)

    def stop_polling(self) -> None:
        """Stop the background polling thread."""
        self._polling = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=self._poll_interval * 2)
            self._poll_thread = None
        logger.info("Signal bus polling stopped")

    def poll_once(self) -> list[Signal]:
        """Poll for new signals once (synchronous).

        Reads unprocessed signals via the backend, dispatches to
        subscribers, and marks them as processed.

        Returns:
            List of new signals found during this poll cycle.
        """
        unprocessed = self._backend.get_unprocessed(self._consumer_id)
        ids: list[str] = []
        signals: list[Signal] = []

        for backend_id, signal in unprocessed:
            ids.append(backend_id)
            signals.append(signal)
            self._dispatch(signal)

        if ids:
            self._backend.mark_processed(self._consumer_id, ids)

        return signals

    def get_signals(
        self,
        signal_type: str | None = None,
        since: datetime | None = None,
        source_agent: str | None = None,
    ) -> list[Signal]:
        """Read signals, optionally filtered.

        Args:
            signal_type: Filter by signal type. None for all.
            since: Only return signals after this time. None for all.
            source_agent: Filter by source agent. None for all.

        Returns:
            List of matching signals, ordered by timestamp.
        """
        return self._backend.get_signals(
            signal_type=signal_type,
            since=since,
            source_agent=source_agent,
        )

    def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        """Remove signals older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds before deletion.

        Returns:
            Count of signals deleted.
        """
        return self._backend.cleanup_expired(max_age_seconds)

    def clear(self) -> int:
        """Remove all signals and reset state.

        Returns:
            Count of signals deleted.
        """
        return self._backend.clear()

    def close(self) -> None:
        """Stop polling and close the backend."""
        self.stop_polling()
        self._backend.close()

    @property
    def is_polling(self) -> bool:
        """Whether the polling thread is active."""
        return self._polling

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._polling:
            try:
                self.poll_once()
            except Exception:
                logger.exception("Error during signal bus poll cycle")
            time.sleep(self._poll_interval)

    def _dispatch(self, signal: Signal) -> None:
        """Dispatch a signal to matching subscribers."""
        with self._lock:
            entries = list(self._subscribers.get(signal.signal_type, []))

        for entry in entries:
            # Filter by target: deliver if broadcast, or if targeted to subscriber
            if (
                signal.target_agent is not None
                and entry.agent_id is not None
                and signal.target_agent != entry.agent_id
            ):
                continue
            try:
                entry.callback(signal)
            except Exception:
                logger.exception("Error in signal subscriber for %s", signal.signal_type)
