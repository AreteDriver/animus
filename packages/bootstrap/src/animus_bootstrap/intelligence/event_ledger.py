"""Operational event ledger — append-only log of everything the runtime does.

Feeds the Cognitive Operations Center dashboard with live telemetry.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


class EventLedger:
    """Thread-safe append-only event log with in-memory ring buffer + SQLite persistence.

    Event types
    -----------
    ``tool_execution``    – Tool was executed (success/fail, duration).
    ``memory_recall``      – Memory was recalled (score, query).
    ``task_created``       – Task created.
    ``task_completed``     – Task completed.
    ``task_deleted``       – Task deleted.
    ``session_started``    – Runtime/session started.
    ``session_ended``      – Runtime/session stopped.
    ``proposal_approved``  – Improvement proposal approved.
    ``proposal_rejected``  – Improvement proposal rejected.
    ``feedback_recorded``  – User feedback recorded.
    ``config_changed``     – Config saved.
    ``error``              – Runtime or component error.
    """

    def __init__(self, max_events: int = 10_000, db_path: Path | None = None) -> None:
        self._max = max_events
        self._events: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._db_path = db_path
        if db_path is not None:
            self._init_db()
            self._load_from_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, event_type: str, source: str, payload: dict[str, Any] | None = None) -> None:
        """Append an event with auto-timestamp."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "source": source,
            "payload": payload or {},
        }
        with self._lock:
            self._events.append(event)
            # Ring buffer eviction
            if len(self._events) > self._max:
                self._events.pop(0)
            if self._db_path is not None:
                self._persist_event(event)

    def query(
        self,
        event_type: str | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return recent events, optionally filtered."""
        with self._lock:
            events = list(self._events)
        # Filter and slice (newest first)
        filtered = events
        if event_type is not None:
            filtered = [e for e in filtered if e["type"] == event_type]
        if source is not None:
            filtered = [e for e in filtered if e["source"] == source]
        return filtered[-limit:][::-1]

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate stats."""
        with self._lock:
            events = list(self._events)
        if not events:
            return {
                "total": 0,
                "by_type": {},
                "by_source": {},
                "last_event_time": None,
                "events_per_min": 0.0,
            }
        counts_by_type = Counter(e["type"] for e in events)
        counts_by_source = Counter(e["source"] for e in events)
        last_time = max(e["timestamp"] for e in events)
        # Events per minute over the last 5 minutes
        now = time.time()
        recent = [e for e in events if now - e["timestamp"] <= 300]
        epm = len(recent) / 5.0 if recent else 0.0
        return {
            "total": len(events),
            "by_type": dict(counts_by_type),
            "by_source": dict(counts_by_source),
            "last_event_time": last_time,
            "events_per_min": round(epm, 2),
        }

    def get_recent_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """Convenience: return recent error events."""
        return self.query(event_type="error", limit=limit)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Ensure SQLite schema exists."""
        if self._db_path is None:
            return
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            # Best-effort; if we can't create the dir, persistence is disabled
            self._db_path = None
            return
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    payload TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type ON events(type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp)"
            )
            conn.commit()

    def _persist_event(self, event: dict[str, Any]) -> None:
        """Write a single event to SQLite."""
        if self._db_path is None:
            return
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "INSERT INTO events (timestamp, type, source, payload) VALUES (?, ?, ?, ?)",
                    (
                        event["timestamp"],
                        event["type"],
                        event["source"],
                        json.dumps(event["payload"]),
                    ),
                )
                conn.commit()
        except Exception:
            # Persistence is best-effort; never crash the hot path
            pass

    def _load_from_db(self) -> None:
        """Hydrate the in-memory ring buffer from SQLite on startup."""
        if self._db_path is None or not self._db_path.exists():
            return
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                cursor = conn.execute(
                    "SELECT timestamp, type, source, payload FROM events ORDER BY timestamp DESC LIMIT ?",
                    (self._max,),
                )
                rows = cursor.fetchall()
                # Insert in chronological order
                for ts, typ, src, payload_json in reversed(rows):
                    payload = json.loads(payload_json) if payload_json else {}
                    self._events.append(
                        {"timestamp": ts, "type": typ, "source": src, "payload": payload}
                    )
        except Exception:
            pass

    def flush(self) -> None:
        """No-op for interface compatibility; events are persisted immediately."""

    def close(self) -> None:
        """No-op; SQLite connections are short-lived."""
