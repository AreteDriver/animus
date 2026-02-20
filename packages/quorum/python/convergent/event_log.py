"""Coordination event log — structured audit trail for multi-agent coordination.

Captures all coordination events (intent publishes, votes, decisions, markers,
signals, score updates) into a single queryable SQLite-backed timeline. Events
are tagged with correlation IDs so related events across subsystems can be
traced together.

Usage::

    from convergent.event_log import EventLog, EventType

    log = EventLog(":memory:")
    log.record(EventType.INTENT_PUBLISHED, agent_id="agent-1",
               payload={"intent": "build_auth"}, correlation_id="task-42")

    # Query by type
    events = log.query(event_type=EventType.INTENT_PUBLISHED)

    # Query by agent
    events = log.query(agent_id="agent-1")

    # Render timeline
    print(event_timeline(log.query(limit=20)))
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of coordination events captured in the log."""

    INTENT_PUBLISHED = "intent_published"
    INTENT_RESOLVED = "intent_resolved"
    CONFLICT_DETECTED = "conflict_detected"
    VOTE_CAST = "vote_cast"
    DECISION_MADE = "decision_made"
    MARKER_LEFT = "marker_left"
    MARKER_EVAPORATED = "marker_evaporated"
    SIGNAL_SENT = "signal_sent"
    SCORE_UPDATED = "score_updated"
    ESCALATION_TRIGGERED = "escalation_triggered"


@dataclass(frozen=True)
class CoordinationEvent:
    """A single coordination event in the log.

    Attributes:
        event_id: Unique event identifier.
        event_type: Category of event.
        agent_id: The agent that triggered or is associated with this event.
        timestamp: ISO 8601 UTC timestamp.
        payload: Structured event data as a dict.
        correlation_id: Links related events across subsystems (e.g., task ID).
    """

    event_id: str
    event_type: EventType
    agent_id: str
    timestamp: str
    payload: dict
    correlation_id: str | None = None


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS coordination_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL,
    correlation_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_type ON coordination_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_agent ON coordination_events(agent_id);
CREATE INDEX IF NOT EXISTS idx_events_time ON coordination_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_corr ON coordination_events(correlation_id);
"""


class EventLog:
    """Append-only event log with SQLite persistence.

    All coordination events are written to a single table with indexes
    on type, agent, timestamp, and correlation_id for efficient queries.

    Args:
        db_path: Path to SQLite database file, or ":memory:" for in-memory.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(
        self,
        event_type: EventType,
        agent_id: str,
        payload: dict | None = None,
        correlation_id: str | None = None,
        timestamp: str | None = None,
    ) -> CoordinationEvent:
        """Record a coordination event.

        Args:
            event_type: The type of event.
            agent_id: The agent associated with this event.
            payload: Structured event data. Defaults to empty dict.
            correlation_id: Optional ID to link related events.
            timestamp: ISO 8601 timestamp. Defaults to now (UTC).

        Returns:
            The recorded CoordinationEvent.
        """
        event_id = str(uuid.uuid4())
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        if payload is None:
            payload = {}

        event = CoordinationEvent(
            event_id=event_id,
            event_type=event_type,
            agent_id=agent_id,
            timestamp=timestamp,
            payload=payload,
            correlation_id=correlation_id,
        )

        self._conn.execute(
            "INSERT INTO coordination_events "
            "(event_id, event_type, agent_id, timestamp, payload, correlation_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.event_type.value,
                event.agent_id,
                event.timestamp,
                json.dumps(event.payload),
                event.correlation_id,
            ),
        )
        self._conn.commit()
        return event

    def query(
        self,
        event_type: EventType | None = None,
        agent_id: str | None = None,
        correlation_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> list[CoordinationEvent]:
        """Query events with optional filters.

        Args:
            event_type: Filter by event type.
            agent_id: Filter by agent.
            correlation_id: Filter by correlation ID.
            since: ISO 8601 timestamp lower bound (inclusive).
            until: ISO 8601 timestamp upper bound (inclusive).
            limit: Maximum results (default 100).

        Returns:
            List of matching events, ordered by timestamp ascending.
        """
        clauses: list[str] = []
        params: list[str | int] = []

        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type.value)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if correlation_id is not None:
            clauses.append("correlation_id = ?")
            params.append(correlation_id)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)

        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)

        params.append(limit)
        cursor = self._conn.execute(
            "SELECT event_id, event_type, agent_id, timestamp, payload, correlation_id "  # noqa: S608
            f"FROM coordination_events {where} "
            "ORDER BY timestamp ASC LIMIT ?",
            params,
        )

        return [self._row_to_event(row) for row in cursor]

    def count(self, event_type: EventType | None = None) -> int:
        """Count events, optionally filtered by type.

        Args:
            event_type: Optional filter.

        Returns:
            Event count.
        """
        if event_type is not None:
            cursor = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM coordination_events WHERE event_type = ?",
                (event_type.value,),
            )
        else:
            cursor = self._conn.execute("SELECT COUNT(*) as cnt FROM coordination_events")
        return cursor.fetchone()["cnt"]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _row_to_event(self, row: sqlite3.Row) -> CoordinationEvent:
        """Convert a database row to a CoordinationEvent."""
        return CoordinationEvent(
            event_id=row["event_id"],
            event_type=EventType(row["event_type"]),
            agent_id=row["agent_id"],
            timestamp=row["timestamp"],
            payload=json.loads(row["payload"]),
            correlation_id=row["correlation_id"],
        )


def event_timeline(events: list[CoordinationEvent]) -> str:
    """Render a list of events as a human-readable timeline.

    Args:
        events: Events to render (should be in timestamp order).

    Returns:
        Multi-line formatted timeline string.
    """
    if not events:
        return "(no events)"

    lines: list[str] = []
    lines.append("=== Coordination Event Timeline ===")
    lines.append("")

    for event in events:
        ts = event.timestamp
        # Truncate to seconds for readability
        if "." in ts:
            ts = ts[: ts.index(".")] + "Z"

        corr = f" [{event.correlation_id}]" if event.correlation_id else ""
        payload_str = ""
        if event.payload:
            payload_str = " " + json.dumps(event.payload, separators=(",", ":"))

        lines.append(f"  {ts} | {event.event_type.value:<22} | {event.agent_id}{corr}{payload_str}")

    lines.append("")
    lines.append(f"Total: {len(events)} events")
    return "\n".join(lines)
