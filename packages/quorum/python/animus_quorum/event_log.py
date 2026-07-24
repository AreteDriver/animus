"""Coordination event log — structured audit trail for multi-agent coordination.

Captures all coordination events (intent publishes, votes, decisions, markers,
signals, score updates) into a single queryable SQLite-backed timeline. Events
are tagged with correlation IDs so related events across subsystems can be
traced together.

Usage::

    from animus_quorum.event_log import EventLog, EventType

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from animus_quorum.signal_bus import SignalBus

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
        agent_id: The agent associated with this event.
        timestamp: ISO 8601 UTC timestamp. Backward-compatible alias
            for ``recorded_at``; on new writes both are equal.
        payload: Structured event data as a dict.
        correlation_id: Links related events across subsystems
            (e.g., task ID).
        valid_from: ISO 8601 timestamp marking when the underlying
            fact became true in the world. Defaults to ``recorded_at``
            if the caller does not supply it. Set explicitly when
            ingesting historical traces.
        recorded_at: ISO 8601 timestamp marking when this system
            observed the fact. Defaults to wall-clock now (UTC).
    """

    event_id: str
    event_type: EventType
    agent_id: str
    timestamp: str
    payload: dict
    correlation_id: str | None = None
    valid_from: str | None = None
    recorded_at: str | None = None


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS coordination_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    valid_from TEXT,
    recorded_at TEXT,
    payload TEXT NOT NULL,
    correlation_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_type
    ON coordination_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_agent
    ON coordination_events(agent_id);
CREATE INDEX IF NOT EXISTS idx_events_time
    ON coordination_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_corr
    ON coordination_events(correlation_id);
"""
# Bitemporal indexes are created in _migrate() after columns
# are guaranteed to exist (legacy DBs predate these columns).


class EventLog:
    """Append-only event log with SQLite persistence.

    All coordination events are written to a single table with
    indexes on type, agent, timestamp, correlation_id, valid_from,
    and recorded_at for efficient queries.

    Bitemporal-lite (ported from memboot's pattern):
        ``valid_from`` records when a fact became true in the world.
        ``recorded_at`` records when this system observed the fact.
        On default writes the two are equal; ingesting historical
        traces sets ``valid_from`` to the past while ``recorded_at``
        stays at observation time.

    Optional signal bridge:
        When ``signal_bus`` is provided, every recorded event is
        also published as a ``Signal`` of type
        ``event.<event_type>`` so subscribers (Watchdog, dashboard)
        can react in real-time. Bridge failures are swallowed and
        logged — the bus is best-effort observability, not a source
        of truth.

    Args:
        db_path: Path to SQLite file, or ``":memory:"``.
        signal_bus: Optional bus to mirror events onto.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        signal_bus: SignalBus | None = None,
    ) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()
        self._signal_bus = signal_bus

    def _migrate(self) -> None:
        """Add bitemporal columns to pre-existing databases.

        Idempotent: safe to run on a fresh schema (no-op) or on a
        legacy database missing ``valid_from`` / ``recorded_at``
        (adds columns and backfills both from ``timestamp``).
        """
        cur = self._conn.execute("PRAGMA table_info(coordination_events)")
        cols = {row["name"] for row in cur.fetchall()}
        if "valid_from" not in cols:
            self._conn.execute("ALTER TABLE coordination_events ADD COLUMN valid_from TEXT")
            self._conn.execute("UPDATE coordination_events SET valid_from = timestamp")
        if "recorded_at" not in cols:
            self._conn.execute("ALTER TABLE coordination_events ADD COLUMN recorded_at TEXT")
            self._conn.execute("UPDATE coordination_events SET recorded_at = timestamp")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_valid_from ON coordination_events(valid_from)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_recorded_at ON coordination_events(recorded_at)"
        )

    def record(
        self,
        event_type: EventType,
        agent_id: str,
        payload: dict | None = None,
        correlation_id: str | None = None,
        timestamp: str | None = None,
        valid_from: str | None = None,
        recorded_at: str | None = None,
    ) -> CoordinationEvent:
        """Record a coordination event.

        Bitemporal semantics (ported from memboot):
            ``recorded_at`` is when this system observed the fact;
            defaults to wall-clock now (UTC). ``valid_from`` is when
            the fact became true in the world; defaults to
            ``recorded_at``. Supply ``valid_from`` explicitly when
            ingesting historical traces.

            ``timestamp`` is preserved as a backward-compatible
            alias for ``recorded_at``; on new writes it equals
            ``recorded_at``.

        Args:
            event_type: The type of event.
            agent_id: The agent associated with this event.
            payload: Structured event data. Defaults to empty dict.
            correlation_id: Optional ID to link related events.
            timestamp: ISO 8601 alias for ``recorded_at``.
            valid_from: ISO 8601 world-time of the fact.
            recorded_at: ISO 8601 observation-time of the fact.

        Returns:
            The recorded CoordinationEvent.
        """
        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        recorded_at = recorded_at or timestamp or now
        valid_from = valid_from or recorded_at
        timestamp = timestamp or recorded_at
        if payload is None:
            payload = {}

        event = CoordinationEvent(
            event_id=event_id,
            event_type=event_type,
            agent_id=agent_id,
            timestamp=timestamp,
            payload=payload,
            correlation_id=correlation_id,
            valid_from=valid_from,
            recorded_at=recorded_at,
        )

        self._conn.execute(
            "INSERT INTO coordination_events "
            "(event_id, event_type, agent_id, timestamp, "
            "valid_from, recorded_at, payload, correlation_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.event_type.value,
                event.agent_id,
                event.timestamp,
                event.valid_from,
                event.recorded_at,
                json.dumps(event.payload),
                event.correlation_id,
            ),
        )
        self._conn.commit()
        if self._signal_bus is not None:
            self._publish_signal(event)
        return event

    def _publish_signal(self, event: CoordinationEvent) -> None:
        """Mirror an event onto the signal bus.

        Best-effort observability — failures are swallowed and
        logged so a misbehaving subscriber cannot break record().
        """
        bus = self._signal_bus
        if bus is None:
            return
        try:
            from animus_quorum.protocol import Signal

            envelope = json.dumps(
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "valid_from": event.valid_from,
                    "recorded_at": event.recorded_at,
                    "correlation_id": event.correlation_id,
                    "payload": event.payload,
                }
            )
            signal = Signal(
                signal_type=f"event.{event.event_type.value}",
                source_agent=event.agent_id,
                target_agent=None,
                payload=envelope,
                timestamp=event.recorded_at or event.timestamp,
            )
            bus.publish(signal)
        except Exception as exc:
            logger.warning("event_log signal bridge failed: %s", exc)

    def query(
        self,
        event_type: EventType | None = None,
        agent_id: str | None = None,
        correlation_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        valid_from_since: str | None = None,
        valid_from_until: str | None = None,
        limit: int = 100,
    ) -> list[CoordinationEvent]:
        """Query events with optional filters.

        ``since`` and ``until`` filter by ``recorded_at`` (i.e.,
        when this system observed the event). ``valid_from_since``
        and ``valid_from_until`` filter by world-time. The two
        windows compose — pass both to constrain on observation
        and world time simultaneously.

        Args:
            event_type: Filter by event type.
            agent_id: Filter by agent.
            correlation_id: Filter by correlation ID.
            since: ISO 8601 ``recorded_at`` lower bound.
            until: ISO 8601 ``recorded_at`` upper bound.
            valid_from_since: ISO 8601 ``valid_from`` lower bound.
            valid_from_until: ISO 8601 ``valid_from`` upper bound.
            limit: Maximum results (default 100).

        Returns:
            List of matching events, ordered by timestamp ASC.
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
            clauses.append("COALESCE(recorded_at, timestamp) >= ?")
            params.append(since)
        if until is not None:
            clauses.append("COALESCE(recorded_at, timestamp) <= ?")
            params.append(until)
        if valid_from_since is not None:
            clauses.append("COALESCE(valid_from, timestamp) >= ?")
            params.append(valid_from_since)
        if valid_from_until is not None:
            clauses.append("COALESCE(valid_from, timestamp) <= ?")
            params.append(valid_from_until)

        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)

        params.append(limit)
        cursor = self._conn.execute(
            "SELECT event_id, event_type, agent_id, timestamp, "
            "valid_from, recorded_at, payload, correlation_id "  # noqa: S608
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
        keys = row.keys()
        valid_from = row["valid_from"] if "valid_from" in keys else None
        recorded_at = row["recorded_at"] if "recorded_at" in keys else None
        return CoordinationEvent(
            event_id=row["event_id"],
            event_type=EventType(row["event_type"]),
            agent_id=row["agent_id"],
            timestamp=row["timestamp"],
            payload=json.loads(row["payload"]),
            correlation_id=row["correlation_id"],
            valid_from=valid_from or row["timestamp"],
            recorded_at=recorded_at or row["timestamp"],
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
