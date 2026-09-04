"""SchedulerMetrics — lightweight event telemetry for the mission scheduler.

Stores counter-style events (tasks dispatched, results processed, etc.) in the
shared ``DatabaseBackend`` for observability and post-hoc analysis.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from animus_forge.state.backends import DatabaseBackend

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scheduler_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    mission_id TEXT,
    task_id TEXT,
    value TEXT,
    recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_type ON scheduler_metrics(event_type);
CREATE INDEX IF NOT EXISTS idx_metrics_mission ON scheduler_metrics(mission_id);
CREATE INDEX IF NOT EXISTS idx_metrics_recorded ON scheduler_metrics(recorded_at);
"""

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

TASK_DISPATCHED = "task_dispatched"
RESULT_PROCESSED = "result_processed"
LEASE_EXPIRED = "lease_expired"
MISSION_COMPLETED = "mission_completed"
MISSION_FAILED = "mission_failed"


class SchedulerMetrics:
    """Record and query scheduler events.

    Args:
        backend: Shared ``DatabaseBackend``.
    """

    def __init__(self, backend: DatabaseBackend):
        self._backend = backend
        self._init_schema()

    def _init_schema(self) -> None:
        with self._backend.transaction():
            self._backend.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        event_type: str,
        *,
        mission_id: str | None = None,
        task_id: str | None = None,
        value: str | None = None,
    ) -> None:
        """Insert a metric event."""
        with self._backend.transaction():
            self._backend.execute(
                """
                INSERT INTO scheduler_metrics
                    (event_type, mission_id, task_id, value, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    mission_id,
                    task_id,
                    value,
                    datetime.now().isoformat(),
                ),
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def count(self, event_type: str, since: datetime | None = None) -> int:
        """Count occurrences of an event type."""
        if since:
            row = self._backend.fetchone(
                """
                SELECT COUNT(*) AS c FROM scheduler_metrics
                WHERE event_type = ? AND recorded_at >= ?
                """,
                (event_type, since.isoformat()),
            )
        else:
            row = self._backend.fetchone(
                "SELECT COUNT(*) AS c FROM scheduler_metrics WHERE event_type = ?",
                (event_type,),
            )
        return row["c"] if row else 0

    def summary(self) -> dict[str, Any]:
        """Return a summary of all recorded events."""
        rows = self._backend.fetchall(
            """
            SELECT event_type, COUNT(*) AS c FROM scheduler_metrics
            GROUP BY event_type
            """
        )
        return {r["event_type"]: r["c"] for r in rows}

    def by_mission(self, mission_id: str) -> list[dict[str, Any]]:
        """Return all metrics for a specific mission."""
        rows = self._backend.fetchall(
            """
            SELECT event_type, mission_id, task_id, value, recorded_at
            FROM scheduler_metrics
            WHERE mission_id = ?
            ORDER BY recorded_at
            """,
            (mission_id,),
        )
        return [
            {
                "event_type": r["event_type"],
                "mission_id": r["mission_id"],
                "task_id": r["task_id"],
                "value": r["value"],
                "recorded_at": r["recorded_at"],
            }
            for r in rows
        ]

    def reset(self) -> None:
        """Truncate all metrics (useful for testing)."""
        with self._backend.transaction():
            self._backend.execute("DELETE FROM scheduler_metrics")
