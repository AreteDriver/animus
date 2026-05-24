"""Outcome store — records nudge outcomes and check fire history for calibration."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


SignalType = str  # "explicit" | "implicit_action"
FireResult = str  # "produced" | "silent" | "error"


@dataclass(frozen=True)
class OutcomeRecord:
    """A single outcome signal recorded against a delivered nudge."""

    id: str
    nudge_id: str
    signal_type: SignalType
    signal_value: float
    captured_at: datetime
    source: str


@dataclass(frozen=True)
class FireRecord:
    """A single scheduled tick of a proactive check."""

    id: str
    check_name: str
    fired_at: datetime
    result: FireResult


class OutcomeStore:
    """SQLite-backed store for nudge outcomes and check fire history.

    Two tables:
    - nudge_outcomes: 1:N — multiple signals per nudge (explicit thumbs + implicit actions).
    - check_fires: every scheduled tick. Captures non-firing rate for calibration.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS nudge_outcomes (
                id TEXT PRIMARY KEY,
                nudge_id TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                signal_value REAL NOT NULL,
                captured_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT ''
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_outcomes_nudge ON nudge_outcomes(nudge_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_outcomes_captured ON nudge_outcomes(captured_at)"
        )
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS check_fires (
                id TEXT PRIMARY KEY,
                check_name TEXT NOT NULL,
                fired_at TEXT NOT NULL,
                result TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_fires_check ON check_fires(check_name)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_fires_at ON check_fires(fired_at)")
        self._conn.commit()

    def record_outcome(
        self,
        nudge_id: str,
        signal_type: SignalType,
        signal_value: float,
        source: str = "",
        captured_at: datetime | None = None,
    ) -> str:
        """Record an outcome signal for a nudge. Returns the new outcome id."""
        outcome_id = str(uuid.uuid4())
        ts = (captured_at or datetime.now(UTC)).isoformat()
        self._conn.execute(
            """
            INSERT INTO nudge_outcomes
                (id, nudge_id, signal_type, signal_value, captured_at, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (outcome_id, nudge_id, signal_type, float(signal_value), ts, source),
        )
        self._conn.commit()
        return outcome_id

    def record_fire(
        self,
        check_name: str,
        result: FireResult,
        fired_at: datetime | None = None,
    ) -> str:
        """Record a scheduled tick. Returns the new fire id."""
        fire_id = str(uuid.uuid4())
        ts = (fired_at or datetime.now(UTC)).isoformat()
        self._conn.execute(
            "INSERT INTO check_fires (id, check_name, fired_at, result) VALUES (?, ?, ?, ?)",
            (fire_id, check_name, ts, result),
        )
        self._conn.commit()
        return fire_id

    def outcomes_for(self, nudge_id: str) -> list[OutcomeRecord]:
        """Return all outcome signals recorded for a nudge."""
        cursor = self._conn.execute(
            "SELECT * FROM nudge_outcomes WHERE nudge_id = ? ORDER BY captured_at ASC",
            (nudge_id,),
        )
        return [self._row_to_outcome(row) for row in cursor.fetchall()]

    def fires_for(
        self,
        check_name: str,
        since: datetime | None = None,
    ) -> list[FireRecord]:
        """Return fire records for a check, optionally filtered by time."""
        if since is not None:
            cursor = self._conn.execute(
                "SELECT * FROM check_fires WHERE check_name = ? AND fired_at >= ? "
                "ORDER BY fired_at ASC",
                (check_name, since.isoformat()),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM check_fires WHERE check_name = ? ORDER BY fired_at ASC",
                (check_name,),
            )
        return [self._row_to_fire(row) for row in cursor.fetchall()]

    def fire_counts(
        self,
        check_name: str,
        window_days: int,
        now: datetime | None = None,
    ) -> dict[str, int]:
        """Return counts of {produced, silent, error, total} for a check in the window."""
        anchor = now or datetime.now(UTC)
        since = (anchor - timedelta(days=window_days)).isoformat()
        cursor = self._conn.execute(
            "SELECT result, COUNT(*) as n FROM check_fires "
            "WHERE check_name = ? AND fired_at >= ? GROUP BY result",
            (check_name, since),
        )
        counts = {"produced": 0, "silent": 0, "error": 0}
        for row in cursor.fetchall():
            counts[row["result"]] = row["n"]
        counts["total"] = sum(counts.values())
        return counts

    def close(self) -> None:
        """Close the DB connection."""
        self._conn.close()

    @staticmethod
    def _row_to_outcome(row: sqlite3.Row) -> OutcomeRecord:
        return OutcomeRecord(
            id=row["id"],
            nudge_id=row["nudge_id"],
            signal_type=row["signal_type"],
            signal_value=float(row["signal_value"]),
            captured_at=datetime.fromisoformat(row["captured_at"]),
            source=row["source"] or "",
        )

    @staticmethod
    def _row_to_fire(row: sqlite3.Row) -> FireRecord:
        return FireRecord(
            id=row["id"],
            check_name=row["check_name"],
            fired_at=datetime.fromisoformat(row["fired_at"]),
            result=row["result"],
        )
