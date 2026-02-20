"""SQLite-backed signal backend for cross-process coordination.

Delivers cross-process signal consumption with ACID guarantees, zero new
dependencies, and the same WAL pattern used by ScoreStore and StigmergyField.

Each consumer independently tracks which signals it has processed. Multiple
processes sharing the same ``.signals.db`` file can publish and consume
signals concurrently.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

from convergent.protocol import Signal

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_type TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    target_agent TEXT,
    payload TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type);
CREATE INDEX IF NOT EXISTS idx_signals_source ON signals(source_agent);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);

CREATE TABLE IF NOT EXISTS signal_consumers (
    consumer_id TEXT NOT NULL,
    signal_id INTEGER NOT NULL,
    processed_at TEXT NOT NULL,
    PRIMARY KEY (consumer_id, signal_id)
);
"""


class SQLiteSignalBackend:
    """Signal backend backed by SQLite for cross-process coordination.

    Uses WAL mode for concurrent reads and ``check_same_thread=False``
    for thread-safe access — same pattern as ScoreStore and StigmergyField.

    Args:
        db_path: Path to SQLite database file, or ``:memory:`` for in-memory.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def store_signal(self, signal: Signal) -> None:
        """Store a signal in the database."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO signals "
            "(signal_type, source_agent, target_agent, payload, timestamp, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                signal.signal_type,
                signal.source_agent,
                signal.target_agent,
                signal.payload,
                signal.timestamp,
                now,
            ),
        )
        self._conn.commit()
        logger.info(
            "Stored signal %s from %s (target=%s)",
            signal.signal_type,
            signal.source_agent,
            signal.target_agent or "broadcast",
        )

    def get_unprocessed(self, consumer_id: str) -> list[tuple[str, Signal]]:
        """Return signals not yet processed by this consumer.

        Uses a LEFT JOIN to find signals without a matching entry in
        signal_consumers for this consumer_id.

        Returns:
            List of (str(row_id), Signal) tuples.
        """
        cursor = self._conn.execute(
            "SELECT s.id, s.signal_type, s.source_agent, s.target_agent, "
            "s.payload, s.timestamp "
            "FROM signals s "
            "LEFT JOIN signal_consumers sc ON s.id = sc.signal_id AND sc.consumer_id = ? "
            "WHERE sc.signal_id IS NULL "
            "ORDER BY s.id ASC",
            (consumer_id,),
        )
        results: list[tuple[str, Signal]] = []
        for row in cursor:
            signal = Signal(
                signal_type=row["signal_type"],
                source_agent=row["source_agent"],
                target_agent=row["target_agent"],
                payload=row["payload"],
                timestamp=row["timestamp"],
            )
            results.append((str(row["id"]), signal))
        return results

    def mark_processed(self, consumer_id: str, signal_ids: list[str]) -> None:
        """Mark signals as processed by a consumer."""
        now = datetime.now(timezone.utc).isoformat()
        for sid in signal_ids:
            self._conn.execute(
                "INSERT OR IGNORE INTO signal_consumers (consumer_id, signal_id, processed_at) "
                "VALUES (?, ?, ?)",
                (consumer_id, int(sid), now),
            )
        self._conn.commit()

    def get_signals(
        self,
        signal_type: str | None = None,
        since: datetime | None = None,
        source_agent: str | None = None,
    ) -> list[Signal]:
        """Query signals with optional filters."""
        clauses: list[str] = []
        params: list[str] = []

        if signal_type is not None:
            clauses.append("signal_type = ?")
            params.append(signal_type)
        if source_agent is not None:
            clauses.append("source_agent = ?")
            params.append(source_agent)
        if since is not None:
            since_iso = since.isoformat()
            clauses.append("timestamp > ?")
            params.append(since_iso)

        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)

        cursor = self._conn.execute(
            f"SELECT signal_type, source_agent, target_agent, payload, timestamp "  # noqa: S608
            f"FROM signals {where} ORDER BY timestamp ASC",
            params,
        )
        return [
            Signal(
                signal_type=row["signal_type"],
                source_agent=row["source_agent"],
                target_agent=row["target_agent"],
                payload=row["payload"],
                timestamp=row["timestamp"],
            )
            for row in cursor
        ]

    def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        """Remove signals older than max_age_seconds.

        Cascades to signal_consumers to maintain referential integrity.
        """
        cutoff = datetime.now(timezone.utc)
        # Calculate cutoff ISO timestamp
        from datetime import timedelta

        cutoff_ts = (cutoff - timedelta(seconds=max_age_seconds)).isoformat()

        # Find expired signal IDs
        cursor = self._conn.execute("SELECT id FROM signals WHERE timestamp <= ?", (cutoff_ts,))
        expired_ids = [row["id"] for row in cursor]

        if not expired_ids:
            return 0

        placeholders = ",".join("?" * len(expired_ids))
        self._conn.execute(
            f"DELETE FROM signal_consumers WHERE signal_id IN ({placeholders})",  # noqa: S608
            expired_ids,
        )
        self._conn.execute(
            f"DELETE FROM signals WHERE id IN ({placeholders})",  # noqa: S608
            expired_ids,
        )
        self._conn.commit()

        logger.info("Cleaned up %d expired signals", len(expired_ids))
        return len(expired_ids)

    def clear(self) -> int:
        """Remove all signals and consumer records."""
        cursor = self._conn.execute("SELECT COUNT(*) as cnt FROM signals")
        count = cursor.fetchone()["cnt"]
        self._conn.execute("DELETE FROM signal_consumers")
        self._conn.execute("DELETE FROM signals")
        self._conn.commit()
        return count

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @property
    def db_path(self) -> str:
        """The database path."""
        return self._db_path
