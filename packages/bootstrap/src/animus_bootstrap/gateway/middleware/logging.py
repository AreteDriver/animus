"""Message logging middleware — SQLite-backed audit log."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from animus_bootstrap.gateway.models import GatewayMessage, GatewayResponse

logger = logging.getLogger(__name__)


class MessageLogger:
    """Persist inbound and outbound gateway messages to SQLite.

    Follows the same WAL-mode SQLite pattern used by
    :class:`~animus_bootstrap.gateway.session.SessionManager`.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS gateway_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                channel TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                sender_name TEXT NOT NULL,
                direction TEXT NOT NULL,
                text TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_log_channel
                ON gateway_log(channel);
            CREATE INDEX IF NOT EXISTS idx_log_timestamp
                ON gateway_log(timestamp);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_inbound(self, message: GatewayMessage) -> None:
        """Record an inbound (user -> gateway) message."""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO gateway_log "
            "(id, timestamp, channel, sender_id, sender_name, direction, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                message.id,
                message.timestamp.isoformat(),
                message.channel,
                message.sender_id,
                message.sender_name,
                "inbound",
                message.text,
            ),
        )
        self._conn.commit()

    def log_outbound(self, response: GatewayResponse, channel: str) -> None:
        """Record an outbound (gateway -> channel) response."""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO gateway_log "
            "(id, timestamp, channel, sender_id, sender_name, direction, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                datetime.now(UTC).isoformat(),
                channel,
                "animus",
                "Animus",
                "outbound",
                response.text,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_logs(
        self,
        channel: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, str]]:
        """Return recent log entries, optionally filtered by *channel*.

        Results are ordered newest-first.
        """
        cur = self._conn.cursor()
        if channel is not None:
            cur.execute(
                "SELECT * FROM gateway_log WHERE channel = ? ORDER BY timestamp DESC LIMIT ?",
                (channel, limit),
            )
        else:
            cur.execute(
                "SELECT * FROM gateway_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in cur.fetchall()]

    def clear_logs(self, before: datetime | None = None) -> None:
        """Delete log entries.

        If *before* is given only entries older than that timestamp are
        removed; otherwise **all** entries are deleted.
        """
        cur = self._conn.cursor()
        if before is not None:
            cur.execute(
                "DELETE FROM gateway_log WHERE timestamp < ?",
                (before.isoformat(),),
            )
        else:
            cur.execute("DELETE FROM gateway_log")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
