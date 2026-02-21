"""Persistent tool execution history — SQLite-backed store."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ToolHistoryStore:
    """SQLite-backed persistent store for tool execution history.

    Complements the in-memory history in ToolExecutor so that execution
    records survive restarts.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create the history table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_history (
                id TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                success INTEGER NOT NULL,
                output TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                timestamp TEXT NOT NULL,
                arguments TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_history_ts
            ON tool_history (timestamp)
        """)
        self._conn.commit()

    def save(self, result: object) -> None:
        """Persist a ToolResult to the database."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO tool_history
                (id, tool_name, success, output, duration_ms, timestamp, arguments)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.id,
                result.tool_name,
                1 if result.success else 0,
                result.output,
                result.duration_ms,
                result.timestamp.isoformat(),
                json.dumps(result.arguments),
            ),
        )
        self._conn.commit()

    def list_recent(self, limit: int = 50) -> list[dict]:
        """Return the most recent history entries."""
        cursor = self._conn.execute(
            "SELECT * FROM tool_history ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = [self._row_to_dict(row) for row in cursor]
        rows.reverse()  # oldest first, matching in-memory convention
        return rows

    def count(self) -> int:
        """Return total number of history entries."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM tool_history")
        return cursor.fetchone()[0]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return {
            "id": row["id"],
            "tool_name": row["tool_name"],
            "success": bool(row["success"]),
            "output": row["output"],
            "duration_ms": row["duration_ms"],
            "timestamp": row["timestamp"],
            "arguments": json.loads(row["arguments"]),
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
