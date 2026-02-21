"""Persistent timer storage — SQLite-backed store for dynamic timers."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class TimerStore:
    """SQLite-backed persistent store for dynamic timers.

    Survives restarts so timers created via the timer_create tool are
    restored when the runtime boots.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create timers table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS timers (
                name TEXT PRIMARY KEY,
                schedule TEXT NOT NULL,
                action TEXT NOT NULL,
                channels TEXT NOT NULL DEFAULT '[]',
                created TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def save(
        self, name: str, schedule: str, action: str, channels: list[str], created: str
    ) -> None:
        """Insert or replace a timer."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO timers (name, schedule, action, channels, created)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, schedule, action, json.dumps(channels), created),
        )
        self._conn.commit()

    def remove(self, name: str) -> bool:
        """Remove a timer by name. Returns True if a row was deleted."""
        cursor = self._conn.execute("DELETE FROM timers WHERE name = ?", (name,))
        self._conn.commit()
        return cursor.rowcount > 0

    def list_all(self) -> list[dict]:
        """Return all saved timers as dicts."""
        cursor = self._conn.execute(
            "SELECT name, schedule, action, channels, created FROM timers ORDER BY created"
        )
        timers = []
        for row in cursor:
            timers.append(
                {
                    "name": row["name"],
                    "schedule": row["schedule"],
                    "action": row["action"],
                    "channels": json.loads(row["channels"]),
                    "created": row["created"],
                }
            )
        return timers

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
