"""Persistent improvement storage — SQLite-backed store for improvement proposals."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class ImprovementStore:
    """SQLite-backed persistent store for self-improvement proposals.

    Survives restarts so the improvement audit trail persists across sessions.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create improvements table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS improvements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                area TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'proposed',
                timestamp TEXT NOT NULL,
                analysis TEXT,
                patch TEXT,
                applied_at TEXT
            )
        """)
        self._conn.commit()

    def save(self, proposal: dict) -> int:
        """Insert a proposal and return its assigned ID."""
        cursor = self._conn.execute(
            """
            INSERT INTO improvements (area, description, status, timestamp, analysis, patch)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                proposal["area"],
                proposal["description"],
                proposal.get("status", "proposed"),
                proposal["timestamp"],
                proposal.get("analysis"),
                proposal.get("patch"),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def update_status(
        self, proposal_id: int, status: str, applied_at: str | None = None
    ) -> bool:
        """Update a proposal's status. Returns True if a row was updated."""
        if applied_at:
            cursor = self._conn.execute(
                "UPDATE improvements SET status = ?, applied_at = ? WHERE id = ?",
                (status, applied_at, proposal_id),
            )
        else:
            cursor = self._conn.execute(
                "UPDATE improvements SET status = ? WHERE id = ?",
                (status, proposal_id),
            )
        self._conn.commit()
        return cursor.rowcount > 0

    def update_analysis(self, proposal_id: int, analysis: str) -> bool:
        """Update a proposal's analysis text. Returns True if a row was updated."""
        cursor = self._conn.execute(
            "UPDATE improvements SET analysis = ? WHERE id = ?",
            (analysis, proposal_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get(self, proposal_id: int) -> dict | None:
        """Get a single proposal by ID."""
        cursor = self._conn.execute(
            "SELECT * FROM improvements WHERE id = ?", (proposal_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_all(self, status: str = "all") -> list[dict]:
        """Return proposals, optionally filtered by status."""
        if status == "all":
            cursor = self._conn.execute(
                "SELECT * FROM improvements ORDER BY id"
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM improvements WHERE status = ? ORDER BY id",
                (status,),
            )
        return [self._row_to_dict(row) for row in cursor]

    def next_id(self) -> int:
        """Return the next auto-increment ID (for preview before insert)."""
        cursor = self._conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'improvements'"
        )
        row = cursor.fetchone()
        return (row[0] + 1) if row else 1

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return {
            "id": row["id"],
            "area": row["area"],
            "description": row["description"],
            "status": row["status"],
            "timestamp": row["timestamp"],
            "analysis": row["analysis"],
            "patch": row["patch"],
            "applied_at": row["applied_at"],
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
