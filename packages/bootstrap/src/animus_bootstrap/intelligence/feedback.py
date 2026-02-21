"""Feedback store — persists user thumbs up/down on Animus responses."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class FeedbackStore:
    """SQLite-backed persistent store for user feedback on responses.

    Stores thumbs up/down ratings and optional comments. Used by the
    reflection loop to identify patterns and improve over time.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create feedback table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                message_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT DEFAULT '',
                channel TEXT DEFAULT '',
                timestamp TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def record(
        self,
        message_text: str,
        response_text: str,
        rating: int,
        comment: str = "",
        channel: str = "",
    ) -> str:
        """Record a feedback entry. Rating: 1 = positive, -1 = negative."""
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO feedback
            (id, message_text, response_text, rating, comment, channel, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (feedback_id, message_text, response_text, rating, comment, channel, timestamp),
        )
        self._conn.commit()
        return feedback_id

    def get_recent(self, limit: int = 50) -> list[dict]:
        """Return the most recent feedback entries."""
        cursor = self._conn.execute(
            "SELECT * FROM feedback ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        return [self._row_to_dict(row) for row in cursor]

    def get_positive_patterns(self, limit: int = 20) -> list[dict]:
        """Return recent positive feedback entries."""
        cursor = self._conn.execute(
            "SELECT * FROM feedback WHERE rating > 0 ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_dict(row) for row in cursor]

    def get_negative_patterns(self, limit: int = 20) -> list[dict]:
        """Return recent negative feedback entries."""
        cursor = self._conn.execute(
            "SELECT * FROM feedback WHERE rating < 0 ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_dict(row) for row in cursor]

    def get_stats(self) -> dict:
        """Return feedback statistics."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM feedback")
        total = cursor.fetchone()[0]

        cursor = self._conn.execute("SELECT COUNT(*) FROM feedback WHERE rating > 0")
        positive = cursor.fetchone()[0]

        cursor = self._conn.execute("SELECT COUNT(*) FROM feedback WHERE rating < 0")
        negative = cursor.fetchone()[0]

        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "positive_pct": round(positive / total * 100, 1) if total > 0 else 0,
            "negative_pct": round(negative / total * 100, 1) if total > 0 else 0,
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return {
            "id": row["id"],
            "message_text": row["message_text"],
            "response_text": row["response_text"],
            "rating": row["rating"],
            "comment": row["comment"],
            "channel": row["channel"],
            "timestamp": row["timestamp"],
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
