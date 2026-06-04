"""SQLite-backed store for Web Push subscriptions."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class PushSubscriptionStore:
    """Persist browser PushSubscription objects (WAL, cross-thread safe)."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS push_subscriptions (
                endpoint TEXT PRIMARY KEY,
                subscription TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        self._conn.commit()

    def add(self, subscription: dict) -> None:
        """Insert or update a subscription keyed by its endpoint."""
        endpoint = subscription.get("endpoint")
        if not endpoint:
            msg = "Subscription missing 'endpoint'"
            raise ValueError(msg)
        self._conn.execute(
            "INSERT OR REPLACE INTO push_subscriptions (endpoint, subscription) VALUES (?, ?)",
            (endpoint, json.dumps(subscription)),
        )
        self._conn.commit()

    def remove(self, endpoint: str) -> None:
        """Delete the subscription with the given endpoint."""
        self._conn.execute(
            "DELETE FROM push_subscriptions WHERE endpoint = ?",
            (endpoint,),
        )
        self._conn.commit()

    def all(self) -> list[dict]:
        """Return all stored subscriptions."""
        cur = self._conn.execute("SELECT subscription FROM push_subscriptions")
        return [json.loads(row["subscription"]) for row in cur.fetchall()]

    def count(self) -> int:
        """Return the number of stored subscriptions."""
        cur = self._conn.execute("SELECT COUNT(*) AS n FROM push_subscriptions")
        return int(cur.fetchone()["n"])

    def close(self) -> None:
        self._conn.close()
