"""Persistent task storage — SQLite-backed store for local tasks."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskStore:
    """SQLite-backed persistent store for local tasks.

    Supports CRUD operations plus queries for overdue and upcoming tasks.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create tasks table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                priority TEXT NOT NULL DEFAULT 'normal',
                due_date TEXT,
                created TEXT NOT NULL,
                updated TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def create(
        self,
        name: str,
        description: str = "",
        priority: str = "normal",
        due_date: str = "",
    ) -> str:
        """Create a new task. Returns the task ID."""
        task_id = uuid.uuid4().hex[:8]
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO tasks (id, name, description, status, priority, due_date, created, updated)
            VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)
            """,
            (task_id, name, description, priority, due_date or None, now, now),
        )
        self._conn.commit()
        return task_id

    def get(self, task_id: str) -> dict | None:
        """Get a task by ID. Returns None if not found."""
        cursor = self._conn.execute(
            "SELECT id, name, description, status, priority, due_date, created, updated "
            "FROM tasks WHERE id = ?",
            (task_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def list_all(self, status: str = "") -> list[dict]:
        """Return all tasks, optionally filtered by status."""
        if status:
            cursor = self._conn.execute(
                "SELECT id, name, description, status, priority, due_date, created, updated "
                "FROM tasks WHERE status = ? ORDER BY created",
                (status,),
            )
        else:
            cursor = self._conn.execute(
                "SELECT id, name, description, status, priority, due_date, created, updated "
                "FROM tasks ORDER BY created"
            )
        return [dict(row) for row in cursor]

    def complete(self, task_id: str) -> bool:
        """Mark a task as completed. Returns True if found."""
        now = datetime.now(UTC).isoformat()
        cursor = self._conn.execute(
            "UPDATE tasks SET status = 'completed', updated = ? WHERE id = ?",
            (now, task_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def delete(self, task_id: str) -> bool:
        """Delete a task by ID. Returns True if a row was deleted."""
        cursor = self._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def get_overdue(self) -> list[dict]:
        """Return tasks with due_date in the past that are not completed."""
        now = datetime.now(UTC).isoformat()
        cursor = self._conn.execute(
            "SELECT id, name, description, status, priority, due_date, created, updated "
            "FROM tasks WHERE due_date IS NOT NULL AND due_date < ? AND status != 'completed' "
            "ORDER BY due_date",
            (now,),
        )
        return [dict(row) for row in cursor]

    def get_upcoming(self, hours: int = 24) -> list[dict]:
        """Return non-completed tasks due within the next N hours."""
        from datetime import timedelta

        now = datetime.now(UTC)
        cutoff = (now + timedelta(hours=hours)).isoformat()
        now_iso = now.isoformat()
        cursor = self._conn.execute(
            "SELECT id, name, description, status, priority, due_date, created, updated "
            "FROM tasks WHERE due_date IS NOT NULL AND due_date >= ? AND due_date <= ? "
            "AND status != 'completed' ORDER BY due_date",
            (now_iso, cutoff),
        )
        return [dict(row) for row in cursor]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
