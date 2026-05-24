"""Recall log — records every memory retrieval and which memories were returned."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecallRecord:
    """A logged memory recall event."""

    id: str
    query: str
    recalled_at: datetime
    context_hint: str
    memory_ids: tuple[str, ...]


class RecallLog:
    """SQLite-backed log of memory recalls and the memory IDs they returned.

    Mirrors the ProactiveEngine + OutcomeStore split: a recall is a unit that
    can later receive outcome signals (implicit action follow-on or explicit
    feedback), which then propagate back to the constituent memory_ids.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS recalls (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                recalled_at TEXT NOT NULL,
                context_hint TEXT NOT NULL DEFAULT ''
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_recalls_at ON recalls(recalled_at)")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS recall_memories (
                recall_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                rank INTEGER NOT NULL,
                memory_type TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (recall_id, memory_id)
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_recall_memories_memory ON recall_memories(memory_id)"
        )
        self._conn.commit()

    def log_recall(
        self,
        query: str,
        memory_entries: list[tuple[str, str]],
        context_hint: str = "",
        recalled_at: datetime | None = None,
    ) -> str:
        """Log a recall and the (memory_id, memory_type) tuples it returned.

        Returns the new recall id. Empty memory_entries is allowed — the recall
        is still recorded so silent retrievals can be measured.
        """
        recall_id = str(uuid.uuid4())
        ts = (recalled_at or datetime.now(UTC)).isoformat()
        self._conn.execute(
            "INSERT INTO recalls (id, query, recalled_at, context_hint) VALUES (?, ?, ?, ?)",
            (recall_id, query, ts, context_hint),
        )
        seen: set[str] = set()
        for rank, (memory_id, memory_type) in enumerate(memory_entries):
            if not memory_id or memory_id in seen:
                continue
            seen.add(memory_id)
            self._conn.execute(
                "INSERT INTO recall_memories (recall_id, memory_id, rank, memory_type) "
                "VALUES (?, ?, ?, ?)",
                (recall_id, memory_id, rank, memory_type or ""),
            )
        self._conn.commit()
        return recall_id

    def get_recall(self, recall_id: str) -> RecallRecord | None:
        cursor = self._conn.execute(
            "SELECT id, query, recalled_at, context_hint FROM recalls WHERE id = ?",
            (recall_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        memory_cursor = self._conn.execute(
            "SELECT memory_id FROM recall_memories WHERE recall_id = ? ORDER BY rank ASC",
            (recall_id,),
        )
        memory_ids = tuple(r["memory_id"] for r in memory_cursor.fetchall())
        return RecallRecord(
            id=row["id"],
            query=row["query"],
            recalled_at=datetime.fromisoformat(row["recalled_at"]),
            context_hint=row["context_hint"] or "",
            memory_ids=memory_ids,
        )

    def memory_ids_for(self, recall_id: str) -> list[str]:
        cursor = self._conn.execute(
            "SELECT memory_id FROM recall_memories WHERE recall_id = ? ORDER BY rank ASC",
            (recall_id,),
        )
        return [row["memory_id"] for row in cursor.fetchall()]

    def recall_count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) AS n FROM recalls")
        return int(cursor.fetchone()["n"])

    def close(self) -> None:
        self._conn.close()
