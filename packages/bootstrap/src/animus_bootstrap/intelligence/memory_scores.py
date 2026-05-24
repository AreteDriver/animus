"""Memory score store — per-memory cumulative outcome with recency decay."""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_HALF_LIFE_DAYS = 30.0
"""Half-life for the recency multiplier on stored scores. After 30 days a memory's
contribution is halved when read."""


@dataclass(frozen=True)
class MemoryScore:
    """Aggregated outcome score for one memory."""

    memory_id: str
    n_recalls: int
    sum_outcome: float
    last_updated: datetime

    @property
    def raw_mean(self) -> float:
        return self.sum_outcome / self.n_recalls if self.n_recalls > 0 else 0.0


class MemoryScoreStore:
    """SQLite-backed cumulative scoring of individual memories.

    A memory's score is the rolling mean of outcome signals from recalls it
    appeared in. ``effective_score`` applies a recency multiplier so memories
    that haven't been touched in a long time fade toward neutral.
    """

    def __init__(
        self,
        db_path: Path | str,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> None:
        self._half_life_days = float(half_life_days)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_scores (
                memory_id TEXT PRIMARY KEY,
                n_recalls INTEGER NOT NULL DEFAULT 0,
                sum_outcome REAL NOT NULL DEFAULT 0.0,
                last_updated TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def record_outcome(self, memory_id: str, signal_value: float) -> None:
        """Add one outcome signal for a memory. Creates the row if missing."""
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO memory_scores (memory_id, n_recalls, sum_outcome, last_updated)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                n_recalls = n_recalls + 1,
                sum_outcome = sum_outcome + excluded.sum_outcome,
                last_updated = excluded.last_updated
            """,
            (memory_id, float(signal_value), now),
        )
        self._conn.commit()

    def get(self, memory_id: str) -> MemoryScore | None:
        cursor = self._conn.execute(
            "SELECT memory_id, n_recalls, sum_outcome, last_updated "
            "FROM memory_scores WHERE memory_id = ?",
            (memory_id,),
        )
        row = cursor.fetchone()
        return self._row_to_score(row) if row else None

    def effective_score(
        self,
        memory_id: str,
        now: datetime | None = None,
    ) -> float:
        """Recency-decayed mean outcome for a memory. 0.0 if unknown.

        The raw mean lives in [-1, 1]. The decay multiplier ∈ (0, 1] shrinks
        the magnitude — never flips sign — based on time since last update.
        """
        score = self.get(memory_id)
        if score is None or score.n_recalls == 0:
            return 0.0
        anchor = now or datetime.now(UTC)
        days_since = max(0.0, (anchor - score.last_updated).total_seconds() / 86400.0)
        if self._half_life_days <= 0:
            decay = 1.0
        else:
            decay = math.pow(0.5, days_since / self._half_life_days)
        return score.raw_mean * decay

    def top_memories(self, limit: int = 20) -> list[MemoryScore]:
        cursor = self._conn.execute(
            "SELECT memory_id, n_recalls, sum_outcome, last_updated FROM memory_scores "
            "WHERE n_recalls > 0 "
            "ORDER BY (sum_outcome / n_recalls) DESC, n_recalls DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_score(row) for row in cursor.fetchall()]

    def bottom_memories(self, limit: int = 20) -> list[MemoryScore]:
        cursor = self._conn.execute(
            "SELECT memory_id, n_recalls, sum_outcome, last_updated FROM memory_scores "
            "WHERE n_recalls > 0 "
            "ORDER BY (sum_outcome / n_recalls) ASC, n_recalls DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_score(row) for row in cursor.fetchall()]

    def stats(self) -> dict:
        cursor = self._conn.execute(
            "SELECT COUNT(*) AS scored, SUM(n_recalls) AS recalls, "
            "AVG(sum_outcome / n_recalls) AS mean_score "
            "FROM memory_scores WHERE n_recalls > 0"
        )
        row = cursor.fetchone()
        return {
            "scored_memories": int(row["scored"] or 0),
            "total_recalls_attributed": int(row["recalls"] or 0),
            "mean_score": float(row["mean_score"] or 0.0),
        }

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_score(row: sqlite3.Row) -> MemoryScore:
        return MemoryScore(
            memory_id=row["memory_id"],
            n_recalls=int(row["n_recalls"]),
            sum_outcome=float(row["sum_outcome"]),
            last_updated=datetime.fromisoformat(row["last_updated"]),
        )
