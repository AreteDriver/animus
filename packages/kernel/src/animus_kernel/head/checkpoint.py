"""Session checkpoint persistence for Animus Head.

Stores conversation state, metadata, and statistics to SQLite so sessions
survive process restarts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HeadCheckpoint:
    """A snapshot of a Head session."""

    session_id: str
    started_at: datetime
    last_active_at: datetime
    project_root: str | None = None
    model: str = ""
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    total_tokens: int = 0
    turns: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat(),
            "project_root": self.project_root,
            "model": self.model,
            "messages": self.messages,
            "summary": self.summary,
            "total_tokens": self.total_tokens,
            "turns": self.turns,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HeadCheckpoint:
        return cls(
            session_id=data["session_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            last_active_at=datetime.fromisoformat(data["last_active_at"]),
            project_root=data.get("project_root"),
            model=data.get("model", ""),
            messages=data.get("messages", []),
            summary=data.get("summary", ""),
            total_tokens=data.get("total_tokens", 0),
            turns=data.get("turns", 0),
            metadata=data.get("metadata", {}),
        )


class HeadCheckpointStore:
    """SQLite-backed checkpoint store."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS head_checkpoints (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            last_active_at TEXT NOT NULL,
            project_root TEXT,
            model TEXT NOT NULL DEFAULT '',
            messages_json TEXT NOT NULL DEFAULT '[]',
            summary TEXT NOT NULL DEFAULT '',
            total_tokens INTEGER NOT NULL DEFAULT 0,
            turns INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_head_cp_active
        ON head_checkpoints(last_active_at DESC);
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".animus" / "sessions" / "head.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(self.SCHEMA)

    def save(self, checkpoint: HeadCheckpoint) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO head_checkpoints
                (session_id, started_at, last_active_at, project_root, model,
                 messages_json, summary, total_tokens, turns, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_active_at=excluded.last_active_at,
                    project_root=excluded.project_root,
                    model=excluded.model,
                    messages_json=excluded.messages_json,
                    summary=excluded.summary,
                    total_tokens=excluded.total_tokens,
                    turns=excluded.turns,
                    metadata_json=excluded.metadata_json
                """,
                (
                    checkpoint.session_id,
                    checkpoint.started_at.isoformat(),
                    checkpoint.last_active_at.isoformat(),
                    checkpoint.project_root,
                    checkpoint.model,
                    json.dumps(checkpoint.messages),
                    checkpoint.summary,
                    checkpoint.total_tokens,
                    checkpoint.turns,
                    json.dumps(checkpoint.metadata),
                ),
            )
            conn.commit()
        logger.debug("Checkpoint saved: %s", checkpoint.session_id)

    def load(self, session_id: str) -> HeadCheckpoint | None:
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT * FROM head_checkpoints WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None

        # Map row to checkpoint (column order matches SCHEMA)
        return HeadCheckpoint(
            session_id=row[0],
            started_at=datetime.fromisoformat(row[1]),
            last_active_at=datetime.fromisoformat(row[2]),
            project_root=row[3],
            model=row[4] or "",
            messages=json.loads(row[5]) if row[5] else [],
            summary=row[6] or "",
            total_tokens=row[7] or 0,
            turns=row[8] or 0,
            metadata=json.loads(row[9]) if row[9] else {},
        )

    def list_recent(self, limit: int = 10) -> list[HeadCheckpoint]:
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                """
                SELECT session_id, started_at, last_active_at, project_root, model,
                       messages_json, summary, total_tokens, turns, metadata_json
                FROM head_checkpoints
                ORDER BY last_active_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        checkpoints = []
        for row in rows:
            checkpoints.append(
                HeadCheckpoint(
                    session_id=row[0],
                    started_at=datetime.fromisoformat(row[1]),
                    last_active_at=datetime.fromisoformat(row[2]),
                    project_root=row[3],
                    model=row[4] or "",
                    messages=json.loads(row[5]) if row[5] else [],
                    summary=row[6] or "",
                    total_tokens=row[7] or 0,
                    turns=row[8] or 0,
                    metadata=json.loads(row[9]) if row[9] else {},
                )
            )
        return checkpoints

    def delete(self, session_id: str) -> bool:
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.execute(
                "DELETE FROM head_checkpoints WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            return cur.rowcount > 0
