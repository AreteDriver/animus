"""Session management with SQLite persistence."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from animus_bootstrap.gateway.models import GatewayMessage

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """A conversation session spanning one or more channels."""

    id: str
    user_id: str
    user_name: str
    messages: list[GatewayMessage]
    channel_ids: dict[str, str]  # channel -> platform_user_id
    created_at: datetime
    last_active: datetime
    context_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """Manages gateway sessions backed by SQLite (WAL mode)."""

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
            CREATE TABLE IF NOT EXISTS gateway_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                user_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_active TEXT NOT NULL,
                context_tokens INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS gateway_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                channel_message_id TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                sender_name TEXT NOT NULL,
                text TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                timestamp TEXT NOT NULL,
                reply_to TEXT,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES gateway_sessions(id)
            );

            CREATE TABLE IF NOT EXISTS channel_identities (
                channel TEXT NOT NULL,
                platform_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                PRIMARY KEY (channel, platform_id),
                FOREIGN KEY (session_id) REFERENCES gateway_sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON gateway_messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                ON gateway_messages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_identities_session
                ON channel_identities(session_id);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public async API (delegates to threads for non-blocking I/O)
    # ------------------------------------------------------------------

    async def get_or_create_session(self, message: GatewayMessage) -> Session:
        return await asyncio.to_thread(self._get_or_create_session_sync, message)

    async def add_message(self, session: Session, message: GatewayMessage) -> None:
        await asyncio.to_thread(self._add_message_sync, session, message)

    async def link_channel(self, session_id: str, channel: str, platform_id: str) -> None:
        await asyncio.to_thread(self._link_channel_sync, session_id, channel, platform_id)

    async def get_context(self, session: Session, max_messages: int = 50) -> list[dict[str, str]]:
        return await asyncio.to_thread(self._get_context_sync, session, max_messages)

    async def get_recent_messages(self, limit: int = 50) -> list[GatewayMessage]:
        return await asyncio.to_thread(self._get_recent_messages_sync, limit)

    async def prune_old_sessions(self, max_age_days: int = 30) -> int:
        return await asyncio.to_thread(self._prune_old_sessions_sync, max_age_days)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Synchronous implementations
    # ------------------------------------------------------------------

    def _get_or_create_session_sync(self, message: GatewayMessage) -> Session:
        cur = self._conn.cursor()

        # Look up existing session via channel identity
        cur.execute(
            "SELECT session_id FROM channel_identities WHERE channel = ? AND platform_id = ?",
            (message.channel, message.sender_id),
        )
        row = cur.fetchone()

        if row:
            session_id = row["session_id"]
            return self._load_session_sync(session_id)

        # Create new session
        session_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        cur.execute(
            "INSERT INTO gateway_sessions (id, user_id, user_name, created_at, last_active) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, message.sender_id, message.sender_name, now, now),
        )
        cur.execute(
            "INSERT OR IGNORE INTO channel_identities (channel, platform_id, session_id) "
            "VALUES (?, ?, ?)",
            (message.channel, message.sender_id, session_id),
        )
        self._conn.commit()

        return Session(
            id=session_id,
            user_id=message.sender_id,
            user_name=message.sender_name,
            messages=[],
            channel_ids={message.channel: message.sender_id},
            created_at=datetime.fromisoformat(now),
            last_active=datetime.fromisoformat(now),
        )

    def _load_session_sync(self, session_id: str) -> Session:
        cur = self._conn.cursor()

        cur.execute("SELECT * FROM gateway_sessions WHERE id = ?", (session_id,))
        sess_row = cur.fetchone()
        if not sess_row:
            msg = f"Session {session_id} not found"
            raise ValueError(msg)

        # Load channel identities
        cur.execute(
            "SELECT channel, platform_id FROM channel_identities WHERE session_id = ?",
            (session_id,),
        )
        channel_ids = {r["channel"]: r["platform_id"] for r in cur.fetchall()}

        # Load messages
        cur.execute(
            "SELECT * FROM gateway_messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        )
        messages = [self._row_to_message(r) for r in cur.fetchall()]

        return Session(
            id=sess_row["id"],
            user_id=sess_row["user_id"],
            user_name=sess_row["user_name"],
            messages=messages,
            channel_ids=channel_ids,
            created_at=datetime.fromisoformat(sess_row["created_at"]),
            last_active=datetime.fromisoformat(sess_row["last_active"]),
            context_tokens=sess_row["context_tokens"],
            metadata=json.loads(sess_row["metadata"]),
        )

    def _add_message_sync(self, session: Session, message: GatewayMessage) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO gateway_messages "
            "(id, session_id, channel, channel_message_id, sender_id, sender_name, "
            "text, role, timestamp, reply_to, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                message.id,
                session.id,
                message.channel,
                message.channel_message_id,
                message.sender_id,
                message.sender_name,
                message.text,
                message.role,
                message.timestamp.isoformat(),
                message.reply_to,
                json.dumps(message.metadata),
            ),
        )
        cur.execute(
            "UPDATE gateway_sessions SET last_active = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), session.id),
        )
        self._conn.commit()
        session.messages.append(message)

    def _link_channel_sync(self, session_id: str, channel: str, platform_id: str) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO channel_identities (channel, platform_id, session_id) "
            "VALUES (?, ?, ?)",
            (channel, platform_id, session_id),
        )
        self._conn.commit()

    def _get_context_sync(self, session: Session, max_messages: int = 50) -> list[dict[str, str]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT role, text FROM gateway_messages WHERE session_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (session.id, max_messages),
        )
        rows = cur.fetchall()
        # Reverse to chronological order
        return [{"role": r["role"], "content": r["text"]} for r in reversed(rows)]

    def _get_recent_messages_sync(self, limit: int = 50) -> list[GatewayMessage]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM gateway_messages ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_message(r) for r in cur.fetchall()]

    def _prune_old_sessions_sync(self, max_age_days: int = 30) -> int:
        cur = self._conn.cursor()
        cutoff = datetime.now(UTC)
        # Compute cutoff by subtracting days manually (avoids timedelta import)
        from datetime import timedelta

        cutoff_str = (cutoff - timedelta(days=max_age_days)).isoformat()

        # Find old sessions
        cur.execute(
            "SELECT id FROM gateway_sessions WHERE last_active < ?",
            (cutoff_str,),
        )
        old_ids = [r["id"] for r in cur.fetchall()]

        if not old_ids:
            return 0

        placeholders = ",".join("?" for _ in old_ids)
        cur.execute(f"DELETE FROM gateway_messages WHERE session_id IN ({placeholders})", old_ids)
        cur.execute(f"DELETE FROM channel_identities WHERE session_id IN ({placeholders})", old_ids)
        cur.execute(f"DELETE FROM gateway_sessions WHERE id IN ({placeholders})", old_ids)
        self._conn.commit()

        return len(old_ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> GatewayMessage:
        return GatewayMessage(
            id=row["id"],
            channel=row["channel"],
            channel_message_id=row["channel_message_id"],
            sender_id=row["sender_id"],
            sender_name=row["sender_name"],
            text=row["text"],
            role=row["role"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            reply_to=row["reply_to"],
            metadata=json.loads(row["metadata"]),
        )
