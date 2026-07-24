"""Cognitive Event Ledger — SQLite-backed append-only store.

Guarantees:
- Atomic writes (event + chain hash committed in a single transaction).
- Sequential chain_hash computation (always based on the previous entry).
- Deterministic JSON serialization for hash inputs.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from animus_bootstrap.ledger.models import EventType, IntegrityChain, LedgerEntry, LedgerEvent


class LedgerStore:
    """Thread-safe SQLite-backed event store with integrity chaining.

    Usage::

        store = LedgerStore(Path("./ledger.db"))
        event = LedgerEvent(...)
        entry = store.append(event)

        # Query by event id
        entry = store.get_by_event_id("evt-001")

        # Query recent events for an object
        entries = store.query(object_id="obj-001", limit=50)

        # Verify chain integrity
        assert store.verify_chain()
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        if db_path is not None:
            self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Ensure the SQLite table and indexes exist."""
        if self._db_path is None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    object_version INTEGER NOT NULL,
                    principal TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    integrity_hash TEXT NOT NULL,
                    tx_time TEXT NOT NULL,
                    parent_event_id TEXT,
                    chain_hash TEXT NOT NULL,
                    FOREIGN KEY (parent_event_id) REFERENCES ledger_events(event_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_object ON ledger_events(object_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_type ON ledger_events(event_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_workspace ON ledger_events(workspace_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_time ON ledger_events(tx_time)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, event: LedgerEvent) -> LedgerEntry:
        """Append an event atomically with chain-hash computation.

        The chain hash is computed per-object, so each object's history
        forms an independently verifiable integrity chain.

        Raises:
            sqlite3.IntegrityError: If ``event_id`` already exists.
        """
        if self._db_path is None:
            raise RuntimeError("LedgerStore is not backed by a database")

        with self._lock:
            prev_hash = self._get_last_chain_hash_for_object(event.object_id)
            chain_hash = IntegrityChain.compute_chain_hash(event, prev_hash)
            entry = LedgerEntry(
                **event.model_dump(mode="json"),
                chain_hash=chain_hash,
            )
            db_id = self._insert(entry)
            entry.db_id = db_id
            return entry

    def get_by_event_id(self, event_id: str) -> LedgerEntry | None:
        """Retrieve a single entry by its canonical ``event_id``."""
        if self._db_path is None:
            return None
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT * FROM ledger_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def query(
        self,
        object_id: str | None = None,
        event_type: EventType | None = None,
        workspace_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LedgerEntry]:
        """Query entries with optional filters."""
        if self._db_path is None:
            return []

        conditions: list[str] = []
        params: list[Any] = []

        if object_id is not None:
            conditions.append("object_id = ?")
            params.append(object_id)
        if event_type is not None:
            conditions.append("event_type = ?")
            params.append(event_type.value)
        if workspace_id is not None:
            conditions.append("workspace_id = ?")
            params.append(workspace_id)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT * FROM ledger_events
            {where_clause}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_entry(dict(r)) for r in rows]

    def get_chain(self, object_id: str) -> list[LedgerEntry]:
        """Return all entries for *object_id* in chronological order."""
        if self._db_path is None:
            return []
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM ledger_events WHERE object_id = ? ORDER BY id ASC",
                (object_id,),
            ).fetchall()
        return [self._row_to_entry(dict(r)) for r in rows]

    def verify_chain(self, object_id: str | None = None) -> bool:
        """Verify integrity chain for all entries, or a single object."""
        if self._db_path is None:
            return True
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if object_id is not None:
                rows = conn.execute(
                    "SELECT * FROM ledger_events WHERE object_id = ? ORDER BY id ASC",
                    (object_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM ledger_events ORDER BY id ASC"
                ).fetchall()
        entries = [self._row_to_entry(dict(r)) for r in rows]
        return IntegrityChain.verify(entries)

    def get_last_event_id(self) -> str | None:
        """Return the ``event_id`` of the most recently appended entry."""
        if self._db_path is None:
            return None
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT event_id FROM ledger_events ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def count(self, object_id: str | None = None) -> int:
        """Return total entry count, optionally filtered by object."""
        if self._db_path is None:
            return 0
        with sqlite3.connect(str(self._db_path)) as conn:
            if object_id is not None:
                row = conn.execute(
                    "SELECT COUNT(*) FROM ledger_events WHERE object_id = ?",
                    (object_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM ledger_events").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_last_chain_hash_for_object(self, object_id: str) -> str | None:
        """Fetch the ``chain_hash`` of the most recent entry for *object_id*."""
        if self._db_path is None:
            return None
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT chain_hash FROM ledger_events WHERE object_id = ? ORDER BY id DESC LIMIT 1",
                (object_id,),
            ).fetchone()
        return row[0] if row else None

    def _insert(self, entry: LedgerEntry) -> int:
        """Insert an entry into SQLite and return the row id."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                """
                INSERT INTO ledger_events (
                    event_id, event_type, object_id, object_version,
                    principal, workspace_id, payload, integrity_hash,
                    tx_time, parent_event_id, chain_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.event_id,
                    entry.event_type.value,
                    entry.object_id,
                    entry.object_version,
                    entry.principal,
                    entry.workspace_id,
                    json.dumps(entry.payload, sort_keys=True),
                    entry.integrity_hash,
                    entry.tx_time.isoformat(),
                    entry.parent_event_id,
                    entry.chain_hash,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def _row_to_entry(row: dict[str, Any] | sqlite3.Row | tuple) -> LedgerEntry:
        """Convert a SQLite row to :class:`LedgerEntry`.

        Accepts dict, sqlite3.Row, or plain tuple (column order must match
        the SELECT statements in this module).
        """
        if isinstance(row, (dict, sqlite3.Row)):
            return LedgerEntry(
                db_id=row["id"],
                event_id=row["event_id"],
                event_type=EventType(row["event_type"]),
                object_id=row["object_id"],
                object_version=row["object_version"],
                principal=row["principal"],
                workspace_id=row["workspace_id"],
                payload=json.loads(row["payload"]),
                integrity_hash=row["integrity_hash"],
                tx_time=row["tx_time"],
                parent_event_id=row["parent_event_id"],
                chain_hash=row["chain_hash"],
            )
        # Fallback for plain tuple (column order: id, event_id, event_type, ...)
        return LedgerEntry(
            db_id=row[0],
            event_id=row[1],
            event_type=EventType(row[2]),
            object_id=row[3],
            object_version=row[4],
            principal=row[5],
            workspace_id=row[6],
            payload=json.loads(row[7]),
            integrity_hash=row[8],
            tx_time=row[9],
            parent_event_id=row[10],
            chain_hash=row[11],
        )
