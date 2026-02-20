"""SQLite-backed intent graph backend.

Provides persistent storage using stdlib sqlite3. Mirrors the Rust schema
design with a denormalized intent_interfaces table for efficient overlap queries.
"""

from __future__ import annotations

import json
import logging
import sqlite3

from convergent._serialization import (
    constraint_to_dict,
    evidence_to_dict,
    row_to_intent,
    spec_to_dict,
)
from convergent.intent import Intent, InterfaceSpec
from convergent.matching import normalize_name

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS intents (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    intent TEXT NOT NULL,
    provides TEXT NOT NULL,
    requires TEXT NOT NULL,
    constraints TEXT NOT NULL,
    evidence TEXT NOT NULL,
    stability REAL NOT NULL,
    parent_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_intents_agent ON intents(agent_id);

CREATE TABLE IF NOT EXISTS intent_interfaces (
    intent_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    role TEXT NOT NULL,
    tags TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ifaces_name ON intent_interfaces(normalized_name);
CREATE INDEX IF NOT EXISTS idx_ifaces_agent ON intent_interfaces(agent_id);
"""


class SQLiteBackend:
    """Persistent intent graph backed by SQLite.

    Uses WAL mode for concurrent reads and a denormalized intent_interfaces
    table for efficient find_overlapping() queries.

    Args:
        db_path: Path to SQLite database file, or ":memory:" for in-memory.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def publish(self, intent: Intent) -> float:
        """Publish an intent and return its computed stability."""
        stability = intent.compute_stability()

        self._conn.execute(
            "INSERT OR REPLACE INTO intents "
            "(id, agent_id, timestamp, intent, provides, requires, "
            "constraints, evidence, stability, parent_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                intent.id,
                intent.agent_id,
                intent.timestamp.isoformat(),
                intent.intent,
                json.dumps([spec_to_dict(s) for s in intent.provides]),
                json.dumps([spec_to_dict(s) for s in intent.requires]),
                json.dumps([constraint_to_dict(c) for c in intent.constraints]),
                json.dumps([evidence_to_dict(e) for e in intent.evidence]),
                stability,
                intent.parent_id,
            ),
        )

        # Populate denormalized interface lookup
        self._conn.execute(
            "DELETE FROM intent_interfaces WHERE intent_id = ?",
            (intent.id,),
        )
        for spec in intent.provides:
            self._conn.execute(
                "INSERT INTO intent_interfaces "
                "(intent_id, agent_id, normalized_name, role, tags) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    intent.id,
                    intent.agent_id,
                    normalize_name(spec.name),
                    "provides",
                    json.dumps(spec.tags),
                ),
            )
        for spec in intent.requires:
            self._conn.execute(
                "INSERT INTO intent_interfaces "
                "(intent_id, agent_id, normalized_name, role, tags) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    intent.id,
                    intent.agent_id,
                    normalize_name(spec.name),
                    "requires",
                    json.dumps(spec.tags),
                ),
            )

        self._conn.commit()

        logger.debug(
            "Published intent '%s' from %s (stability: %.2f)",
            intent.intent,
            intent.agent_id,
            stability,
        )
        return stability

    def query_all(self, min_stability: float | None = None) -> list[Intent]:
        """Query all intents, optionally filtered by minimum stability."""
        min_stab = min_stability or 0.0
        rows = self._conn.execute(
            "SELECT * FROM intents WHERE stability >= ?",
            (min_stab,),
        ).fetchall()
        return [row_to_intent(r) for r in rows]

    def query_by_agent(self, agent_id: str) -> list[Intent]:
        """Query intents published by a specific agent."""
        rows = self._conn.execute(
            "SELECT * FROM intents WHERE agent_id = ?",
            (agent_id,),
        ).fetchall()
        return [row_to_intent(r) for r in rows]

    def find_overlapping(
        self,
        specs: list[InterfaceSpec],
        exclude_agent: str,
        min_stability: float,
    ) -> list[Intent]:
        """Find intents with overlapping interfaces.

        Two-phase: SQL indexed lookup on intent_interfaces for candidate
        intent IDs, then Python structurally_overlaps() validation.
        """
        if not specs:
            return []

        # Phase 1: SQL candidate lookup by normalized name or shared tags
        normalized_names = [normalize_name(s.name) for s in specs]
        all_tags: set[str] = set()
        for s in specs:
            all_tags.update(s.tags)

        placeholders = ",".join("?" * len(normalized_names))
        candidate_ids: set[str] = set()

        # Name-based candidates
        rows = self._conn.execute(
            f"SELECT DISTINCT intent_id FROM intent_interfaces "
            f"WHERE agent_id != ? AND normalized_name IN ({placeholders})",
            (exclude_agent, *normalized_names),
        ).fetchall()
        candidate_ids.update(r["intent_id"] for r in rows)

        # Tag-based candidates (intents with any matching tag)
        if all_tags:
            tag_rows = self._conn.execute(
                "SELECT DISTINCT intent_id, tags FROM intent_interfaces WHERE agent_id != ?",
                (exclude_agent,),
            ).fetchall()
            for r in tag_rows:
                their_tags = set(json.loads(r["tags"]))
                if len(their_tags & all_tags) >= 2:
                    candidate_ids.add(r["intent_id"])

        if not candidate_ids:
            return []

        # Phase 2: Load candidates and validate with structurally_overlaps
        placeholders = ",".join("?" * len(candidate_ids))
        rows = self._conn.execute(
            f"SELECT * FROM intents WHERE id IN ({placeholders}) AND stability >= ?",
            (*candidate_ids, min_stability),
        ).fetchall()

        results = []
        for row in rows:
            intent = row_to_intent(row)
            their_specs = intent.provides + intent.requires
            for my_spec in specs:
                if any(my_spec.structurally_overlaps(ts) for ts in their_specs):
                    results.append(intent)
                    break

        return results

    def count(self) -> int:
        """Return the total number of intents in the graph."""
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM intents").fetchone()
        return row["cnt"]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self._conn.close()
