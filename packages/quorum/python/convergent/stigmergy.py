"""Stigmergy — trail markers for indirect inter-agent communication.

Agents leave markers that influence future agents, like ant pheromone
trails. Markers carry information and decay over time (evaporation),
so stale data fades naturally. Multiple agents marking the same target
reinforces the signal.

Marker types:
- file_modified: "I changed src/auth.py" (warns of potential conflicts)
- known_issue: "The login endpoint has a race condition" (knowledge sharing)
- pattern_found: "This repo uses repository pattern for DB" (style guidance)
- dependency: "Module X depends on Module Y" (sequencing hints)
- quality_signal: "Tests in test_auth.py are flaky" (reliability info)
"""

from __future__ import annotations

import logging
import math
import sqlite3
import uuid
from datetime import datetime, timezone

from convergent.protocol import StigmergyMarker

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS stigmergy_markers (
    marker_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    marker_type TEXT NOT NULL,
    target TEXT NOT NULL,
    content TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    expires_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_markers_target ON stigmergy_markers(target);
CREATE INDEX IF NOT EXISTS idx_markers_type ON stigmergy_markers(marker_type);
CREATE INDEX IF NOT EXISTS idx_markers_agent ON stigmergy_markers(agent_id);
"""


class StigmergyField:
    """Manages stigmergy trail markers with evaporation and reinforcement.

    Args:
        db_path: SQLite database path, or ":memory:" for in-memory.
        evaporation_rate: Exponential decay rate per day (higher = faster fade).
        min_strength: Markers below this threshold are removed during evaporation.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        evaporation_rate: float = 0.1,
        min_strength: float = 0.05,
    ) -> None:
        self._db_path = db_path
        self._evaporation_rate = evaporation_rate
        self._min_strength = min_strength
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def leave_marker(
        self,
        agent_id: str,
        marker_type: str,
        target: str,
        content: str,
        strength: float = 1.0,
        expires_at: str | None = None,
    ) -> StigmergyMarker:
        """Leave a trail marker for future agents.

        Args:
            agent_id: The agent leaving the marker.
            marker_type: Category (e.g. "file_modified", "known_issue").
            target: What this refers to (file path, module name, etc.).
            content: The information to convey.
            strength: Initial marker strength (default 1.0).
            expires_at: Optional explicit expiry (ISO 8601 UTC).

        Returns:
            The created StigmergyMarker.
        """
        marker_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        marker = StigmergyMarker(
            marker_id=marker_id,
            agent_id=agent_id,
            marker_type=marker_type,
            target=target,
            content=content,
            strength=strength,
            created_at=now,
            expires_at=expires_at,
        )
        self._conn.execute(
            "INSERT INTO stigmergy_markers "
            "(marker_id, agent_id, marker_type, target, content, strength, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (marker_id, agent_id, marker_type, target, content, strength, now, expires_at),
        )
        self._conn.commit()
        logger.info(
            "Marker left by %s: %s on %s (strength=%.2f)",
            agent_id,
            marker_type,
            target,
            strength,
        )
        return marker

    def get_markers(self, target: str) -> list[StigmergyMarker]:
        """Get all markers for a specific target.

        Args:
            target: The target to look up (file path, module name, etc.).

        Returns:
            List of markers for this target, with current decayed strengths.
        """
        cursor = self._conn.execute(
            "SELECT * FROM stigmergy_markers WHERE target = ? ORDER BY created_at DESC",
            (target,),
        )
        return [self._row_to_marker(row) for row in cursor]

    def get_markers_by_type(self, marker_type: str) -> list[StigmergyMarker]:
        """Get all markers of a specific type.

        Args:
            marker_type: The marker type to filter by.

        Returns:
            List of matching markers with current decayed strengths.
        """
        cursor = self._conn.execute(
            "SELECT * FROM stigmergy_markers WHERE marker_type = ? ORDER BY created_at DESC",
            (marker_type,),
        )
        return [self._row_to_marker(row) for row in cursor]

    def get_markers_by_agent(self, agent_id: str) -> list[StigmergyMarker]:
        """Get all markers left by a specific agent.

        Args:
            agent_id: The agent to look up.

        Returns:
            List of markers left by this agent.
        """
        cursor = self._conn.execute(
            "SELECT * FROM stigmergy_markers WHERE agent_id = ? ORDER BY created_at DESC",
            (agent_id,),
        )
        return [self._row_to_marker(row) for row in cursor]

    def reinforce(self, marker_id: str, amount: float = 0.5) -> float | None:
        """Increase a marker's strength (reinforcement).

        When multiple agents confirm the same observation, the marker
        gets stronger — like multiple ants reinforcing a pheromone trail.

        Args:
            marker_id: The marker to reinforce.
            amount: How much to increase strength.

        Returns:
            The new strength, or None if marker not found.
        """
        cursor = self._conn.execute(
            "SELECT strength FROM stigmergy_markers WHERE marker_id = ?",
            (marker_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        new_strength = min(row["strength"] + amount, 2.0)  # Cap at 2.0
        self._conn.execute(
            "UPDATE stigmergy_markers SET strength = ? WHERE marker_id = ?",
            (new_strength, marker_id),
        )
        self._conn.commit()
        logger.debug("Reinforced marker %s to strength %.2f", marker_id, new_strength)
        return new_strength

    def evaporate(self) -> int:
        """Decay all marker strengths and remove weak ones.

        Applies exponential decay: new_strength = strength * e^(-rate * age_days).
        Markers below min_strength are deleted.

        Returns:
            Count of markers removed.
        """
        now = datetime.now(timezone.utc)
        cursor = self._conn.execute("SELECT marker_id, strength, created_at FROM stigmergy_markers")
        rows = cursor.fetchall()

        to_delete: list[str] = []
        to_update: list[tuple[float, str]] = []

        for row in rows:
            created = datetime.fromisoformat(row["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = (now - created).total_seconds() / 86400.0
            decayed = row["strength"] * math.exp(-self._evaporation_rate * age_days)

            if decayed < self._min_strength:
                to_delete.append(row["marker_id"])
            else:
                to_update.append((decayed, row["marker_id"]))

        if to_delete:
            placeholders = ",".join("?" for _ in to_delete)
            self._conn.execute(
                f"DELETE FROM stigmergy_markers WHERE marker_id IN ({placeholders})",  # noqa: S608
                to_delete,
            )

        for strength, marker_id in to_update:
            self._conn.execute(
                "UPDATE stigmergy_markers SET strength = ? WHERE marker_id = ?",
                (strength, marker_id),
            )

        self._conn.commit()
        if to_delete:
            logger.info("Evaporation removed %d weak markers", len(to_delete))
        return len(to_delete)

    def get_context_for_agent(self, file_paths: list[str]) -> str:
        """Build a context string from markers relevant to the given files.

        This is the key integration point with Gorgon: before an agent starts
        work, call this to get "here's what previous agents learned about
        these files" and inject it into the agent prompt.

        Args:
            file_paths: Files the agent is about to work on.

        Returns:
            Human-readable context string, or empty string if no markers.
        """
        if not file_paths:
            return ""

        placeholders = ",".join("?" for _ in file_paths)
        cursor = self._conn.execute(
            f"SELECT * FROM stigmergy_markers WHERE target IN ({placeholders}) "  # noqa: S608
            "ORDER BY strength DESC, created_at DESC",
            file_paths,
        )
        rows = cursor.fetchall()
        if not rows:
            return ""

        lines = ["## Stigmergy Context (from previous agents)", ""]
        for row in rows:
            marker = self._row_to_marker(row)
            lines.append(
                f"- **[{marker.marker_type}]** `{marker.target}` "
                f"(strength={marker.strength:.2f}, by {marker.agent_id}): "
                f"{marker.content}"
            )

        return "\n".join(lines)

    def remove_marker(self, marker_id: str) -> bool:
        """Remove a specific marker.

        Args:
            marker_id: The marker to remove.

        Returns:
            True if the marker was found and removed, False otherwise.
        """
        cursor = self._conn.execute(
            "DELETE FROM stigmergy_markers WHERE marker_id = ?",
            (marker_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Get the total number of markers.

        Returns:
            Total marker count.
        """
        cursor = self._conn.execute("SELECT COUNT(*) as cnt FROM stigmergy_markers")
        return cursor.fetchone()["cnt"]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _row_to_marker(self, row: sqlite3.Row) -> StigmergyMarker:
        """Convert a database row to a StigmergyMarker."""
        return StigmergyMarker(
            marker_id=row["marker_id"],
            agent_id=row["agent_id"],
            marker_type=row["marker_type"],
            target=row["target"],
            content=row["content"],
            strength=row["strength"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )
