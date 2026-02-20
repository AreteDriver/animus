"""SQLite persistence for phi-weighted agent scores, outcomes, and decisions.

Follows the same patterns as sqlite_backend.py: WAL mode,
check_same_thread=False for concurrent reads. Uses a separate
database from the intent graph to keep concerns isolated.

v0.6.0 adds decisions and vote_records tables for decision history
query API — enables post-hoc analysis of consensus outcomes.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    skill_domain TEXT NOT NULL,
    outcome TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outcomes_agent_domain
    ON outcomes(agent_id, skill_domain);

CREATE TABLE IF NOT EXISTS scores (
    agent_id TEXT NOT NULL,
    skill_domain TEXT NOT NULL,
    phi_score REAL NOT NULL,
    last_updated TEXT NOT NULL,
    PRIMARY KEY (agent_id, skill_domain)
);

CREATE TABLE IF NOT EXISTS decisions (
    request_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    question TEXT NOT NULL,
    outcome TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    decision_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_task ON decisions(task_id);
CREATE INDEX IF NOT EXISTS idx_decisions_outcome ON decisions(outcome);

CREATE TABLE IF NOT EXISTS vote_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    choice TEXT NOT NULL,
    confidence REAL NOT NULL,
    weighted_score REAL NOT NULL,
    reasoning TEXT NOT NULL,
    voted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_votes_request ON vote_records(request_id);
CREATE INDEX IF NOT EXISTS idx_votes_agent ON vote_records(agent_id);
"""


class ScoreStore:
    """SQLite persistence layer for agent outcomes, phi scores, and decisions.

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

    # --- Outcomes ---

    def record_outcome(
        self,
        agent_id: str,
        skill_domain: str,
        outcome: str,
        timestamp: str | None = None,
    ) -> None:
        """Record a task outcome for an agent in a skill domain.

        Args:
            agent_id: The agent whose outcome is being recorded.
            skill_domain: The skill domain (e.g. "code_review", "testing").
            outcome: The outcome ("approved", "rejected", "failed").
            timestamp: ISO 8601 timestamp. Defaults to now (UTC).
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO outcomes (agent_id, skill_domain, outcome, timestamp) VALUES (?, ?, ?, ?)",
            (agent_id, skill_domain, outcome, timestamp),
        )
        self._conn.commit()

    def get_outcomes(
        self,
        agent_id: str,
        skill_domain: str,
    ) -> list[tuple[str, str]]:
        """Get all outcomes for an agent in a skill domain.

        Returns:
            List of (outcome, timestamp) tuples, ordered oldest first.
        """
        cursor = self._conn.execute(
            "SELECT outcome, timestamp FROM outcomes "
            "WHERE agent_id = ? AND skill_domain = ? "
            "ORDER BY timestamp ASC",
            (agent_id, skill_domain),
        )
        return [(row["outcome"], row["timestamp"]) for row in cursor]

    def get_all_domains(self, agent_id: str) -> list[str]:
        """Get all skill domains that have outcomes for an agent.

        Returns:
            List of distinct skill domain names.
        """
        cursor = self._conn.execute(
            "SELECT DISTINCT skill_domain FROM outcomes WHERE agent_id = ?",
            (agent_id,),
        )
        return [row["skill_domain"] for row in cursor]

    # --- Scores ---

    def save_score(
        self,
        agent_id: str,
        skill_domain: str,
        phi_score: float,
    ) -> None:
        """Save or update a computed phi score.

        Args:
            agent_id: The agent.
            skill_domain: The skill domain.
            phi_score: The computed phi score.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO scores "
            "(agent_id, skill_domain, phi_score, last_updated) "
            "VALUES (?, ?, ?, ?)",
            (agent_id, skill_domain, phi_score, now),
        )
        self._conn.commit()

    def get_score(self, agent_id: str, skill_domain: str) -> float | None:
        """Get the stored phi score for an agent in a skill domain.

        Returns:
            The phi score, or None if no score exists.
        """
        cursor = self._conn.execute(
            "SELECT phi_score FROM scores WHERE agent_id = ? AND skill_domain = ?",
            (agent_id, skill_domain),
        )
        row = cursor.fetchone()
        return row["phi_score"] if row else None

    def get_all_scores(self, agent_id: str) -> dict[str, float]:
        """Get all stored phi scores for an agent across all domains.

        Returns:
            Dict mapping skill_domain to phi_score.
        """
        cursor = self._conn.execute(
            "SELECT skill_domain, phi_score FROM scores WHERE agent_id = ?",
            (agent_id,),
        )
        return {row["skill_domain"]: row["phi_score"] for row in cursor}

    # --- Decision History ---

    def record_decision(self, decision: object) -> None:
        """Persist a Decision and its votes.

        Accepts a ``Decision`` dataclass from ``convergent.protocol``.
        Stores the full decision as JSON plus individual vote records.

        Args:
            decision: A ``convergent.protocol.Decision`` instance.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO decisions "
            "(request_id, task_id, question, outcome, decided_at, decision_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                decision.request.request_id,  # type: ignore[attr-defined]
                decision.request.task_id,  # type: ignore[attr-defined]
                decision.request.question,  # type: ignore[attr-defined]
                decision.outcome.value,  # type: ignore[attr-defined]
                decision.decided_at,  # type: ignore[attr-defined]
                decision.to_json(),  # type: ignore[attr-defined]
            ),
        )
        for vote in decision.votes:  # type: ignore[attr-defined]
            self._conn.execute(
                "INSERT INTO vote_records "
                "(request_id, agent_id, choice, confidence, "
                "weighted_score, reasoning, voted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    decision.request.request_id,  # type: ignore[attr-defined]
                    vote.agent.agent_id,
                    vote.choice.value,
                    vote.confidence,
                    vote.weighted_score,
                    vote.reasoning,
                    now,
                ),
            )
        self._conn.commit()

    def get_decision_history(
        self,
        task_id: str | None = None,
        outcome: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query decision history with optional filters.

        Args:
            task_id: Filter by task. None for all.
            outcome: Filter by outcome string (e.g. "approved"). None for all.
            since: ISO 8601 timestamp cutoff. None for all.
            limit: Maximum results (default 100).

        Returns:
            List of dicts with request_id, task_id, question, outcome,
            decided_at keys.
        """
        clauses: list[str] = []
        params: list[str | int] = []

        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if outcome is not None:
            clauses.append("outcome = ?")
            params.append(outcome)
        if since is not None:
            clauses.append("decided_at > ?")
            params.append(since)

        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)

        params.append(limit)
        cursor = self._conn.execute(
            "SELECT request_id, task_id, question, outcome, decided_at "  # noqa: S608
            f"FROM decisions {where} "
            "ORDER BY decided_at DESC LIMIT ?",
            params,
        )
        return [
            {
                "request_id": row["request_id"],
                "task_id": row["task_id"],
                "question": row["question"],
                "outcome": row["outcome"],
                "decided_at": row["decided_at"],
            }
            for row in cursor
        ]

    def get_decision_json(self, request_id: str) -> str | None:
        """Get the full decision JSON for a request.

        Args:
            request_id: The consensus request ID.

        Returns:
            JSON string of the full Decision, or None if not found.
        """
        cursor = self._conn.execute(
            "SELECT decision_json FROM decisions WHERE request_id = ?",
            (request_id,),
        )
        row = cursor.fetchone()
        return row["decision_json"] if row else None

    def get_vote_records(
        self,
        agent_id: str | None = None,
        request_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query individual vote records.

        Args:
            agent_id: Filter by voting agent. None for all.
            request_id: Filter by request. None for all.
            limit: Maximum results.

        Returns:
            List of vote record dicts.
        """
        clauses: list[str] = []
        params: list[str | int] = []

        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if request_id is not None:
            clauses.append("request_id = ?")
            params.append(request_id)

        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)

        params.append(limit)
        cursor = self._conn.execute(
            "SELECT request_id, agent_id, choice, confidence, "  # noqa: S608
            f"weighted_score, reasoning, voted_at FROM vote_records {where} "
            "ORDER BY voted_at DESC LIMIT ?",
            params,
        )
        return [
            {
                "request_id": row["request_id"],
                "agent_id": row["agent_id"],
                "choice": row["choice"],
                "confidence": row["confidence"],
                "weighted_score": row["weighted_score"],
                "reasoning": row["reasoning"],
                "voted_at": row["voted_at"],
            }
            for row in cursor
        ]

    def get_agent_vote_stats(self, agent_id: str) -> dict:
        """Get voting statistics for an agent.

        Args:
            agent_id: The agent to look up.

        Returns:
            Dict with total, approve_count, reject_count, abstain_count,
            escalate_count, avg_confidence.
        """
        cursor = self._conn.execute(
            "SELECT choice, COUNT(*) as cnt, AVG(confidence) as avg_conf "
            "FROM vote_records WHERE agent_id = ? GROUP BY choice",
            (agent_id,),
        )
        stats: dict = {
            "total": 0,
            "approve_count": 0,
            "reject_count": 0,
            "abstain_count": 0,
            "escalate_count": 0,
            "avg_confidence": 0.0,
        }
        total_conf = 0.0
        for row in cursor:
            choice = row["choice"]
            count = row["cnt"]
            stats["total"] += count
            stats[f"{choice}_count"] = count
            total_conf += row["avg_conf"] * count

        if stats["total"] > 0:
            stats["avg_confidence"] = total_conf / stats["total"]

        return stats

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
