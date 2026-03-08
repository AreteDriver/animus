"""Persistent storage for agent runs.

Provides SQLite-backed persistence so SubAgentManager runs
survive process restarts and can be queried for analytics.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class AgentRunStore:
    """SQLite persistence layer for agent runs.

    Works with any DatabaseBackend (SQLite or PostgreSQL).
    Stores completed runs for history/analytics while
    SubAgentManager keeps active runs in memory.

    Args:
        backend: Database backend instance.
    """

    def __init__(self, backend: Any):
        self._backend = backend

    def save_run(self, run: Any) -> None:
        """Persist an agent run to the database.

        Args:
            run: AgentRun dataclass instance.
        """
        children_json = json.dumps(run.children) if run.children else "[]"
        config_json = "{}"
        if hasattr(run, "config") and run.config is not None:
            try:
                config_json = json.dumps(
                    {
                        "timeout_seconds": getattr(run.config, "timeout_seconds", 300),
                        "max_output_chars": getattr(run.config, "max_output_chars", 50000),
                        "model": getattr(run.config, "model", None),
                    }
                )
            except (TypeError, ValueError):
                config_json = "{}"

        status = run.status.value if hasattr(run.status, "value") else str(run.status)

        query = self._backend.adapt_query(
            "INSERT OR REPLACE INTO agent_runs "
            "(run_id, agent, task, status, result, error, "
            "started_at, completed_at, parent_id, children, config_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        self._backend.execute(
            query,
            (
                run.run_id,
                run.agent,
                run.task[:2000],
                status,
                run.result[:5000] if run.result else None,
                run.error,
                run.started_at,
                run.completed_at,
                run.parent_id,
                children_json,
                config_json,
            ),
        )

    def get_run(self, run_id: str) -> dict | None:
        """Load a run by ID.

        Args:
            run_id: Run identifier.

        Returns:
            Dict of run data or None.
        """
        query = self._backend.adapt_query("SELECT * FROM agent_runs WHERE run_id = ?")
        row = self._backend.fetchone(query, (run_id,))
        if row:
            row["children"] = json.loads(row.get("children", "[]"))
        return row

    def list_runs(
        self,
        status: str | None = None,
        agent: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List runs with optional filters.

        Args:
            status: Filter by status string.
            agent: Filter by agent role.
            limit: Max results.

        Returns:
            List of run dicts, newest first.
        """
        conditions = []
        params: list[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if agent:
            conditions.append("agent = ?")
            params.append(agent)

        where = ""
        if conditions:
            where = " WHERE " + " AND ".join(conditions)

        query = self._backend.adapt_query(
            f"SELECT * FROM agent_runs{where} ORDER BY started_at DESC LIMIT ?"
        )
        params.append(limit)
        rows = self._backend.fetchall(query, tuple(params))
        for row in rows:
            row["children"] = json.loads(row.get("children", "[]"))
        return rows

    def delete_older_than(self, cutoff_epoch: float) -> int:
        """Delete runs older than cutoff.

        Args:
            cutoff_epoch: Epoch time threshold.

        Returns:
            Number of rows deleted.
        """
        query = self._backend.adapt_query(
            "DELETE FROM agent_runs WHERE completed_at > 0 AND completed_at < ?"
        )
        cursor = self._backend.execute(query, (cutoff_epoch,))
        return getattr(cursor, "rowcount", 0)

    def count(self, status: str | None = None) -> int:
        """Count runs, optionally filtered by status.

        Args:
            status: Optional status filter.

        Returns:
            Count of matching runs.
        """
        if status:
            query = self._backend.adapt_query(
                "SELECT COUNT(*) as cnt FROM agent_runs WHERE status = ?"
            )
            row = self._backend.fetchone(query, (status,))
        else:
            query = "SELECT COUNT(*) as cnt FROM agent_runs"
            row = self._backend.fetchone(query)
        return row["cnt"] if row else 0
