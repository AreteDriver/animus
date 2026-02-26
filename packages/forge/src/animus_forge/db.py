"""Task history store for analytics and phi-weighted scoring.

Provides denormalized task records, agent performance scores, and
daily budget rollups via the shared DatabaseBackend.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from animus_forge.state.backends import DatabaseBackend

logger = logging.getLogger(__name__)

_task_store: TaskStore | None = None


class TaskStore:
    """Analytics store for completed/failed tasks.

    Follows the PersistentBudgetManager pattern — takes a shared
    DatabaseBackend and provides domain-specific CRUD + aggregation.
    """

    def __init__(self, backend: DatabaseBackend):
        self.backend = backend

    # =========================================================================
    # Record
    # =========================================================================

    def record_task(
        self,
        job_id: str,
        workflow_id: str,
        status: str,
        *,
        agent_role: str | None = None,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> int:
        """Record a completed/failed task and update aggregates.

        Returns the new task_history row id.
        """
        now = datetime.now()
        created = created_at or now
        completed = completed_at or now
        meta_json = json.dumps(metadata) if metadata else None

        with self.backend.transaction():
            cursor = self.backend.execute(
                """
                INSERT INTO task_history
                    (job_id, workflow_id, status, agent_role, model,
                     input_tokens, output_tokens, total_tokens,
                     cost_usd, duration_ms, error, metadata,
                     created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    workflow_id,
                    status,
                    agent_role,
                    model,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cost_usd,
                    duration_ms,
                    error,
                    meta_json,
                    created.isoformat(),
                    completed.isoformat(),
                ),
            )
            task_id = cursor.lastrowid

            self._update_agent_score(
                agent_role or "unknown",
                status=status,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                duration_ms=duration_ms,
            )

            self._log_budget(
                date=completed.strftime("%Y-%m-%d"),
                agent_role=agent_role or "unknown",
                total_tokens=total_tokens,
                cost_usd=cost_usd,
            )

        return task_id

    # =========================================================================
    # Query
    # =========================================================================

    def query_tasks(
        self,
        *,
        status: str | None = None,
        workflow_id: str | None = None,
        agent_role: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """Query task history with optional filters."""
        query = """
            SELECT id, job_id, workflow_id, status, agent_role, model,
                   input_tokens, output_tokens, total_tokens,
                   cost_usd, duration_ms, error, metadata,
                   created_at, completed_at
            FROM task_history
            WHERE 1=1
        """
        params: list[Any] = []

        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if workflow_id is not None:
            query += " AND workflow_id = ?"
            params.append(workflow_id)
        if agent_role is not None:
            query += " AND agent_role = ?"
            params.append(agent_role)

        query += " ORDER BY completed_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.backend.fetchall(query, tuple(params))
        return [self._parse_task_row(row) for row in rows]

    def get_task(self, task_id: int) -> dict | None:
        """Get a single task by id."""
        row = self.backend.fetchone(
            """
            SELECT id, job_id, workflow_id, status, agent_role, model,
                   input_tokens, output_tokens, total_tokens,
                   cost_usd, duration_ms, error, metadata,
                   created_at, completed_at
            FROM task_history
            WHERE id = ?
            """,
            (task_id,),
        )
        return self._parse_task_row(row) if row else None

    def get_agent_stats(self, agent_role: str | None = None) -> list[dict]:
        """Get agent performance stats.

        If agent_role is specified, returns a single-item list for that agent.
        Otherwise returns all agents sorted by total_tasks descending.
        """
        if agent_role is not None:
            row = self.backend.fetchone(
                "SELECT * FROM agent_scores WHERE agent_role = ?",
                (agent_role,),
            )
            return [dict(row)] if row else []

        rows = self.backend.fetchall("SELECT * FROM agent_scores ORDER BY total_tasks DESC")
        return [dict(row) for row in rows]

    def get_daily_budget(self, days: int = 7, agent_role: str | None = None) -> list[dict]:
        """Get daily budget rollups for the last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        query = "SELECT * FROM budget_log WHERE date >= ?"
        params: list[Any] = [cutoff]

        if agent_role is not None:
            query += " AND agent_role = ?"
            params.append(agent_role)

        query += " ORDER BY date DESC"

        rows = self.backend.fetchall(query, tuple(params))
        return [dict(row) for row in rows]

    def get_summary(self) -> dict:
        """Get high-level summary: total tasks, success rate, cost, top agents."""
        row = self.backend.fetchone(
            """
            SELECT
                COUNT(*) as total_tasks,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(cost_usd), 0.0) as total_cost_usd
            FROM task_history
            """
        )

        total = row["total_tasks"] if row else 0
        successful = row["successful"] if row else 0
        total_tokens = row["total_tokens"] if row else 0
        total_cost = row["total_cost_usd"] if row else 0.0

        top_agents = self.backend.fetchall(
            """
            SELECT agent_role, total_tasks, success_rate
            FROM agent_scores
            ORDER BY total_tasks DESC
            LIMIT 5
            """
        )

        return {
            "total_tasks": total,
            "successful": successful,
            "failed": row["failed"] if row else 0,
            "success_rate": round(successful / total * 100, 1) if total > 0 else 0.0,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "top_agents": [dict(a) for a in top_agents],
        }

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _update_agent_score(
        self,
        agent_role: str,
        *,
        status: str,
        total_tokens: int,
        cost_usd: float,
        duration_ms: int,
    ) -> None:
        """Insert or update running agent score aggregates using UPSERT."""
        is_success = 1 if status == "completed" else 0
        is_fail = 1 if status != "completed" else 0
        success_rate = 100.0 if status == "completed" else 0.0
        now = datetime.now().isoformat()

        self.backend.execute(
            """
            INSERT INTO agent_scores
                (agent_role, total_tasks, successful_tasks, failed_tasks,
                 total_tokens, total_cost_usd, avg_duration_ms,
                 success_rate, updated_at)
            VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_role) DO UPDATE SET
                total_tasks = total_tasks + 1,
                successful_tasks = successful_tasks + ?,
                failed_tasks = failed_tasks + ?,
                total_tokens = total_tokens + ?,
                total_cost_usd = total_cost_usd + ?,
                avg_duration_ms = (avg_duration_ms * total_tasks + ?) / (total_tasks + 1),
                success_rate = ROUND(
                    (successful_tasks + ?) * 100.0 / (total_tasks + 1), 1
                ),
                updated_at = ?
            """,
            (
                agent_role,
                is_success,
                is_fail,
                total_tokens,
                cost_usd,
                float(duration_ms),
                success_rate,
                now,
                is_success,
                is_fail,
                total_tokens,
                cost_usd,
                float(duration_ms),
                is_success,
                now,
            ),
        )

    def _log_budget(
        self,
        date: str,
        agent_role: str,
        total_tokens: int,
        cost_usd: float,
    ) -> None:
        """UPSERT daily budget rollup."""
        existing = self.backend.fetchone(
            "SELECT * FROM budget_log WHERE date = ? AND agent_role = ?",
            (date, agent_role),
        )

        if existing is None:
            self.backend.execute(
                """
                INSERT INTO budget_log
                    (date, agent_role, total_tokens, total_cost_usd, task_count)
                VALUES (?, ?, ?, ?, 1)
                """,
                (date, agent_role, total_tokens, cost_usd),
            )
        else:
            self.backend.execute(
                """
                UPDATE budget_log
                SET total_tokens = total_tokens + ?,
                    total_cost_usd = total_cost_usd + ?,
                    task_count = task_count + 1
                WHERE date = ? AND agent_role = ?
                """,
                (total_tokens, cost_usd, date, agent_role),
            )

    def _parse_task_row(self, row: dict) -> dict:
        """Convert a task_history row dict, parsing JSON metadata."""
        result = dict(row)
        if result.get("metadata"):
            try:
                result["metadata"] = json.loads(result["metadata"])
            except (json.JSONDecodeError, TypeError):
                logger.debug("Failed to parse task metadata as JSON, keeping raw value")
        return result


# =============================================================================
# Global access
# =============================================================================


def get_task_store() -> TaskStore:
    """Get or create the global TaskStore singleton."""
    global _task_store
    if _task_store is None:
        from animus_forge.state.database import get_database

        _task_store = TaskStore(get_database())
    return _task_store


def reset_task_store() -> None:
    """Reset the global TaskStore singleton (for testing)."""
    global _task_store
    _task_store = None
