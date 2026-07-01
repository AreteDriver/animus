"""Database-backed workflow state persistence."""

from __future__ import annotations

import json
from enum import Enum

from .backends import DatabaseBackend


class WorkflowStatus(Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"


class StatePersistence:
    """Database-backed workflow state persistence.

    Provides checkpoint/resume capability for workflow executions.
    Thread-safe for concurrent access. Supports SQLite and PostgreSQL.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            current_stage TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            config TEXT,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            input_data TEXT,
            output_data TEXT,
            tokens_used INTEGER DEFAULT 0,
            duration_ms INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (workflow_id) REFERENCES workflows(id)
        );

        CREATE INDEX IF NOT EXISTS idx_checkpoints_workflow
        ON checkpoints(workflow_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_workflows_status
        ON workflows(status);
    """

    def __init__(
        self,
        db_path: str = "gorgon-state.db",
        *,
        backend: DatabaseBackend | None = None,
    ):
        """Initialize state persistence.

        Args:
            db_path: Path to SQLite database (if no backend provided)
            backend: Database backend to use (keyword-only, overrides db_path)
        """
        if backend:
            self.backend = backend
        else:
            from .backends import SQLiteBackend

            self.backend = SQLiteBackend(db_path=db_path)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self.backend.executescript(self.SCHEMA)

    def create_workflow(
        self,
        workflow_id: str,
        name: str,
        config: dict = None,
    ) -> None:
        """Create a new workflow record.

        Args:
            workflow_id: Unique workflow identifier
            name: Human-readable workflow name
            config: Optional workflow configuration
        """
        with self.backend.transaction():
            self.backend.execute(
                """
                INSERT INTO workflows (id, name, status, config)
                VALUES (?, ?, ?, ?)
                """,
                (
                    workflow_id,
                    name,
                    WorkflowStatus.PENDING.value,
                    json.dumps(config) if config else None,
                ),
            )

    def update_status(
        self,
        workflow_id: str,
        status: WorkflowStatus,
        error: str = None,
    ) -> None:
        """Update workflow status.

        Args:
            workflow_id: Workflow identifier
            status: New status
            error: Optional error message (for failed status)
        """
        with self.backend.transaction():
            self.backend.execute(
                """
                UPDATE workflows
                SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status.value, error, workflow_id),
            )

    def checkpoint(
        self,
        workflow_id: str,
        stage: str,
        status: str,
        input_data: dict = None,
        output_data: dict = None,
        tokens_used: int = 0,
        duration_ms: int = 0,
    ) -> int:
        """Save a checkpoint for a workflow stage.

        Args:
            workflow_id: Workflow identifier
            stage: Stage name
            status: Stage status (success, failed, etc.)
            input_data: Input data for the stage
            output_data: Output data from the stage
            tokens_used: Tokens consumed
            duration_ms: Stage duration in milliseconds

        Returns:
            Checkpoint ID
        """
        with self.backend.transaction():
            cursor = self.backend.execute(
                """
                INSERT INTO checkpoints
                (workflow_id, stage, status, input_data, output_data, tokens_used, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workflow_id,
                    stage,
                    status,
                    json.dumps(input_data) if input_data else None,
                    json.dumps(output_data) if output_data else None,
                    tokens_used,
                    duration_ms,
                ),
            )

            # Update workflow's current stage
            self.backend.execute(
                """
                UPDATE workflows
                SET current_stage = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (stage, WorkflowStatus.RUNNING.value, workflow_id),
            )

            return cursor.lastrowid

    def get_last_checkpoint(self, workflow_id: str) -> dict | None:
        """Get the most recent checkpoint for a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Checkpoint data or None if not found
        """
        row = self.backend.fetchone(
            """
            SELECT * FROM checkpoints
            WHERE workflow_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (workflow_id,),
        )

        if row:
            return {
                "id": row["id"],
                "stage": row["stage"],
                "status": row["status"],
                "input_data": json.loads(row["input_data"]) if row["input_data"] else None,
                "output_data": json.loads(row["output_data"]) if row["output_data"] else None,
                "tokens_used": row["tokens_used"],
                "duration_ms": row["duration_ms"],
                "created_at": row["created_at"],
            }
        return None

    def get_all_checkpoints(self, workflow_id: str) -> list[dict]:
        """Get all checkpoints for a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of checkpoint data
        """
        rows = self.backend.fetchall(
            """
            SELECT * FROM checkpoints
            WHERE workflow_id = ?
            ORDER BY created_at ASC
            """,
            (workflow_id,),
        )

        return [
            {
                "id": row["id"],
                "stage": row["stage"],
                "status": row["status"],
                "input_data": json.loads(row["input_data"]) if row["input_data"] else None,
                "output_data": json.loads(row["output_data"]) if row["output_data"] else None,
                "tokens_used": row["tokens_used"],
                "duration_ms": row["duration_ms"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_workflow(self, workflow_id: str) -> dict | None:
        """Get workflow by ID.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Workflow data or None if not found
        """
        row = self.backend.fetchone(
            "SELECT * FROM workflows WHERE id = ?",
            (workflow_id,),
        )

        if row:
            return {
                "id": row["id"],
                "name": row["name"],
                "status": row["status"],
                "current_stage": row["current_stage"],
                "config": json.loads(row["config"]) if row["config"] else None,
                "error": row["error"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        return None

    def get_resumable_workflows(self) -> list[dict]:
        """Get workflows that can be resumed.

        Returns:
            List of workflows with status in (running, paused, failed)
        """
        rows = self.backend.fetchall(
            """
            SELECT id, name, current_stage, status, updated_at
            FROM workflows
            WHERE status IN (?, ?, ?)
            ORDER BY updated_at DESC
            """,
            (
                WorkflowStatus.RUNNING.value,
                WorkflowStatus.PAUSED.value,
                WorkflowStatus.FAILED.value,
            ),
        )

        return list(rows)

    def resume_from_checkpoint(self, workflow_id: str) -> dict:
        """Resume a workflow from its last checkpoint.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Last checkpoint data

        Raises:
            ValueError: If no checkpoint found
        """
        checkpoint = self.get_last_checkpoint(workflow_id)
        if not checkpoint:
            raise ValueError(f"No checkpoint found for workflow {workflow_id}")

        self.update_status(workflow_id, WorkflowStatus.RUNNING)
        return checkpoint

    def mark_complete(self, workflow_id: str) -> None:
        """Mark workflow as completed.

        Args:
            workflow_id: Workflow identifier
        """
        self.update_status(workflow_id, WorkflowStatus.COMPLETED)

    def mark_failed(self, workflow_id: str, error: str) -> None:
        """Mark workflow as failed.

        Args:
            workflow_id: Workflow identifier
            error: Error message
        """
        self.update_status(workflow_id, WorkflowStatus.FAILED, error)

    def mark_paused(self, workflow_id: str) -> None:
        """Mark workflow as paused.

        Args:
            workflow_id: Workflow identifier
        """
        self.update_status(workflow_id, WorkflowStatus.PAUSED)

    def list_workflows(
        self,
        status: WorkflowStatus = None,
        limit: int = 50,
    ) -> list[dict]:
        """List workflows with optional status filter.

        Args:
            status: Optional status filter
            limit: Maximum number of results

        Returns:
            List of workflow summaries
        """
        if status:
            rows = self.backend.fetchall(
                """
                SELECT id, name, status, current_stage, created_at, updated_at
                FROM workflows
                WHERE status = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (status.value, limit),
            )
        else:
            rows = self.backend.fetchall(
                """
                SELECT id, name, status, current_stage, created_at, updated_at
                FROM workflows
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            )

        return list(rows)

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow and its checkpoints.

        Args:
            workflow_id: Workflow identifier

        Returns:
            True if deleted, False if not found
        """
        with self.backend.transaction():
            self.backend.execute(
                "DELETE FROM checkpoints WHERE workflow_id = ?",
                (workflow_id,),
            )
            cursor = self.backend.execute(
                "DELETE FROM workflows WHERE id = ?",
                (workflow_id,),
            )
            return cursor.rowcount > 0

    def get_stats(self) -> dict:
        """Get persistence statistics.

        Returns:
            Dictionary with workflow counts by status
        """
        total_row = self.backend.fetchone("SELECT COUNT(*) as count FROM workflows")
        total = total_row["count"] if total_row else 0

        by_status = {}
        for status in WorkflowStatus:
            count_row = self.backend.fetchone(
                "SELECT COUNT(*) as count FROM workflows WHERE status = ?",
                (status.value,),
            )
            by_status[status.value] = count_row["count"] if count_row else 0

        checkpoint_row = self.backend.fetchone("SELECT COUNT(*) as count FROM checkpoints")
        checkpoint_count = checkpoint_row["count"] if checkpoint_row else 0

        return {
            "total_workflows": total,
            "by_status": by_status,
            "total_checkpoints": checkpoint_count,
        }
