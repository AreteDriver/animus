"""SQLite-backed checkpoint persistence for Forge workflows."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from animus.forge.models import StepResult, WorkflowState
from animus.logging import get_logger

logger = get_logger("forge.checkpoint")

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS workflows (
    name TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    current_step INTEGER NOT NULL,
    total_tokens INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0.0,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS step_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_name TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    outputs_json TEXT NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    success INTEGER NOT NULL,
    error TEXT,
    FOREIGN KEY (workflow_name) REFERENCES workflows(name)
);
"""


class CheckpointStore:
    """SQLite-backed workflow state persistence."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        logger.debug(f"CheckpointStore opened: {db_path}")

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def save_state(self, state: WorkflowState) -> None:
        """Save or update workflow state and step results."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO workflows "
                "(name, status, current_step, total_tokens, total_cost, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    state.workflow_name,
                    state.status,
                    state.current_step,
                    state.total_tokens,
                    state.total_cost,
                    now,
                ),
            )
            # Replace step results — delete existing and re-insert
            self._conn.execute(
                "DELETE FROM step_results WHERE workflow_name = ?",
                (state.workflow_name,),
            )
            for i, result in enumerate(state.results):
                self._conn.execute(
                    "INSERT INTO step_results "
                    "(workflow_name, agent_name, step_index, outputs_json, "
                    "tokens_used, cost_usd, success, error) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        state.workflow_name,
                        result.agent_name,
                        i,
                        json.dumps(result.outputs),
                        result.tokens_used,
                        result.cost_usd,
                        1 if result.success else 0,
                        result.error,
                    ),
                )
        logger.debug(
            f"Saved checkpoint: {state.workflow_name} "
            f"step={state.current_step} status={state.status}"
        )

    def load_state(self, workflow_name: str) -> WorkflowState | None:
        """Load workflow state from checkpoint.

        Returns None if no checkpoint exists for this workflow.
        """
        row = self._conn.execute(
            "SELECT name, status, current_step, total_tokens, total_cost "
            "FROM workflows WHERE name = ?",
            (workflow_name,),
        ).fetchone()

        if row is None:
            return None

        state = WorkflowState(
            workflow_name=row[0],
            status=row[1],
            current_step=row[2],
            total_tokens=row[3],
            total_cost=row[4],
        )

        step_rows = self._conn.execute(
            "SELECT agent_name, outputs_json, tokens_used, cost_usd, success, error "
            "FROM step_results WHERE workflow_name = ? ORDER BY step_index",
            (workflow_name,),
        ).fetchall()

        for sr in step_rows:
            state.results.append(
                StepResult(
                    agent_name=sr[0],
                    outputs=json.loads(sr[1]),
                    tokens_used=sr[2],
                    cost_usd=sr[3],
                    success=bool(sr[4]),
                    error=sr[5],
                )
            )

        logger.debug(
            f"Loaded checkpoint: {workflow_name} "
            f"step={state.current_step} results={len(state.results)}"
        )
        return state

    def delete_state(self, workflow_name: str) -> None:
        """Delete all checkpoint data for a workflow."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM step_results WHERE workflow_name = ?",
                (workflow_name,),
            )
            self._conn.execute(
                "DELETE FROM workflows WHERE name = ?",
                (workflow_name,),
            )
        logger.debug(f"Deleted checkpoint: {workflow_name}")

    def list_workflows(self) -> list[tuple[str, str, int]]:
        """List all checkpointed workflows.

        Returns:
            List of (name, status, current_step) tuples.
        """
        rows = self._conn.execute(
            "SELECT name, status, current_step FROM workflows ORDER BY name"
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]
