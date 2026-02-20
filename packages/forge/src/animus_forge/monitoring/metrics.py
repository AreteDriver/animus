"""Metrics data structures and storage for Gorgon monitoring."""

from __future__ import annotations

import sqlite3
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class StepMetrics:
    """Metrics for a single workflow step execution."""

    step_id: str
    step_type: str
    action: str
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: float = 0
    status: str = "running"  # running, success, failed, skipped
    error: str | None = None
    tokens_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def complete(self, status: str = "success", error: str | None = None):
        """Mark step as completed."""
        self.completed_at = datetime.now(UTC)
        self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000
        self.status = status
        self.error = error

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "action": self.action,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "tokens_used": self.tokens_used,
        }


@dataclass
class WorkflowMetrics:
    """Metrics for a workflow execution."""

    workflow_id: str
    execution_id: str
    workflow_name: str
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: float = 0
    status: str = "running"  # running, completed, failed
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    total_tokens: int = 0
    steps: list[StepMetrics] = field(default_factory=list)
    error: str | None = None

    def add_step(self, step: StepMetrics):
        """Add step metrics."""
        self.steps.append(step)
        self.total_steps = len(self.steps)

    def update_step(self, step_id: str, status: str, error: str | None = None):
        """Update step status."""
        for step in self.steps:
            if step.step_id == step_id:
                step.complete(status, error)
                if status == "success":
                    self.completed_steps += 1
                elif status == "failed":
                    self.failed_steps += 1
                self.total_tokens += step.tokens_used
                break

    def complete(self, status: str = "completed", error: str | None = None):
        """Mark workflow as completed."""
        self.completed_at = datetime.now(UTC)
        self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000
        self.status = status
        self.error = error

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "workflow_name": self.workflow_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "total_tokens": self.total_tokens,
            "steps": [s.to_dict() for s in self.steps],
            "error": self.error,
        }


class MetricsStore:
    """In-memory store for metrics with optional SQLite persistence."""

    _instance: MetricsStore | None = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str | None = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: str | None = None):
        if self._initialized:
            return
        self._initialized = True

        self._workflows: dict[str, WorkflowMetrics] = {}
        self._recent_executions: list[WorkflowMetrics] = []
        self._max_recent = 100
        self._db_path = db_path
        self._lock = threading.Lock()

        # Aggregate counters
        self._counters = defaultdict(int)
        self._timings = defaultdict(list)

        if db_path:
            self._init_db()

    def _init_db(self):
        """Initialize SQLite database for persistence."""
        db = Path(self._db_path)
        db.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT UNIQUE NOT NULL,
                workflow_id TEXT NOT NULL,
                workflow_name TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                duration_ms REAL,
                status TEXT,
                total_steps INTEGER,
                completed_steps INTEGER,
                failed_steps INTEGER,
                total_tokens INTEGER,
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS step_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                step_type TEXT,
                action TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                duration_ms REAL,
                status TEXT,
                tokens_used INTEGER,
                error TEXT,
                FOREIGN KEY (execution_id) REFERENCES workflow_executions(execution_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wf_exec_id ON workflow_executions(execution_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_wf_started ON workflow_executions(started_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_step_exec ON step_executions(execution_id)")
        conn.commit()
        conn.close()

    def start_workflow(self, workflow: WorkflowMetrics):
        """Record workflow start."""
        with self._lock:
            self._workflows[workflow.execution_id] = workflow
            self._counters["workflows_started"] += 1

    def complete_workflow(self, execution_id: str, status: str, error: str | None = None):
        """Record workflow completion."""
        with self._lock:
            if execution_id in self._workflows:
                wf = self._workflows[execution_id]
                wf.complete(status, error)

                # Move to recent executions
                self._recent_executions.insert(0, wf)
                if len(self._recent_executions) > self._max_recent:
                    self._recent_executions.pop()

                # Update counters
                self._counters["workflows_completed"] += 1
                if status == "failed":
                    self._counters["workflows_failed"] += 1
                self._timings["workflow_duration"].append(wf.duration_ms)

                # Persist if configured
                if self._db_path:
                    self._persist_workflow(wf)

                del self._workflows[execution_id]

    def start_step(self, execution_id: str, step: StepMetrics):
        """Record step start."""
        with self._lock:
            if execution_id in self._workflows:
                self._workflows[execution_id].add_step(step)
                self._counters["steps_started"] += 1

    def complete_step(
        self,
        execution_id: str,
        step_id: str,
        status: str,
        error: str | None = None,
        tokens: int = 0,
    ):
        """Record step completion."""
        with self._lock:
            if execution_id in self._workflows:
                wf = self._workflows[execution_id]
                for step in wf.steps:
                    if step.step_id == step_id:
                        step.tokens_used = tokens
                        break
                wf.update_step(step_id, status, error)

                self._counters["steps_completed"] += 1
                if status == "failed":
                    self._counters["steps_failed"] += 1

    def _persist_workflow(self, wf: WorkflowMetrics):
        """Persist workflow to SQLite."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            """
            INSERT INTO workflow_executions
            (execution_id, workflow_id, workflow_name, started_at, completed_at,
             duration_ms, status, total_steps, completed_steps, failed_steps,
             total_tokens, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                wf.execution_id,
                wf.workflow_id,
                wf.workflow_name,
                wf.started_at.isoformat(),
                wf.completed_at.isoformat() if wf.completed_at else None,
                wf.duration_ms,
                wf.status,
                wf.total_steps,
                wf.completed_steps,
                wf.failed_steps,
                wf.total_tokens,
                wf.error,
            ),
        )

        for step in wf.steps:
            conn.execute(
                """
                INSERT INTO step_executions
                (execution_id, step_id, step_type, action, started_at, completed_at,
                 duration_ms, status, tokens_used, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    wf.execution_id,
                    step.step_id,
                    step.step_type,
                    step.action,
                    step.started_at.isoformat(),
                    step.completed_at.isoformat() if step.completed_at else None,
                    step.duration_ms,
                    step.status,
                    step.tokens_used,
                    step.error,
                ),
            )

        conn.commit()
        conn.close()

    def get_active_workflows(self) -> list[dict]:
        """Get currently running workflows."""
        with self._lock:
            return [wf.to_dict() for wf in self._workflows.values()]

    def get_recent_executions(self, limit: int = 20) -> list[dict]:
        """Get recent completed executions."""
        with self._lock:
            return [wf.to_dict() for wf in self._recent_executions[:limit]]

    def get_summary(self) -> dict:
        """Get summary metrics."""
        with self._lock:
            avg_duration = 0
            if self._timings["workflow_duration"]:
                avg_duration = sum(self._timings["workflow_duration"]) / len(
                    self._timings["workflow_duration"]
                )

            success_rate = 0
            total = self._counters["workflows_completed"]
            if total > 0:
                success_rate = ((total - self._counters["workflows_failed"]) / total) * 100

            return {
                "active_workflows": len(self._workflows),
                "total_executions": self._counters["workflows_completed"],
                "failed_executions": self._counters["workflows_failed"],
                "success_rate": round(success_rate, 1),
                "avg_duration_ms": round(avg_duration, 2),
                "total_steps_executed": self._counters["steps_completed"],
                "total_tokens_used": sum(wf.total_tokens for wf in self._recent_executions),
            }

    def get_step_performance(self) -> dict[str, dict]:
        """Get performance metrics by step type."""
        step_stats: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "total_ms": 0, "failures": 0}
        )

        with self._lock:
            for wf in self._recent_executions:
                for step in wf.steps:
                    key = f"{step.step_type}:{step.action}"
                    step_stats[key]["count"] += 1
                    step_stats[key]["total_ms"] += step.duration_ms
                    if step.status == "failed":
                        step_stats[key]["failures"] += 1

        # Calculate averages
        result = {}
        for key, stats in step_stats.items():
            result[key] = {
                "count": stats["count"],
                "avg_ms": round(stats["total_ms"] / stats["count"], 2) if stats["count"] > 0 else 0,
                "failure_rate": round((stats["failures"] / stats["count"]) * 100, 1)
                if stats["count"] > 0
                else 0,
            }

        return result

    def get_historical_data(self, hours: int = 24) -> list[dict]:
        """Get historical execution data from database."""
        if not self._db_path:
            return self.get_recent_executions()

        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT * FROM workflow_executions
            WHERE started_at > ?
            ORDER BY started_at DESC
            """,
            (cutoff,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]
