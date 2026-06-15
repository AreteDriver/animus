"""Execution Tracker for Gorgon Workflows.

Provides context manager and decorators for automatic metrics collection.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from typing import Any

from .metrics import MetricsStore, StepMetrics, WorkflowMetrics

# Global tracker instance
_tracker: ExecutionTracker | None = None


def get_tracker(db_path: str | None = None) -> ExecutionTracker:
    """Get or create global execution tracker."""
    global _tracker
    if _tracker is None:
        _tracker = ExecutionTracker(db_path)
    return _tracker


class ExecutionTracker:
    """Tracks workflow and step executions with metrics collection."""

    def __init__(self, db_path: str | None = None):
        self.store = MetricsStore(db_path)
        self._current_execution: str | None = None
        self._current_workflow: WorkflowMetrics | None = None

    @contextmanager
    def track_workflow(
        self,
        workflow_id: str,
        workflow_name: str = "",
    ) -> Generator[str, None, None]:
        """Context manager to track a workflow execution.

        Usage:
            with tracker.track_workflow("my_workflow", "My Workflow") as exec_id:
                # Execute workflow steps
                pass
        """
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        workflow = WorkflowMetrics(
            workflow_id=workflow_id,
            execution_id=execution_id,
            workflow_name=workflow_name or workflow_id,
            started_at=datetime.now(UTC),
        )

        self._current_execution = execution_id
        self._current_workflow = workflow
        self.store.start_workflow(workflow)

        try:
            yield execution_id
            self.store.complete_workflow(execution_id, "completed")
        except Exception as e:
            self.store.complete_workflow(execution_id, "failed", str(e))
            raise
        finally:
            self._current_execution = None
            self._current_workflow = None

    @contextmanager
    def track_step(
        self,
        step_id: str,
        step_type: str,
        action: str,
        execution_id: str | None = None,
    ) -> Generator[StepMetrics, None, None]:
        """Context manager to track a step execution.

        Usage:
            with tracker.track_step("step_1", "claude_code", "execute_agent") as step:
                # Execute step
                step.tokens_used = 150
        """
        exec_id = execution_id or self._current_execution
        if not exec_id:
            raise ValueError("No active workflow execution")

        step = StepMetrics(
            step_id=step_id,
            step_type=step_type,
            action=action,
            started_at=datetime.now(UTC),
        )

        self.store.start_step(exec_id, step)

        try:
            yield step
            self.store.complete_step(exec_id, step_id, "success", tokens=step.tokens_used)
        except Exception as e:
            self.store.complete_step(exec_id, step_id, "failed", str(e))
            raise

    def track_step_decorator(
        self,
        step_type: str,
        action: str,
    ) -> Callable:
        """Decorator to track function execution as a step.

        Usage:
            @tracker.track_step_decorator("transform", "format")
            def format_data(data):
                return formatted_data
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                step_id = f"{func.__name__}_{uuid.uuid4().hex[:8]}"
                with self.track_step(step_id, step_type, action):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def record_tokens(self, execution_id: str, step_id: str, tokens: int):
        """Record token usage for a step."""
        self.store.complete_step(execution_id, step_id, "success", tokens=tokens)

    def get_status(self) -> dict:
        """Get current tracker status."""
        summary = self.store.get_summary()
        active = self.store.get_active_workflows()

        return {
            "summary": summary,
            "active_workflows": active,
            "current_execution": self._current_execution,
        }

    def get_dashboard_data(self) -> dict:
        """Get all data needed for dashboard display."""
        return {
            "summary": self.store.get_summary(),
            "active_workflows": self.store.get_active_workflows(),
            "recent_executions": self.store.get_recent_executions(20),
            "step_performance": self.store.get_step_performance(),
        }


class AgentTracker:
    """Tracks AI agent activity across workflows."""

    def __init__(self):
        self._active_agents: dict[str, dict] = {}
        self._agent_history: list[dict] = []
        self._max_history = 50

    def register_agent(
        self,
        agent_id: str,
        role: str,
        workflow_id: str | None = None,
    ):
        """Register an active agent."""
        self._active_agents[agent_id] = {
            "agent_id": agent_id,
            "role": role,
            "workflow_id": workflow_id,
            "started_at": datetime.now(UTC).isoformat(),
            "status": "active",
            "tasks_completed": 0,
        }

    def update_agent(self, agent_id: str, **updates):
        """Update agent status."""
        if agent_id in self._active_agents:
            self._active_agents[agent_id].update(updates)

    def complete_agent(self, agent_id: str, status: str = "completed"):
        """Mark agent as completed."""
        if agent_id in self._active_agents:
            agent = self._active_agents.pop(agent_id)
            agent["status"] = status
            agent["completed_at"] = datetime.now(UTC).isoformat()
            self._agent_history.insert(0, agent)
            if len(self._agent_history) > self._max_history:
                self._agent_history.pop()

    def get_active_agents(self) -> list[dict]:
        """Get currently active agents."""
        return list(self._active_agents.values())

    def get_agent_history(self, limit: int = 20) -> list[dict]:
        """Get recent agent activity."""
        return self._agent_history[:limit]

    def get_agent_summary(self) -> dict:
        """Get agent activity summary."""
        roles = {}
        for agent in self._active_agents.values():
            role = agent["role"]
            roles[role] = roles.get(role, 0) + 1

        return {
            "active_count": len(self._active_agents),
            "by_role": roles,
            "recent_count": len(self._agent_history),
        }
