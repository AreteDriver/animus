"""Metrics collection for workflow execution."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class StepMetrics:
    """Metrics for a single workflow step."""

    step_id: str
    step_type: str
    status: str = "pending"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int = 0
    tokens_used: int = 0
    retries: int = 0
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def start(self) -> None:
        """Mark step as started."""
        self.started_at = datetime.now(UTC)
        self.status = "running"

    def complete(self, tokens: int = 0) -> None:
        """Mark step as completed."""
        self.completed_at = datetime.now(UTC)
        self.status = "success"
        self.tokens_used = tokens
        if self.started_at:
            self.duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)

    def fail(self, error: str) -> None:
        """Mark step as failed."""
        self.completed_at = datetime.now(UTC)
        self.status = "failed"
        self.error = error
        if self.started_at:
            self.duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "retries": self.retries,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowMetrics:
    """Metrics for a workflow execution."""

    workflow_id: str
    workflow_name: str
    execution_id: str
    status: str = "pending"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int = 0
    total_tokens: int = 0
    steps: dict[str, StepMetrics] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def start(self) -> None:
        """Mark workflow as started."""
        self.started_at = datetime.now(UTC)
        self.status = "running"

    def complete(self) -> None:
        """Mark workflow as completed."""
        self.completed_at = datetime.now(UTC)
        self.status = "success"
        self._finalize()

    def fail(self, error: str | None = None) -> None:
        """Mark workflow as failed."""
        self.completed_at = datetime.now(UTC)
        self.status = "failed"
        if error:
            self.metadata["error"] = error
        self._finalize()

    def _finalize(self) -> None:
        """Calculate final metrics."""
        if self.started_at and self.completed_at:
            self.duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
        self.total_tokens = sum(s.tokens_used for s in self.steps.values())

    def add_step(self, step_id: str, step_type: str) -> StepMetrics:
        """Add a step to track."""
        step = StepMetrics(step_id=step_id, step_type=step_type)
        self.steps[step_id] = step
        return step

    def get_step(self, step_id: str) -> StepMetrics | None:
        """Get step metrics."""
        return self.steps.get(step_id)

    @property
    def success_rate(self) -> float:
        """Calculate step success rate."""
        if not self.steps:
            return 0.0
        successful = sum(1 for s in self.steps.values() if s.status == "success")
        return successful / len(self.steps)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "execution_id": self.execution_id,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "total_tokens": self.total_tokens,
            "success_rate": self.success_rate,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "metadata": self.metadata,
        }


class MetricsCollector:
    """Collects and aggregates workflow metrics.

    Thread-safe metrics collection with support for
    real-time callbacks and historical data.
    """

    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector.

        Args:
            max_history: Maximum workflow executions to keep in history
        """
        self._lock = threading.Lock()
        self._active: dict[str, WorkflowMetrics] = {}
        self._history: list[WorkflowMetrics] = []
        self._max_history = max_history
        self._callbacks: list[Callable[[str, WorkflowMetrics], None]] = []

        # Aggregated metrics
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)

    def register_callback(self, callback: Callable[[str, WorkflowMetrics], None]) -> None:
        """Register a callback for metric events.

        Args:
            callback: Function called with (event_type, metrics)
        """
        self._callbacks.append(callback)

    def _notify(self, event_type: str, metrics: WorkflowMetrics) -> None:
        """Notify all callbacks."""
        for callback in self._callbacks:
            try:
                callback(event_type, metrics)
            except Exception:
                pass  # Best-effort cleanup: callback failure must not break metrics collection

    def start_workflow(
        self,
        workflow_id: str,
        workflow_name: str,
        execution_id: str,
        metadata: dict | None = None,
    ) -> WorkflowMetrics:
        """Start tracking a workflow execution.

        Args:
            workflow_id: Workflow identifier
            workflow_name: Human-readable name
            execution_id: Unique execution identifier
            metadata: Optional metadata

        Returns:
            WorkflowMetrics instance
        """
        with self._lock:
            metrics = WorkflowMetrics(
                workflow_id=workflow_id,
                workflow_name=workflow_name,
                execution_id=execution_id,
                metadata=metadata or {},
            )
            metrics.start()
            self._active[execution_id] = metrics

            self._counters["workflows_started"] += 1
            self._gauges["active_workflows"] = len(self._active)

        self._notify("workflow_started", metrics)
        return metrics

    def start_step(
        self,
        execution_id: str,
        step_id: str,
        step_type: str,
    ) -> StepMetrics | None:
        """Start tracking a workflow step.

        Args:
            execution_id: Workflow execution ID
            step_id: Step identifier
            step_type: Step type (e.g., 'claude_code', 'shell')

        Returns:
            StepMetrics instance or None if execution not found
        """
        with self._lock:
            workflow = self._active.get(execution_id)
            if not workflow:
                return None

            step = workflow.add_step(step_id, step_type)
            step.start()

            self._counters["steps_started"] += 1
            self._counters[f"steps_started_{step_type}"] += 1

        return step

    def complete_step(
        self,
        execution_id: str,
        step_id: str,
        tokens: int = 0,
        metadata: dict | None = None,
    ) -> None:
        """Mark a step as completed.

        Args:
            execution_id: Workflow execution ID
            step_id: Step identifier
            tokens: Tokens used by this step
            metadata: Optional metadata
        """
        with self._lock:
            workflow = self._active.get(execution_id)
            if not workflow:
                return

            step = workflow.get_step(step_id)
            if not step:
                return

            step.complete(tokens)
            if metadata:
                step.metadata.update(metadata)

            self._counters["steps_completed"] += 1
            self._histograms["step_duration_ms"].append(step.duration_ms)
            self._histograms["step_tokens"].append(tokens)

    def fail_step(
        self,
        execution_id: str,
        step_id: str,
        error: str,
    ) -> None:
        """Mark a step as failed.

        Args:
            execution_id: Workflow execution ID
            step_id: Step identifier
            error: Error message
        """
        with self._lock:
            workflow = self._active.get(execution_id)
            if not workflow:
                return

            step = workflow.get_step(step_id)
            if not step:
                return

            step.fail(error)

            self._counters["steps_failed"] += 1

    def complete_workflow(
        self,
        execution_id: str,
        metadata: dict | None = None,
    ) -> WorkflowMetrics | None:
        """Mark a workflow as completed.

        Args:
            execution_id: Workflow execution ID
            metadata: Optional metadata

        Returns:
            Final WorkflowMetrics or None
        """
        with self._lock:
            workflow = self._active.pop(execution_id, None)
            if not workflow:
                return None

            workflow.complete()
            if metadata:
                workflow.metadata.update(metadata)

            self._history.append(workflow)
            if len(self._history) > self._max_history:
                self._history.pop(0)

            self._counters["workflows_completed"] += 1
            self._gauges["active_workflows"] = len(self._active)
            self._histograms["workflow_duration_ms"].append(workflow.duration_ms)
            self._histograms["workflow_tokens"].append(workflow.total_tokens)

        self._notify("workflow_completed", workflow)
        return workflow

    def fail_workflow(
        self,
        execution_id: str,
        error: str | None = None,
    ) -> WorkflowMetrics | None:
        """Mark a workflow as failed.

        Args:
            execution_id: Workflow execution ID
            error: Error message

        Returns:
            Final WorkflowMetrics or None
        """
        with self._lock:
            workflow = self._active.pop(execution_id, None)
            if not workflow:
                return None

            workflow.fail(error)

            self._history.append(workflow)
            if len(self._history) > self._max_history:
                self._history.pop(0)

            self._counters["workflows_failed"] += 1
            self._gauges["active_workflows"] = len(self._active)

        self._notify("workflow_failed", workflow)
        return workflow

    def get_active(self) -> list[WorkflowMetrics]:
        """Get all active workflow metrics."""
        with self._lock:
            return list(self._active.values())

    def get_history(self, limit: int = 100) -> list[WorkflowMetrics]:
        """Get workflow execution history.

        Args:
            limit: Maximum entries to return

        Returns:
            List of WorkflowMetrics, most recent first
        """
        with self._lock:
            return list(reversed(self._history[-limit:]))

    def get_summary(self) -> dict:
        """Get aggregated metrics summary."""
        with self._lock:
            # Calculate histogram stats
            def histogram_stats(values: list[float]) -> dict:
                if not values:
                    return {
                        "count": 0,
                        "min": 0,
                        "max": 0,
                        "avg": 0,
                        "p50": 0,
                        "p95": 0,
                    }
                sorted_vals = sorted(values)
                return {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p50": sorted_vals[len(sorted_vals) // 2],
                    "p95": sorted_vals[int(len(sorted_vals) * 0.95)]
                    if len(sorted_vals) > 1
                    else sorted_vals[0],
                }

            total_workflows = (
                self._counters["workflows_completed"] + self._counters["workflows_failed"]
            )
            success_rate = (
                self._counters["workflows_completed"] / total_workflows * 100
                if total_workflows > 0
                else 0
            )

            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "active_workflows": len(self._active),
                "total_executions": total_workflows,
                "success_rate": success_rate,
                "workflow_duration": histogram_stats(self._histograms["workflow_duration_ms"]),
                "workflow_tokens": histogram_stats(self._histograms["workflow_tokens"]),
                "step_duration": histogram_stats(self._histograms["step_duration_ms"]),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._active.clear()
            self._history.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# Global collector instance
_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _collector
    with _collector_lock:
        if _collector is None:
            _collector = MetricsCollector()
        return _collector
