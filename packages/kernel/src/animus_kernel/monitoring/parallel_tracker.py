"""Parallel Execution Metrics for Gorgon Workflows.

Tracks fan-out, fan-in, map-reduce, and auto-parallel execution patterns
with real-time visibility into concurrent operations.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ParallelPatternType(Enum):
    """Types of parallel execution patterns."""

    FAN_OUT = "fan_out"
    FAN_IN = "fan_in"
    MAP_REDUCE = "map_reduce"
    AUTO_PARALLEL = "auto_parallel"
    PARALLEL_GROUP = "parallel_group"


@dataclass
class BranchMetrics:
    """Metrics for a single parallel branch/item."""

    branch_id: str
    parent_id: str
    item_index: int
    item_value: Any = None
    status: str = "pending"  # pending, running, success, failed, cancelled
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float = 0
    tokens_used: int = 0
    error: str | None = None
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Mark branch as started."""
        self.started_at = datetime.now(UTC)
        self.status = "running"

    def complete(self, tokens: int = 0) -> None:
        """Mark branch as completed."""
        self.completed_at = datetime.now(UTC)
        self.status = "success"
        self.tokens_used = tokens
        if self.started_at:
            self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000

    def fail(self, error: str) -> None:
        """Mark branch as failed."""
        self.completed_at = datetime.now(UTC)
        self.status = "failed"
        self.error = error
        if self.started_at:
            self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000

    def cancel(self) -> None:
        """Mark branch as cancelled."""
        self.completed_at = datetime.now(UTC)
        self.status = "cancelled"
        if self.started_at:
            self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "branch_id": self.branch_id,
            "parent_id": self.parent_id,
            "item_index": self.item_index,
            "item_value": str(self.item_value) if self.item_value else None,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "error": self.error,
            "retries": self.retries,
        }


@dataclass
class ParallelExecutionMetrics:
    """Metrics for a parallel execution (fan-out, map-reduce, etc.)."""

    execution_id: str
    pattern_type: ParallelPatternType
    step_id: str
    workflow_id: str | None = None
    total_items: int = 0
    max_concurrent: int = 0
    status: str = "pending"  # pending, running, completed, failed
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float = 0
    branches: dict[str, BranchMetrics] = field(default_factory=dict)
    # Rate limiting metrics
    rate_limit_waits: int = 0
    rate_limit_wait_ms: float = 0
    # Aggregated metrics
    total_tokens: int = 0
    successful_count: int = 0
    failed_count: int = 0
    cancelled_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Mark execution as started."""
        self.started_at = datetime.now(UTC)
        self.status = "running"

    def complete(self) -> None:
        """Mark execution as completed."""
        self.completed_at = datetime.now(UTC)
        self.status = "completed"
        self._finalize()

    def fail(self, error: str | None = None) -> None:
        """Mark execution as failed."""
        self.completed_at = datetime.now(UTC)
        self.status = "failed"
        if error:
            self.metadata["error"] = error
        self._finalize()

    def _finalize(self) -> None:
        """Calculate final aggregate metrics."""
        if self.started_at and self.completed_at:
            self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000

        self.total_tokens = sum(b.tokens_used for b in self.branches.values())
        self.successful_count = sum(1 for b in self.branches.values() if b.status == "success")
        self.failed_count = sum(1 for b in self.branches.values() if b.status == "failed")
        self.cancelled_count = sum(1 for b in self.branches.values() if b.status == "cancelled")

    def add_branch(
        self,
        branch_id: str,
        item_index: int,
        item_value: Any = None,
    ) -> BranchMetrics:
        """Add a branch to track."""
        branch = BranchMetrics(
            branch_id=branch_id,
            parent_id=self.execution_id,
            item_index=item_index,
            item_value=item_value,
        )
        self.branches[branch_id] = branch
        return branch

    def get_branch(self, branch_id: str) -> BranchMetrics | None:
        """Get branch metrics by ID."""
        return self.branches.get(branch_id)

    def record_rate_limit_wait(self, wait_ms: float) -> None:
        """Record a rate limit wait event."""
        self.rate_limit_waits += 1
        self.rate_limit_wait_ms += wait_ms

    @property
    def active_branch_count(self) -> int:
        """Count of currently running branches."""
        return sum(1 for b in self.branches.values() if b.status == "running")

    @property
    def pending_branch_count(self) -> int:
        """Count of pending branches."""
        return sum(1 for b in self.branches.values() if b.status == "pending")

    @property
    def completion_ratio(self) -> float:
        """Ratio of completed branches (success + failed) to total."""
        if not self.branches:
            return 0.0
        completed = sum(1 for b in self.branches.values() if b.status in ("success", "failed"))
        return completed / len(self.branches)

    @property
    def success_rate(self) -> float:
        """Success rate of completed branches."""
        completed = self.successful_count + self.failed_count
        if completed == 0:
            return 0.0
        return self.successful_count / completed

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "pattern_type": self.pattern_type.value,
            "step_id": self.step_id,
            "workflow_id": self.workflow_id,
            "total_items": self.total_items,
            "max_concurrent": self.max_concurrent,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "total_tokens": self.total_tokens,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "cancelled_count": self.cancelled_count,
            "active_branches": self.active_branch_count,
            "pending_branches": self.pending_branch_count,
            "completion_ratio": self.completion_ratio,
            "success_rate": self.success_rate,
            "rate_limit_waits": self.rate_limit_waits,
            "rate_limit_wait_ms": self.rate_limit_wait_ms,
            "branches": {k: v.to_dict() for k, v in self.branches.items()},
        }


@dataclass
class RateLimitState:
    """Current state of rate limiting for a provider."""

    provider: str
    current_concurrent: int = 0
    max_concurrent: int = 0
    waiting_count: int = 0
    total_requests: int = 0
    total_wait_ms: float = 0
    last_request_at: datetime | None = None
    # Adaptive rate limiting fields
    base_limit: int = 0
    current_limit: int = 0
    total_429s: int = 0
    is_throttled: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "current_concurrent": self.current_concurrent,
            "max_concurrent": self.max_concurrent,
            "waiting_count": self.waiting_count,
            "utilization": (
                self.current_concurrent / self.max_concurrent * 100
                if self.max_concurrent > 0
                else 0
            ),
            "total_requests": self.total_requests,
            "avg_wait_ms": (
                self.total_wait_ms / self.total_requests if self.total_requests > 0 else 0
            ),
            "last_request_at": (self.last_request_at.isoformat() if self.last_request_at else None),
            # Adaptive rate limiting
            "base_limit": self.base_limit,
            "current_limit": self.current_limit,
            "total_429s": self.total_429s,
            "is_throttled": self.is_throttled,
        }


class ParallelExecutionTracker:
    """Tracks parallel execution patterns across workflows.

    Thread-safe tracking of fan-out, fan-in, map-reduce, and auto-parallel
    executions with real-time metrics and callbacks.
    """

    def __init__(self, max_history: int = 100):
        """Initialize parallel execution tracker.

        Args:
            max_history: Maximum completed executions to keep in history
        """
        self._lock = threading.Lock()
        self._active: dict[str, ParallelExecutionMetrics] = {}
        self._history: list[ParallelExecutionMetrics] = []
        self._max_history = max_history
        self._callbacks: list[Callable[[str, ParallelExecutionMetrics], None]] = []

        # Rate limit tracking by provider
        self._rate_limits: dict[str, RateLimitState] = {}

        # Aggregated counters
        self._counters: dict[str, int] = defaultdict(int)
        self._timings: dict[str, list[float]] = defaultdict(list)

    def register_callback(self, callback: Callable[[str, ParallelExecutionMetrics], None]) -> None:
        """Register callback for parallel execution events.

        Args:
            callback: Function called with (event_type, metrics)
        """
        self._callbacks.append(callback)

    def _notify(self, event_type: str, metrics: ParallelExecutionMetrics) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event_type, metrics)
            except Exception:
                pass  # Best-effort cleanup: callback failure must not break tracking

    def start_execution(
        self,
        execution_id: str,
        pattern_type: ParallelPatternType,
        step_id: str,
        total_items: int,
        max_concurrent: int,
        workflow_id: str | None = None,
        metadata: dict | None = None,
    ) -> ParallelExecutionMetrics:
        """Start tracking a parallel execution.

        Args:
            execution_id: Unique execution identifier
            pattern_type: Type of parallel pattern
            step_id: Parent step ID
            total_items: Total items to process
            max_concurrent: Maximum concurrent operations
            workflow_id: Optional workflow ID
            metadata: Optional metadata

        Returns:
            ParallelExecutionMetrics instance
        """
        with self._lock:
            metrics = ParallelExecutionMetrics(
                execution_id=execution_id,
                pattern_type=pattern_type,
                step_id=step_id,
                workflow_id=workflow_id,
                total_items=total_items,
                max_concurrent=max_concurrent,
                metadata=metadata or {},
            )
            metrics.start()
            self._active[execution_id] = metrics

            self._counters["executions_started"] += 1
            self._counters[f"executions_started_{pattern_type.value}"] += 1

        self._notify("execution_started", metrics)
        return metrics

    def start_branch(
        self,
        execution_id: str,
        branch_id: str,
        item_index: int,
        item_value: Any = None,
    ) -> BranchMetrics | None:
        """Start tracking a branch within a parallel execution.

        Args:
            execution_id: Parent execution ID
            branch_id: Unique branch identifier
            item_index: Index of item being processed
            item_value: Optional item value

        Returns:
            BranchMetrics instance or None
        """
        with self._lock:
            execution = self._active.get(execution_id)
            if not execution:
                return None

            branch = execution.add_branch(branch_id, item_index, item_value)
            branch.start()

            self._counters["branches_started"] += 1

        return branch

    def complete_branch(
        self,
        execution_id: str,
        branch_id: str,
        tokens: int = 0,
        metadata: dict | None = None,
    ) -> None:
        """Mark a branch as completed.

        Args:
            execution_id: Parent execution ID
            branch_id: Branch identifier
            tokens: Tokens used
            metadata: Optional metadata
        """
        with self._lock:
            execution = self._active.get(execution_id)
            if not execution:
                return

            branch = execution.get_branch(branch_id)
            if not branch:
                return

            branch.complete(tokens)
            if metadata:
                branch.metadata.update(metadata)

            self._counters["branches_completed"] += 1
            self._timings["branch_duration_ms"].append(branch.duration_ms)

    def fail_branch(
        self,
        execution_id: str,
        branch_id: str,
        error: str,
    ) -> None:
        """Mark a branch as failed.

        Args:
            execution_id: Parent execution ID
            branch_id: Branch identifier
            error: Error message
        """
        with self._lock:
            execution = self._active.get(execution_id)
            if not execution:
                return

            branch = execution.get_branch(branch_id)
            if not branch:
                return

            branch.fail(error)
            self._counters["branches_failed"] += 1

    def cancel_branch(
        self,
        execution_id: str,
        branch_id: str,
    ) -> None:
        """Mark a branch as cancelled.

        Args:
            execution_id: Parent execution ID
            branch_id: Branch identifier
        """
        with self._lock:
            execution = self._active.get(execution_id)
            if not execution:
                return

            branch = execution.get_branch(branch_id)
            if not branch:
                return

            branch.cancel()
            self._counters["branches_cancelled"] += 1

    def complete_execution(
        self,
        execution_id: str,
        metadata: dict | None = None,
    ) -> ParallelExecutionMetrics | None:
        """Mark a parallel execution as completed.

        Args:
            execution_id: Execution identifier
            metadata: Optional metadata

        Returns:
            Final ParallelExecutionMetrics or None
        """
        with self._lock:
            execution = self._active.pop(execution_id, None)
            if not execution:
                return None

            execution.complete()
            if metadata:
                execution.metadata.update(metadata)

            self._history.append(execution)
            if len(self._history) > self._max_history:
                self._history.pop(0)

            self._counters["executions_completed"] += 1
            self._timings["execution_duration_ms"].append(execution.duration_ms)
            self._timings["execution_tokens"].append(execution.total_tokens)

        self._notify("execution_completed", execution)
        return execution

    def fail_execution(
        self,
        execution_id: str,
        error: str | None = None,
    ) -> ParallelExecutionMetrics | None:
        """Mark a parallel execution as failed.

        Args:
            execution_id: Execution identifier
            error: Error message

        Returns:
            Final ParallelExecutionMetrics or None
        """
        with self._lock:
            execution = self._active.pop(execution_id, None)
            if not execution:
                return None

            execution.fail(error)

            self._history.append(execution)
            if len(self._history) > self._max_history:
                self._history.pop(0)

            self._counters["executions_failed"] += 1

        self._notify("execution_failed", execution)
        return execution

    def record_rate_limit_wait(
        self,
        execution_id: str,
        provider: str,
        wait_ms: float,
    ) -> None:
        """Record a rate limit wait event.

        Args:
            execution_id: Execution experiencing the wait
            provider: Provider being rate limited
            wait_ms: Wait time in milliseconds
        """
        with self._lock:
            execution = self._active.get(execution_id)
            if execution:
                execution.record_rate_limit_wait(wait_ms)

            # Update provider rate limit state
            if provider not in self._rate_limits:
                self._rate_limits[provider] = RateLimitState(provider=provider)

            state = self._rate_limits[provider]
            state.total_requests += 1
            state.total_wait_ms += wait_ms

            self._counters["rate_limit_waits"] += 1
            self._timings["rate_limit_wait_ms"].append(wait_ms)

    def update_rate_limit_state(
        self,
        provider: str,
        current_limit: int | None = None,
        base_limit: int | None = None,
        total_429s: int | None = None,
        is_throttled: bool | None = None,
        current_concurrent: int | None = None,
        max_concurrent: int | None = None,
        waiting_count: int | None = None,
    ) -> None:
        """Update rate limit state for a provider.

        Args:
            provider: Provider name
            current_limit: Current adaptive rate limit
            base_limit: Base (original) rate limit
            total_429s: Total 429 errors encountered
            is_throttled: Whether provider is currently throttled
            current_concurrent: Current concurrent requests
            max_concurrent: Maximum allowed concurrent
            waiting_count: Number of requests waiting
        """
        with self._lock:
            if provider not in self._rate_limits:
                self._rate_limits[provider] = RateLimitState(provider=provider)

            state = self._rate_limits[provider]

            # Update adaptive rate limit fields
            if current_limit is not None:
                state.current_limit = current_limit
            if base_limit is not None:
                state.base_limit = base_limit
            if total_429s is not None:
                state.total_429s = total_429s
            if is_throttled is not None:
                state.is_throttled = is_throttled

            # Update concurrent tracking fields
            if current_concurrent is not None:
                state.current_concurrent = current_concurrent
            if max_concurrent is not None:
                state.max_concurrent = max_concurrent
            if waiting_count is not None:
                state.waiting_count = waiting_count

            state.last_request_at = datetime.now(UTC)

    def get_active_executions(self) -> list[dict]:
        """Get all active parallel executions."""
        with self._lock:
            return [e.to_dict() for e in self._active.values()]

    def get_execution(self, execution_id: str) -> dict | None:
        """Get a specific execution by ID."""
        with self._lock:
            execution = self._active.get(execution_id)
            if execution:
                return execution.to_dict()
            # Check history
            for e in self._history:
                if e.execution_id == execution_id:
                    return e.to_dict()
            return None

    def get_history(self, limit: int = 50) -> list[dict]:
        """Get recent completed executions.

        Args:
            limit: Maximum entries to return

        Returns:
            List of execution dictionaries, most recent first
        """
        with self._lock:
            return [e.to_dict() for e in reversed(self._history[-limit:])]

    def get_rate_limit_states(self) -> dict[str, dict]:
        """Get current rate limit states for all providers."""
        with self._lock:
            return self._get_rate_limit_states_unlocked()

    def _get_rate_limit_states_unlocked(self) -> dict[str, dict]:
        """Internal: Get rate limit states without acquiring lock."""
        return {k: v.to_dict() for k, v in self._rate_limits.items()}

    def get_summary(self) -> dict:
        """Get aggregated parallel execution summary."""
        with self._lock:

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
                    "min": round(min(values), 2),
                    "max": round(max(values), 2),
                    "avg": round(sum(values) / len(values), 2),
                    "p50": round(sorted_vals[len(sorted_vals) // 2], 2),
                    "p95": round(
                        sorted_vals[int(len(sorted_vals) * 0.95)]
                        if len(sorted_vals) > 1
                        else sorted_vals[0],
                        2,
                    ),
                }

            total_executions = (
                self._counters["executions_completed"] + self._counters["executions_failed"]
            )
            success_rate = (
                self._counters["executions_completed"] / total_executions * 100
                if total_executions > 0
                else 0
            )

            # Active execution stats
            active_branches = sum(e.active_branch_count for e in self._active.values())
            pending_branches = sum(e.pending_branch_count for e in self._active.values())

            return {
                "active_executions": len(self._active),
                "active_branches": active_branches,
                "pending_branches": pending_branches,
                "total_executions": total_executions,
                "success_rate": round(success_rate, 1),
                "counters": dict(self._counters),
                "execution_duration": histogram_stats(self._timings["execution_duration_ms"]),
                "branch_duration": histogram_stats(self._timings["branch_duration_ms"]),
                "execution_tokens": histogram_stats(self._timings["execution_tokens"]),
                "rate_limit_waits": histogram_stats(self._timings["rate_limit_wait_ms"]),
                "rate_limit_states": self._get_rate_limit_states_unlocked(),
            }

    def get_dashboard_data(self) -> dict:
        """Get all data needed for dashboard display."""
        return {
            "summary": self.get_summary(),
            "active_executions": self.get_active_executions(),
            "recent_executions": self.get_history(20),
            "rate_limits": self.get_rate_limit_states(),
        }

    def reset(self) -> None:
        """Reset all tracking data."""
        with self._lock:
            self._active.clear()
            self._history.clear()
            self._rate_limits.clear()
            self._counters.clear()
            self._timings.clear()


# Global tracker instance
_parallel_tracker: ParallelExecutionTracker | None = None
_tracker_lock = threading.Lock()


def get_parallel_tracker() -> ParallelExecutionTracker:
    """Get or create global parallel execution tracker."""
    global _parallel_tracker
    with _tracker_lock:
        if _parallel_tracker is None:
            _parallel_tracker = ParallelExecutionTracker()
        return _parallel_tracker
