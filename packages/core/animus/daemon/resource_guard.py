"""ResourceGuard: token budgets, concurrency limits, and cooldowns.

Prevents runaway agents from exhausting API quotas, disk space, or CPU.
Singleton per daemon instance. All Head executions must check in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from animus.logging import get_logger

logger = get_logger("daemon.resource_guard")


@dataclass
class ResourceLimits:
    """Configurable resource caps for daemon operation."""

    # Token budgets (per time window)
    max_tokens_per_minute: int = 100_000
    max_tokens_per_hour: int = 1_000_000
    max_tokens_per_day: int = 5_000_000

    # Concurrency limits
    max_concurrent_tasks: int = 3
    max_consecutive_failures: int = 5  # Pause after N failures

    # Cooldown periods (seconds)
    task_cooldown_seconds: float = 1.0  # Min time between task starts
    failure_cooldown_seconds: float = 30.0  # Pause after failures
    global_cooldown_seconds: float = 0.0  # Emergency pause

    # Disk / memory guards
    max_log_size_mb: int = 100
    max_session_history: int = 50  # Warm sessions to keep


@dataclass
class TokenWindow:
    """Sliding window token consumption tracker."""

    window_seconds: int
    max_tokens: int
    _events: list[tuple[float, int]] = field(default_factory=list, repr=False)

    def consume(self, tokens: int) -> bool:
        """Record token consumption. Returns True if within limit."""
        now = time.time()
        cutoff = now - self.window_seconds
        # Prune old events
        self._events = [(t, n) for t, n in self._events if t > cutoff]
        self._events.append((now, tokens))
        return self.current <= self.max_tokens

    @property
    def current(self) -> int:
        return sum(n for _, n in self._events)

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.current)


class ResourceGuard:
    """Centralized resource governance for daemon operations.

    All task executions must acquire a slot and report token usage.
    Exceeding limits triggers cooldowns or hard stops.
    """

    def __init__(self, limits: ResourceLimits | None = None):
        self.limits = limits or ResourceLimits()
        self._active_tasks: set[str] = set()
        self._task_start_times: dict[str, float] = {}
        self._consecutive_failures: int = 0
        self._last_task_start: float = 0.0
        self._emergency_stop: bool = False

        # Token windows
        self._minute_window = TokenWindow(60, self.limits.max_tokens_per_minute)
        self._hour_window = TokenWindow(3600, self.limits.max_tokens_per_hour)
        self._day_window = TokenWindow(86400, self.limits.max_tokens_per_day)

    def acquire_task_slot(self, task_id: str) -> tuple[bool, str]:
        """Try to acquire a slot for task execution.

        Returns (granted, reason). If not granted, caller must retry later.
        """
        if self._emergency_stop:
            return False, "Emergency stop is active"

        # Check consecutive failures
        if self._consecutive_failures >= self.limits.max_consecutive_failures:
            return False, f"Too many consecutive failures ({self._consecutive_failures})"

        # Check concurrency
        if len(self._active_tasks) >= self.limits.max_concurrent_tasks:
            return False, f"Max concurrency reached ({self.limits.max_concurrent_tasks})"

        # Check cooldown
        now = time.time()
        elapsed = now - self._last_task_start
        if elapsed < self.limits.task_cooldown_seconds:
            wait = self.limits.task_cooldown_seconds - elapsed
            return False, f"Task cooldown active: wait {wait:.1f}s"

        # Check token budget
        if self._day_window.remaining <= 0:
            return False, "Daily token budget exhausted"

        # Grant slot
        self._active_tasks.add(task_id)
        self._task_start_times[task_id] = now
        self._last_task_start = now
        logger.debug(f"Task slot acquired: {task_id}")
        return True, ""

    def release_task_slot(self, task_id: str, tokens_used: int = 0, success: bool = True) -> None:
        """Release task slot and report resource usage."""
        self._active_tasks.discard(task_id)
        self._task_start_times.pop(task_id, None)

        # Record tokens
        if tokens_used > 0:
            self._minute_window.consume(tokens_used)
            self._hour_window.consume(tokens_used)
            self._day_window.consume(tokens_used)

        # Track failures
        if success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.limits.max_consecutive_failures:
                logger.warning(
                    f"Failure cooldown triggered: {self._consecutive_failures} consecutive failures"
                )

        logger.debug(f"Task slot released: {task_id}, tokens={tokens_used}, success={success}")

    def report_tokens(self, tokens: int) -> bool:
        """Report token usage outside of task lifecycle.

        Returns False if budget would be exceeded.
        """
        if tokens <= 0:
            return True

        ok = (
            self._minute_window.consume(tokens)
            and self._hour_window.consume(tokens)
            and self._day_window.consume(tokens)
        )
        if not ok:
            logger.warning(
                f"Token budget near limit: min={self._minute_window.remaining}, "
                f"hour={self._hour_window.remaining}, day={self._day_window.remaining}"
            )
        return ok

    def emergency_stop(self, reason: str = "") -> None:
        """Trigger emergency stop — no new tasks until cleared."""
        self._emergency_stop = True
        logger.error(f"Emergency stop triggered: {reason}")

    def emergency_clear(self) -> None:
        """Clear emergency stop."""
        self._emergency_stop = False
        self._consecutive_failures = 0
        logger.info("Emergency stop cleared")

    @property
    def can_execute(self) -> bool:
        """Quick check if daemon can accept new work."""
        if self._emergency_stop:
            return False
        if len(self._active_tasks) >= self.limits.max_concurrent_tasks:
            return False
        if self._day_window.remaining <= 0:
            return False
        return True

    @property
    def status(self) -> dict[str, Any]:
        """Current resource status for monitoring."""
        return {
            "active_tasks": len(self._active_tasks),
            "max_concurrent": self.limits.max_concurrent_tasks,
            "consecutive_failures": self._consecutive_failures,
            "emergency_stop": self._emergency_stop,
            "tokens_remaining": {
                "minute": self._minute_window.remaining,
                "hour": self._hour_window.remaining,
                "day": self._day_window.remaining,
            },
        }
