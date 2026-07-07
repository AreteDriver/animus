"""TaskScheduler: cron-like scheduling with natural-language task descriptions.

Supports three schedule types:
- INTERVAL: Run every N seconds/minutes/hours
- CRON: Standard cron expression
- ONE_SHOT: Run once at a specific time

Tasks are stored persistently and survive daemon restarts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from animus.logging import get_logger

logger = get_logger("daemon.scheduler")


class ScheduleType(Enum):
    """Types of task schedules."""

    INTERVAL = "interval"  # Every N seconds
    CRON = "cron"  # Standard cron expression
    ONE_SHOT = "one_shot"  # Run once at specific time


@dataclass
class ScheduledTask:
    """A task scheduled for background execution."""

    task_id: str
    description: str  # Natural language description
    schedule_type: ScheduleType
    schedule_config: dict[str, Any]  # type-specific config
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_run: str | None = None
    next_run: str | None = None
    run_count: int = 0
    max_runs: int | None = None  # None = unlimited
    enabled: bool = True
    priority: str = "normal"  # normal, high, critical
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_due(self) -> bool:
        """Check if task should run now."""
        if not self.enabled:
            return False
        if self.max_runs is not None and self.run_count >= self.max_runs:
            return False

        if self.schedule_type == ScheduleType.ONE_SHOT:
            if self.next_run is None:
                return False
            return datetime.now() >= datetime.fromisoformat(self.next_run)

        if self.schedule_type == ScheduleType.INTERVAL:
            if self.next_run is None:
                return True
            return datetime.now() >= datetime.fromisoformat(self.next_run)

        if self.schedule_type == ScheduleType.CRON:
            # Simplified: check if we're in a new minute/hour/day that matches
            return self._check_cron()

        return False

    def _check_cron(self) -> bool:
        """Simplified cron check — full cron parsing can be added later."""
        cron_expr = self.schedule_config.get("expression", "")
        # For now, support basic patterns like "*/15 * * * *" (every 15 min)
        parts = cron_expr.split()
        if len(parts) != 5:
            return False

        now = datetime.now()
        minute, hour, dom, month, dow = parts

        # Check month
        if month != "*" and now.month != int(month):
            return False

        # Check hour
        if hour != "*" and now.hour != int(hour):
            return False

        # Check minute (support */N)
        if minute.startswith("*/"):
            interval = int(minute[2:])
            if now.minute % interval != 0:
                return False
        elif minute != "*" and now.minute != int(minute):
            return False

        # Check day of week (0=Monday in cron)
        if dow != "*":
            target_dow = int(dow) % 7
            if now.weekday() != target_dow:
                return False

        return True

    def mark_run(self) -> None:
        """Update task after execution."""
        self.last_run = datetime.now().isoformat()
        self.run_count += 1

        if self.schedule_type == ScheduleType.INTERVAL:
            interval_seconds = self.schedule_config.get("seconds", 3600)
            next_time = datetime.now() + timedelta(seconds=interval_seconds)
            self.next_run = next_time.isoformat()

        elif self.schedule_type == ScheduleType.ONE_SHOT:
            self.enabled = False  # Disable after one run
            self.next_run = None

        # CRON: next_run is implicit (checked every minute)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "schedule_type": self.schedule_type.value,
            "schedule_config": self.schedule_config,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "run_count": self.run_count,
            "max_runs": self.max_runs,
            "enabled": self.enabled,
            "priority": self.priority,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduledTask":
        return cls(
            task_id=data["task_id"],
            description=data["description"],
            schedule_type=ScheduleType(data["schedule_type"]),
            schedule_config=data.get("schedule_config", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_run=data.get("last_run"),
            next_run=data.get("next_run"),
            run_count=data.get("run_count", 0),
            max_runs=data.get("max_runs"),
            enabled=data.get("enabled", True),
            priority=data.get("priority", "normal"),
            metadata=data.get("metadata", {}),
        )


class TaskScheduler:
    """Persistent task scheduler for daemon background execution.

    Stores tasks in JSON and checks for due tasks on each tick.
    """

    def __init__(self, persistence_dir: str | Path | None = None):
        self.persistence_dir = Path(persistence_dir or "~/.animus/scheduler").expanduser()
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, ScheduledTask] = {}
        self._load_existing()

    def _task_path(self, task_id: str) -> Path:
        return self.persistence_dir / f"{task_id}.json"

    def _load_existing(self) -> None:
        """Load persisted tasks on startup."""
        for path in self.persistence_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                task = ScheduledTask.from_dict(data)
                self._tasks[task.task_id] = task
                logger.debug(f"Loaded scheduled task: {task.task_id}")
            except Exception as e:
                logger.warning(f"Failed to load task from {path}: {e}")

    def _persist(self, task: ScheduledTask) -> None:
        try:
            path = self._task_path(task.task_id)
            path.write_text(json.dumps(task.to_dict(), indent=2))
        except Exception as e:
            logger.error(f"Failed to persist task {task.task_id}: {e}")

    def schedule_interval(
        self,
        description: str,
        seconds: int,
        max_runs: int | None = None,
        priority: str = "normal",
        metadata: dict | None = None,
    ) -> ScheduledTask:
        """Schedule a task to run at regular intervals."""
        task_id = f"task-{int(time.time()*1000)}"
        task = ScheduledTask(
            task_id=task_id,
            description=description,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={"seconds": seconds},
            max_runs=max_runs,
            priority=priority,
            metadata=metadata or {},
        )
        task.next_run = datetime.now().isoformat()
        self._tasks[task_id] = task
        self._persist(task)
        logger.info(f"Scheduled interval task: {task_id} (every {seconds}s)")
        return task

    def schedule_one_shot(
        self,
        description: str,
        run_at: datetime,
        priority: str = "normal",
        metadata: dict | None = None,
    ) -> ScheduledTask:
        """Schedule a one-time task."""
        task_id = f"task-{int(time.time()*1000)}"
        task = ScheduledTask(
            task_id=task_id,
            description=description,
            schedule_type=ScheduleType.ONE_SHOT,
            schedule_config={"target": run_at.isoformat()},
            max_runs=1,
            priority=priority,
            metadata=metadata or {},
        )
        task.next_run = run_at.isoformat()
        self._tasks[task_id] = task
        self._persist(task)
        logger.info(f"Scheduled one-shot task: {task_id} at {run_at}")
        return task

    def schedule_cron(
        self,
        description: str,
        cron_expression: str,
        max_runs: int | None = None,
        priority: str = "normal",
        metadata: dict | None = None,
    ) -> ScheduledTask:
        """Schedule a task using cron expression."""
        task_id = f"task-{int(time.time()*1000)}"
        task = ScheduledTask(
            task_id=task_id,
            description=description,
            schedule_type=ScheduleType.CRON,
            schedule_config={"expression": cron_expression},
            max_runs=max_runs,
            priority=priority,
            metadata=metadata or {},
        )
        self._tasks[task_id] = task
        self._persist(task)
        logger.info(f"Scheduled cron task: {task_id} ({cron_expression})")
        return task

    def get_due_tasks(self) -> list[ScheduledTask]:
        """Get all tasks that are due to run."""
        return [t for t in self._tasks.values() if t.is_due]

    def get_task(self, task_id: str) -> ScheduledTask | None:
        return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """Cancel and remove a task."""
        task = self._tasks.pop(task_id, None)
        if task:
            path = self._task_path(task_id)
            if path.exists():
                path.unlink()
            logger.info(f"Cancelled task: {task_id}")
            return True
        return False

    def mark_run(self, task_id: str) -> None:
        """Mark a task as executed."""
        task = self._tasks.get(task_id)
        if task:
            task.mark_run()
            self._persist(task)

    def list_tasks(self, enabled_only: bool = False) -> list[ScheduledTask]:
        """List all tasks."""
        tasks = list(self._tasks.values())
        if enabled_only:
            tasks = [t for t in tasks if t.enabled]
        return sorted(tasks, key=lambda t: t.created_at)

    @property
    def task_count(self) -> int:
        return len(self._tasks)