"""
Animus Task Tracking

Basic task management for tracking work items.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from animus.logging import get_logger

logger = get_logger("tasks")


class TaskStatus(Enum):
    """Task status states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """A trackable task item."""

    id: str
    description: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    due_at: datetime | None = None
    completed_at: datetime | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    priority: int = 0  # Higher = more important

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "due_at": self.due_at.isoformat() if self.due_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "tags": self.tags,
            "notes": self.notes,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Task:
        return cls(
            id=data["id"],
            description=data["description"],
            status=TaskStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            due_at=datetime.fromisoformat(data["due_at"]) if data.get("due_at") else None,
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            priority=data.get("priority", 0),
        )

    def is_overdue(self) -> bool:
        """Check if task is past due date."""
        if not self.due_at:
            return False
        return datetime.now() > self.due_at and self.status != TaskStatus.COMPLETED


class TaskTracker:
    """
    Simple task management system.

    Persists tasks to JSON file for durability.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.tasks_file = data_dir / "tasks.json"
        self._tasks: dict[str, Task] = {}
        self._load()
        logger.debug(f"TaskTracker initialized at {data_dir}")

    def _load(self) -> None:
        """Load tasks from disk."""
        if self.tasks_file.exists():
            with open(self.tasks_file) as f:
                data = json.load(f)
                self._tasks = {k: Task.from_dict(v) for k, v in data.items()}
            logger.info(f"Loaded {len(self._tasks)} tasks")

    def _save(self) -> None:
        """Save tasks to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.tasks_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self._tasks.items()}, f, indent=2)

    def add(
        self,
        description: str,
        due_at: datetime | None = None,
        tags: list[str] | None = None,
        priority: int = 0,
    ) -> Task:
        """
        Add a new task.

        Args:
            description: Task description
            due_at: Optional due date
            tags: Optional tags
            priority: Priority level (higher = more important)

        Returns:
            Created Task
        """
        now = datetime.now()
        task = Task(
            id=str(uuid.uuid4()),
            description=description,
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            due_at=due_at,
            tags=[t.lower().strip() for t in (tags or [])],
            priority=priority,
        )
        self._tasks[task.id] = task
        self._save()
        logger.info(f"Added task: {description[:50]}...")
        return task

    def get(self, task_id: str) -> Task | None:
        """Get a task by ID or partial ID."""
        if task_id in self._tasks:
            return self._tasks[task_id]
        # Try partial match
        for tid, task in self._tasks.items():
            if tid.startswith(task_id):
                return task
        return None

    def update_status(self, task_id: str, status: TaskStatus) -> bool:
        """
        Update task status.

        Args:
            task_id: Task ID (or partial)
            status: New status

        Returns:
            True if updated
        """
        task = self.get(task_id)
        if not task:
            return False

        task.status = status
        task.updated_at = datetime.now()

        if status == TaskStatus.COMPLETED:
            task.completed_at = datetime.now()

        self._save()
        logger.debug(f"Task {task_id[:8]} status -> {status.value}")
        return True

    def start(self, task_id: str) -> bool:
        """Mark task as in progress."""
        return self.update_status(task_id, TaskStatus.IN_PROGRESS)

    def complete(self, task_id: str) -> bool:
        """Mark task as completed."""
        return self.update_status(task_id, TaskStatus.COMPLETED)

    def block(self, task_id: str) -> bool:
        """Mark task as blocked."""
        return self.update_status(task_id, TaskStatus.BLOCKED)

    def add_note(self, task_id: str, note: str) -> bool:
        """Add a note to a task."""
        task = self.get(task_id)
        if not task:
            return False

        if task.notes:
            task.notes += f"\n{note}"
        else:
            task.notes = note

        task.updated_at = datetime.now()
        self._save()
        return True

    def delete(self, task_id: str) -> bool:
        """Delete a task."""
        task = self.get(task_id)
        if not task:
            return False

        del self._tasks[task.id]
        self._save()
        logger.debug(f"Deleted task {task_id[:8]}")
        return True

    def list(
        self,
        status: TaskStatus | None = None,
        tags: list[str] | None = None,
        include_completed: bool = False,
    ) -> list[Task]:
        """
        List tasks with optional filters.

        Args:
            status: Filter by status
            tags: Filter by tags (all must match)
            include_completed: Include completed tasks (default: False)

        Returns:
            List of matching tasks, sorted by priority then due date
        """
        tasks = list(self._tasks.values())

        # Filter by status
        if status:
            tasks = [t for t in tasks if t.status == status]
        elif not include_completed:
            tasks = [t for t in tasks if t.status != TaskStatus.COMPLETED]

        # Filter by tags
        if tags:
            normalized_tags = [t.lower().strip() for t in tags]
            tasks = [t for t in tasks if all(tag in t.tags for tag in normalized_tags)]

        # Sort by priority (desc), then by due date (asc, None last)
        def sort_key(task: Task) -> tuple:
            due_sort = task.due_at if task.due_at else datetime.max
            return (-task.priority, due_sort)

        return sorted(tasks, key=sort_key)

    def list_all(
        self,
        status: TaskStatus | None = None,
        tags: list[str] | None = None,
        include_completed: bool = False,
    ) -> list[Task]:
        """Alias for list() for API consistency across modules."""
        return self.list(status=status, tags=tags, include_completed=include_completed)

    def list_overdue(self) -> list[Task]:
        """List overdue tasks."""
        return [t for t in self._tasks.values() if t.is_overdue()]

    def get_statistics(self) -> dict:
        """Get task statistics."""
        all_tasks = list(self._tasks.values())

        by_status = {}
        for task in all_tasks:
            by_status[task.status.value] = by_status.get(task.status.value, 0) + 1

        overdue = len(self.list_overdue())

        return {
            "total": len(all_tasks),
            "by_status": by_status,
            "overdue": overdue,
        }
