"""Task management tools — create, list, complete, delete local tasks."""

from __future__ import annotations

import logging

from animus_bootstrap.intelligence.tools.executor import ToolDefinition

logger = logging.getLogger(__name__)

# Persistent store reference — set at runtime
_task_store = None


def set_task_store(store: object) -> None:
    """Wire the persistent task store."""
    global _task_store  # noqa: PLW0603
    _task_store = store


def get_task_store() -> object | None:
    """Return the current task store (for testing/inspection)."""
    return _task_store


async def _task_create(
    name: str, description: str = "", priority: str = "normal", due_date: str = ""
) -> str:
    """Create a new local task."""
    if _task_store is None:
        return "Task store not available"

    valid_priorities = ("low", "normal", "high", "urgent")
    if priority not in valid_priorities:
        return f"Invalid priority '{priority}'. Must be one of: {', '.join(valid_priorities)}"

    task_id = _task_store.create(
        name=name, description=description, priority=priority, due_date=due_date
    )
    logger.info("Task '%s' created (id=%s)", name, task_id)
    return f"Task created: '{name}' (id: {task_id}, priority: {priority})"


async def _task_list(status: str = "") -> str:
    """List all tasks, optionally filtered by status."""
    if _task_store is None:
        return "Task store not available"

    valid_statuses = ("", "pending", "in_progress", "completed")
    if status not in valid_statuses:
        return f"Invalid status '{status}'. Must be one of: pending, in_progress, completed"

    tasks = _task_store.list_all(status=status)
    if not tasks:
        qualifier = f" with status '{status}'" if status else ""
        return f"No tasks found{qualifier}."

    lines = []
    for t in tasks:
        due = f", due: {t['due_date']}" if t["due_date"] else ""
        lines.append(f"  [{t['id']}] {t['name']} — {t['status']} ({t['priority']}{due})")

    header = f"Tasks ({len(tasks)})"
    if status:
        header += f" [status={status}]"
    return f"{header}:\n" + "\n".join(lines)


async def _task_complete(task_id: str) -> str:
    """Mark a task as completed."""
    if _task_store is None:
        return "Task store not available"

    if _task_store.complete(task_id):
        logger.info("Task '%s' completed", task_id)
        return f"Task '{task_id}' marked as completed"
    return f"Task '{task_id}' not found"


async def _task_delete(task_id: str) -> str:
    """Delete a task."""
    if _task_store is None:
        return "Task store not available"

    if _task_store.delete(task_id):
        logger.info("Task '%s' deleted", task_id)
        return f"Task '{task_id}' deleted"
    return f"Task '{task_id}' not found"


def get_task_tools() -> list[ToolDefinition]:
    """Return task management tool definitions."""
    return [
        ToolDefinition(
            name="task_create",
            description=(
                "Create a new local task with optional priority and due date. "
                "Priority: low, normal, high, urgent. "
                "Due date: ISO format (e.g. '2025-03-15T09:00:00')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name/title of the task.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the task.",
                        "default": "",
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority level: low, normal, high, urgent.",
                        "default": "normal",
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Due date in ISO format.",
                        "default": "",
                    },
                },
                "required": ["name"],
            },
            handler=_task_create,
            category="task",
        ),
        ToolDefinition(
            name="task_list",
            description=(
                "List all tasks, optionally filtered by status (pending, in_progress, completed)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status: pending, in_progress, completed.",
                        "default": "",
                    },
                },
            },
            handler=_task_list,
            category="task",
        ),
        ToolDefinition(
            name="task_complete",
            description="Mark a task as completed by its ID.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "ID of the task to complete.",
                    },
                },
                "required": ["task_id"],
            },
            handler=_task_complete,
            category="task",
        ),
        ToolDefinition(
            name="task_delete",
            description="Delete a task by its ID.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "ID of the task to delete.",
                    },
                },
                "required": ["task_id"],
            },
            handler=_task_delete,
            category="task",
        ),
    ]
