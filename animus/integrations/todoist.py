"""
Todoist Integration

Task management via Todoist API.
"""

from __future__ import annotations

from typing import Any

from animus.integrations.base import AuthType, BaseIntegration
from animus.logging import get_logger
from animus.tools import Tool, ToolResult

logger = get_logger("integrations.todoist")

# Check if Todoist API is available
TODOIST_AVAILABLE = False
try:
    from todoist_api_python.api import TodoistAPI

    TODOIST_AVAILABLE = True
except ImportError:
    pass


class TodoistIntegration(BaseIntegration):
    """
    Todoist integration for task management.

    Provides tools for:
    - Listing tasks
    - Creating tasks
    - Completing tasks
    - Syncing with local TaskTracker
    """

    name = "todoist"
    display_name = "Todoist"
    auth_type = AuthType.API_KEY

    def __init__(self):
        super().__init__()
        self._api: Any = None  # TodoistAPI instance

    async def connect(self, credentials: dict[str, Any]) -> bool:
        """
        Connect to Todoist.

        Credentials:
            api_key: Todoist API token
        """
        if not TODOIST_AVAILABLE:
            self._set_error(
                "Todoist API not installed. Install with: pip install todoist-api-python"
            )
            return False

        api_key = credentials.get("api_key")
        if not api_key:
            self._set_error("API key required")
            return False

        try:
            self._api = TodoistAPI(api_key)
            # Verify by fetching projects
            self._api.get_projects()
            self._credentials = credentials
            self._set_connected()
            logger.info("Connected to Todoist")
            return True
        except Exception as e:
            self._set_error(f"Failed to connect: {e}")
            logger.error(f"Todoist connection failed: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from Todoist."""
        self._api = None
        self._set_disconnected()
        logger.info("Disconnected from Todoist")
        return True

    async def verify(self) -> bool:
        """Verify Todoist connection."""
        if not self._api:
            return False
        try:
            self._api.get_projects()
            return True
        except Exception:
            self._set_expired()
            return False

    def get_tools(self) -> list[Tool]:
        """Get Todoist tools."""
        return [
            Tool(
                name="todoist_list_tasks",
                description="List tasks from Todoist",
                parameters={
                    "project_id": {
                        "type": "string",
                        "description": "Filter by project ID (optional)",
                        "required": False,
                    },
                    "filter": {
                        "type": "string",
                        "description": "Todoist filter string (e.g., 'today', 'p1', '@work')",
                        "required": False,
                    },
                },
                handler=self._tool_list_tasks,
            ),
            Tool(
                name="todoist_create_task",
                description="Create a new task in Todoist",
                parameters={
                    "content": {
                        "type": "string",
                        "description": "Task content/title",
                        "required": True,
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Project ID to add task to",
                        "required": False,
                    },
                    "due_string": {
                        "type": "string",
                        "description": "Natural language due date (e.g., 'tomorrow', 'next monday')",
                        "required": False,
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority 1-4 (4 is highest)",
                        "required": False,
                    },
                    "labels": {
                        "type": "array",
                        "description": "List of label names",
                        "required": False,
                    },
                },
                handler=self._tool_create_task,
            ),
            Tool(
                name="todoist_complete_task",
                description="Mark a task as complete",
                parameters={
                    "task_id": {
                        "type": "string",
                        "description": "Task ID to complete",
                        "required": True,
                    },
                },
                handler=self._tool_complete_task,
            ),
            Tool(
                name="todoist_list_projects",
                description="List Todoist projects",
                parameters={},
                handler=self._tool_list_projects,
            ),
            Tool(
                name="todoist_sync",
                description="Sync tasks between Todoist and local TaskTracker",
                parameters={
                    "direction": {
                        "type": "string",
                        "description": "Sync direction: 'pull' (Todoist→local), 'push' (local→Todoist), or 'both'",
                        "required": False,
                    },
                },
                handler=self._tool_sync,
            ),
        ]

    async def _tool_list_tasks(
        self, project_id: str | None = None, filter: str | None = None
    ) -> ToolResult:
        """List tasks from Todoist."""
        if not self._api:
            return ToolResult(
                tool_name="todoist_tool",
                success=False,
                output=None,
                error="Not connected to Todoist",
            )

        try:
            if filter:
                tasks = self._api.get_tasks(filter=filter)
            elif project_id:
                tasks = self._api.get_tasks(project_id=project_id)
            else:
                tasks = self._api.get_tasks()

            task_list = [
                {
                    "id": task.id,
                    "content": task.content,
                    "description": task.description,
                    "project_id": task.project_id,
                    "priority": task.priority,
                    "due": task.due.string if task.due else None,
                    "labels": task.labels,
                    "created_at": task.created_at,
                }
                for task in tasks
            ]

            return ToolResult(
                tool_name="todoist_tool",
                success=True,
                output={
                    "count": len(task_list),
                    "tasks": task_list,
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="todoist_tool",
                success=False,
                output=None,
                error=f"Failed to list tasks: {e}",
            )

    async def _tool_create_task(
        self,
        content: str,
        project_id: str | None = None,
        due_string: str | None = None,
        priority: int | None = None,
        labels: list[str] | None = None,
    ) -> ToolResult:
        """Create a new task in Todoist."""
        if not self._api:
            return ToolResult(
                tool_name="todoist_tool",
                success=False,
                output=None,
                error="Not connected to Todoist",
            )

        try:
            kwargs: dict[str, Any] = {"content": content}
            if project_id:
                kwargs["project_id"] = project_id
            if due_string:
                kwargs["due_string"] = due_string
            if priority:
                kwargs["priority"] = priority
            if labels:
                kwargs["labels"] = labels

            task = self._api.add_task(**kwargs)

            return ToolResult(
                tool_name="todoist_tool",
                success=True,
                output={
                    "id": task.id,
                    "content": task.content,
                    "project_id": task.project_id,
                    "url": task.url,
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="todoist_tool",
                success=False,
                output=None,
                error=f"Failed to create task: {e}",
            )

    async def _tool_complete_task(self, task_id: str) -> ToolResult:
        """Complete a task."""
        if not self._api:
            return ToolResult(
                tool_name="todoist_tool",
                success=False,
                output=None,
                error="Not connected to Todoist",
            )

        try:
            self._api.close_task(task_id)
            return ToolResult(
                tool_name="todoist_tool",
                success=True,
                output={"task_id": task_id, "status": "completed"},
            )
        except Exception as e:
            return ToolResult(
                tool_name="todoist_tool",
                success=False,
                output=None,
                error=f"Failed to complete task: {e}",
            )

    async def _tool_list_projects(self) -> ToolResult:
        """List Todoist projects."""
        if not self._api:
            return ToolResult(
                tool_name="todoist_tool",
                success=False,
                output=None,
                error="Not connected to Todoist",
            )

        try:
            projects = self._api.get_projects()
            project_list = [
                {
                    "id": p.id,
                    "name": p.name,
                    "color": p.color,
                    "is_favorite": p.is_favorite,
                    "is_inbox": p.is_inbox_project,
                }
                for p in projects
            ]

            return ToolResult(
                tool_name="todoist_tool",
                success=True,
                output={
                    "count": len(project_list),
                    "projects": project_list,
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="todoist_tool",
                success=False,
                output=None,
                error=f"Failed to list projects: {e}",
            )

    async def _tool_sync(self, direction: str = "both") -> ToolResult:
        """
        Sync tasks between Todoist and local TaskTracker.

        Note: Actual sync implementation requires TaskTracker instance.
        This is a placeholder that returns sync info.
        """
        if not self._api:
            return ToolResult(
                tool_name="todoist_tool",
                success=False,
                output=None,
                error="Not connected to Todoist",
            )

        # For now, just return task counts
        try:
            tasks = self._api.get_tasks()
            return ToolResult(
                tool_name="todoist_tool",
                success=True,
                output={
                    "direction": direction,
                    "todoist_task_count": len(tasks),
                    "message": "Sync framework ready. Full bidirectional sync requires TaskTracker integration.",
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="todoist_tool", success=False, output=None, error=f"Sync failed: {e}"
            )
