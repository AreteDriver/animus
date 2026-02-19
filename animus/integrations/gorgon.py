"""
Gorgon Integration — Connects Animus to Gorgon's task queue API.

Gorgon is a headless multi-agent orchestration engine. This integration
allows Animus to delegate complex tasks (code review, refactoring, testing,
security audits) to Gorgon's autonomous agent pipeline.

Requires: httpx (optional dependency)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from animus.integrations.base import AuthType, BaseIntegration
from animus.logging import get_logger
from animus.tools import Tool, ToolResult

logger = get_logger("integrations.gorgon")

# Lazy import — httpx is an optional dependency
HTTPX_AVAILABLE = False
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore[assignment]


class GorgonClient:
    """Async HTTP client for Gorgon's task queue API."""

    def __init__(
        self,
        url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def check_health(self) -> dict[str, Any]:
        """Check Gorgon API health."""
        client = await self._ensure_client()
        resp = await client.get("/health")
        resp.raise_for_status()
        return resp.json()

    async def submit_task(
        self,
        title: str,
        description: str,
        priority: int = 5,
        agent_role: str | None = None,
    ) -> dict[str, Any]:
        """Submit a task to the agent queue."""
        client = await self._ensure_client()
        payload: dict[str, Any] = {
            "title": title,
            "description": description,
            "priority": priority,
        }
        if agent_role:
            payload["agent_role"] = agent_role
        resp = await client.post("/v1/tasks", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def get_task(self, task_id: str) -> dict[str, Any]:
        """Get task status and result."""
        client = await self._ensure_client()
        resp = await client.get(f"/v1/tasks/{task_id}")
        resp.raise_for_status()
        return resp.json()

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Cancel a pending task."""
        client = await self._ensure_client()
        resp = await client.post(f"/v1/tasks/{task_id}/cancel")
        resp.raise_for_status()
        return resp.json()

    async def get_stats(self) -> dict[str, Any]:
        """Get task queue statistics."""
        client = await self._ensure_client()
        resp = await client.get("/v1/tasks/stats")
        resp.raise_for_status()
        return resp.json()

    async def list_tasks(self, limit: int = 10, status: str | None = None) -> list[dict]:
        """List recent tasks."""
        client = await self._ensure_client()
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        resp = await client.get("/v1/tasks", params=params)
        resp.raise_for_status()
        return resp.json()

    async def submit_and_wait(
        self,
        title: str,
        description: str,
        priority: int = 5,
        agent_role: str | None = None,
        poll_interval: float = 5.0,
        max_wait: float = 300.0,
    ) -> dict[str, Any]:
        """Submit a task and poll until completion."""
        task = await self.submit_task(title, description, priority, agent_role)
        task_id = task["id"]
        start = time.monotonic()

        while time.monotonic() - start < max_wait:
            result = await self.get_task(task_id)
            if result.get("status") in ("completed", "failed", "cancelled"):
                return result
            await asyncio.sleep(poll_interval)

        return await self.get_task(task_id)


class GorgonIntegration(BaseIntegration):
    """Integration with Gorgon multi-agent orchestration engine.

    Provides tools for delegating complex tasks to Gorgon's autonomous
    agent pipeline: code review, refactoring, testing, security audits, etc.
    """

    name = "gorgon"
    display_name = "Gorgon"
    auth_type = AuthType.NONE

    def __init__(self):
        super().__init__()
        self._client: GorgonClient | None = None

    async def connect(self, credentials: dict[str, Any]) -> bool:
        """Connect to Gorgon API."""
        if not HTTPX_AVAILABLE:
            self._set_error("httpx not installed. Install with: pip install httpx")
            return False

        url = credentials.get("url", "http://localhost:8000")
        api_key = credentials.get("api_key")
        timeout = float(credentials.get("timeout", 30.0))

        try:
            self._client = GorgonClient(url=url, api_key=api_key, timeout=timeout)
            health = await self._client.check_health()
            self._credentials = credentials
            self._set_connected()
            logger.info(f"Connected to Gorgon at {url}: {health.get('status', 'ok')}")
            return True
        except Exception as e:
            self._set_error(f"Failed to connect to Gorgon: {e}")
            logger.error(f"Gorgon connection failed: {e}")
            if self._client:
                await self._client.close()
                self._client = None
            return False

    async def disconnect(self) -> bool:
        """Disconnect from Gorgon."""
        if self._client:
            await self._client.close()
            self._client = None
        self._set_disconnected()
        logger.info("Disconnected from Gorgon")
        return True

    async def verify(self) -> bool:
        """Verify connection is still valid."""
        if not self._client:
            return False
        try:
            await self._client.check_health()
            return True
        except Exception:
            self._set_expired()
            return False

    def get_tools(self) -> list[Tool]:
        """Return tools for interacting with Gorgon."""
        return [
            Tool(
                name="gorgon_delegate",
                description=(
                    "Delegate a complex task to Gorgon's multi-agent pipeline. "
                    "Use for code review, refactoring, testing, security audits, "
                    "and other tasks that benefit from autonomous agent execution."
                ),
                parameters={
                    "task": {
                        "type": "string",
                        "description": "The task description to delegate",
                        "required": True,
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority 0-10 (default 5)",
                        "required": False,
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "Wait for completion (default false)",
                        "required": False,
                    },
                },
                handler=self._tool_delegate,
                category="orchestration",
            ),
            Tool(
                name="gorgon_status",
                description="Check Gorgon task queue statistics.",
                parameters={},
                handler=self._tool_stats,
                category="orchestration",
            ),
            Tool(
                name="gorgon_check",
                description="Check the status of a specific Gorgon task.",
                parameters={
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to check",
                        "required": True,
                    },
                },
                handler=self._tool_check,
                category="orchestration",
            ),
            Tool(
                name="gorgon_list",
                description="List recent Gorgon tasks.",
                parameters={
                    "limit": {
                        "type": "integer",
                        "description": "Number of tasks (default 5)",
                        "required": False,
                    },
                },
                handler=self._tool_list,
                category="orchestration",
            ),
            Tool(
                name="gorgon_cancel",
                description="Cancel a pending Gorgon task.",
                parameters={
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to cancel",
                        "required": True,
                    },
                },
                handler=self._tool_cancel,
                category="orchestration",
            ),
        ]

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    async def _tool_delegate(
        self, task: str, priority: int = 5, wait: bool = False, **kwargs: Any
    ) -> ToolResult:
        """Delegate a task to Gorgon."""
        if not self._client:
            return ToolResult("gorgon_delegate", False, None, "Not connected to Gorgon")

        try:
            title = task[:80]
            if wait:
                result = await self._client.submit_and_wait(
                    title=title,
                    description=task,
                    priority=priority,
                )
            else:
                result = await self._client.submit_task(
                    title=title,
                    description=task,
                    priority=priority,
                )
            return ToolResult("gorgon_delegate", True, result)
        except Exception as e:
            return ToolResult("gorgon_delegate", False, None, f"Delegation failed: {e}")

    async def _tool_stats(self, **kwargs: Any) -> ToolResult:
        """Get task queue stats."""
        if not self._client:
            return ToolResult("gorgon_status", False, None, "Not connected to Gorgon")

        try:
            stats = await self._client.get_stats()
            return ToolResult("gorgon_status", True, stats)
        except Exception as e:
            return ToolResult("gorgon_status", False, None, f"Failed to get stats: {e}")

    async def _tool_check(self, task_id: str, **kwargs: Any) -> ToolResult:
        """Check task status."""
        if not self._client:
            return ToolResult("gorgon_check", False, None, "Not connected to Gorgon")

        try:
            task = await self._client.get_task(task_id)
            return ToolResult("gorgon_check", True, task)
        except Exception as e:
            return ToolResult("gorgon_check", False, None, f"Failed to check task: {e}")

    async def _tool_list(self, limit: int = 5, **kwargs: Any) -> ToolResult:
        """List recent tasks."""
        if not self._client:
            return ToolResult("gorgon_list", False, None, "Not connected to Gorgon")

        try:
            tasks = await self._client.list_tasks(limit=limit)
            return ToolResult("gorgon_list", True, tasks)
        except Exception as e:
            return ToolResult("gorgon_list", False, None, f"Failed to list tasks: {e}")

    async def _tool_cancel(self, task_id: str, **kwargs: Any) -> ToolResult:
        """Cancel a pending task."""
        if not self._client:
            return ToolResult("gorgon_cancel", False, None, "Not connected to Gorgon")

        try:
            result = await self._client.cancel_task(task_id)
            return ToolResult("gorgon_cancel", True, result)
        except Exception as e:
            return ToolResult("gorgon_cancel", False, None, f"Failed to cancel task: {e}")
