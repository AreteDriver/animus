"""
Gorgon Integration — Connects Animus to Gorgon's orchestration API.

Gorgon is a headless multi-agent orchestration engine. This integration
allows Animus to delegate complex tasks (code review, refactoring, testing,
security audits) to Gorgon's autonomous agent pipeline.

Supports both the legacy task queue API (/v1/tasks) and the workflow
execution API (/v1/workflows, /v1/executions) with approval gates.

Requires: httpx (optional dependency)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
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
    """Async HTTP client for Gorgon's orchestration API."""

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
            if result.get("status") in (
                "completed",
                "failed",
                "cancelled",
                "awaiting_approval",
            ):
                return result
            await asyncio.sleep(poll_interval)

        return await self.get_task(task_id)

    # ------------------------------------------------------------------
    # Workflow execution API (targets /v1/workflows and /v1/executions)
    # ------------------------------------------------------------------

    async def execute_workflow(
        self,
        workflow_id: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a workflow execution on Gorgon.

        Returns: {execution_id, workflow_id, workflow_name, status, poll_url}
        """
        client = await self._ensure_client()
        payload: dict[str, Any] = {"variables": variables or {}}
        resp = await client.post(f"/v1/workflows/{workflow_id}/execute", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def get_execution(self, execution_id: str) -> dict[str, Any]:
        """Get execution status and details."""
        client = await self._ensure_client()
        resp = await client.get(f"/v1/executions/{execution_id}")
        resp.raise_for_status()
        return resp.json()

    async def list_executions(
        self,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List workflow executions with pagination."""
        client = await self._ensure_client()
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if status:
            params["status"] = status
        resp = await client.get("/v1/executions", params=params)
        resp.raise_for_status()
        return resp.json()

    async def get_approval_status(self, execution_id: str) -> dict[str, Any]:
        """Get pending approval details for an execution.

        Returns: {execution_id, pending_approvals: [...], total_tokens}
        """
        client = await self._ensure_client()
        resp = await client.get(f"/v1/executions/{execution_id}/approval")
        resp.raise_for_status()
        return resp.json()

    async def resume_execution(
        self,
        execution_id: str,
        token: str | None = None,
        approve: bool = True,
        approved_by: str = "animus",
        reason: str = "",
    ) -> dict[str, Any]:
        """Resume a paused or approval-awaiting execution.

        For approval gates, token is required.
        """
        client = await self._ensure_client()
        body: dict[str, Any] | None = None
        if token:
            body = {
                "token": token,
                "approve": approve,
                "approved_by": approved_by,
            }
            if reason:
                body["reason"] = reason
        resp = await client.post(f"/v1/executions/{execution_id}/resume", json=body)
        resp.raise_for_status()
        return resp.json()

    async def cancel_execution(self, execution_id: str) -> dict[str, Any]:
        """Cancel a running or paused execution."""
        client = await self._ensure_client()
        resp = await client.post(f"/v1/executions/{execution_id}/cancel")
        resp.raise_for_status()
        return resp.json()

    async def execute_and_wait(
        self,
        workflow_id: str,
        variables: dict[str, Any] | None = None,
        poll_interval: float = 5.0,
        max_wait: float = 300.0,
        on_approval: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Execute a workflow and poll until completion.

        If the execution enters ``awaiting_approval`` and *on_approval*
        is provided, the callback receives the approval info dict and
        must return ``{"approve": bool, "reason": str}``.

        Without a callback, returns immediately when an approval gate
        is reached so the caller can handle it.
        """
        exec_result = await self.execute_workflow(workflow_id, variables)
        execution_id = exec_result["execution_id"]
        start = time.monotonic()

        while time.monotonic() - start < max_wait:
            execution = await self.get_execution(execution_id)
            status = execution.get("status")

            if status in ("completed", "failed", "cancelled"):
                return execution

            if status == "awaiting_approval":
                if on_approval is None:
                    return execution
                approval_info = await self.get_approval_status(execution_id)
                decision = await on_approval(approval_info)
                for pending in approval_info.get("pending_approvals", []):
                    if pending.get("status") == "pending":
                        await self.resume_execution(
                            execution_id,
                            token=pending["token"],
                            approve=decision.get("approve", True),
                            approved_by="animus",
                            reason=decision.get("reason", ""),
                        )
                # Continue polling after approval

            await asyncio.sleep(poll_interval)

        return await self.get_execution(execution_id)


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
            # --- Workflow execution + approval tools ---
            Tool(
                name="gorgon_execute",
                description=(
                    "Execute a workflow on Gorgon's orchestration engine. "
                    "Returns the execution ID for tracking."
                ),
                parameters={
                    "workflow_id": {
                        "type": "string",
                        "description": "Workflow ID to execute",
                        "required": True,
                    },
                    "variables": {
                        "type": "object",
                        "description": "Input variables (optional)",
                        "required": False,
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "Wait for completion (default false)",
                        "required": False,
                    },
                },
                handler=self._tool_execute,
                category="orchestration",
            ),
            Tool(
                name="gorgon_execution_status",
                description="Get the status of a Gorgon workflow execution.",
                parameters={
                    "execution_id": {
                        "type": "string",
                        "description": "Execution ID",
                        "required": True,
                    },
                },
                handler=self._tool_execution_status,
                category="orchestration",
            ),
            Tool(
                name="gorgon_executions",
                description="List recent Gorgon workflow executions.",
                parameters={
                    "status": {
                        "type": "string",
                        "description": "Filter by status (optional)",
                        "required": False,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Page size (default 10)",
                        "required": False,
                    },
                },
                handler=self._tool_executions,
                category="orchestration",
            ),
            Tool(
                name="gorgon_approvals",
                description="Check pending approval gates for a Gorgon execution.",
                parameters={
                    "execution_id": {
                        "type": "string",
                        "description": "Execution ID",
                        "required": True,
                    },
                },
                handler=self._tool_approvals,
                category="orchestration",
            ),
            Tool(
                name="gorgon_approve",
                description="Approve or reject a pending approval gate.",
                parameters={
                    "execution_id": {
                        "type": "string",
                        "description": "Execution ID",
                        "required": True,
                    },
                    "token": {
                        "type": "string",
                        "description": "Approval token",
                        "required": True,
                    },
                    "approve": {
                        "type": "boolean",
                        "description": "True to approve, False to reject",
                        "required": True,
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for decision (optional)",
                        "required": False,
                    },
                },
                handler=self._tool_approve,
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

    # ------------------------------------------------------------------
    # Execution + approval tool handlers
    # ------------------------------------------------------------------

    async def _tool_execute(
        self,
        workflow_id: str,
        variables: dict[str, Any] | None = None,
        wait: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a workflow on Gorgon."""
        if not self._client:
            return ToolResult("gorgon_execute", False, None, "Not connected to Gorgon")

        try:
            if wait:
                result = await self._client.execute_and_wait(
                    workflow_id, variables, poll_interval=5.0
                )
            else:
                result = await self._client.execute_workflow(workflow_id, variables)
            return ToolResult("gorgon_execute", True, result)
        except Exception as e:
            return ToolResult("gorgon_execute", False, None, f"Execution failed: {e}")

    async def _tool_execution_status(self, execution_id: str, **kwargs: Any) -> ToolResult:
        """Get execution status."""
        if not self._client:
            return ToolResult("gorgon_execution_status", False, None, "Not connected to Gorgon")

        try:
            result = await self._client.get_execution(execution_id)
            return ToolResult("gorgon_execution_status", True, result)
        except Exception as e:
            return ToolResult("gorgon_execution_status", False, None, f"Failed: {e}")

    async def _tool_executions(
        self, status: str | None = None, limit: int = 10, **kwargs: Any
    ) -> ToolResult:
        """List recent executions."""
        if not self._client:
            return ToolResult("gorgon_executions", False, None, "Not connected to Gorgon")

        try:
            result = await self._client.list_executions(page_size=limit, status=status)
            return ToolResult("gorgon_executions", True, result)
        except Exception as e:
            return ToolResult("gorgon_executions", False, None, f"Failed: {e}")

    async def _tool_approvals(self, execution_id: str, **kwargs: Any) -> ToolResult:
        """Get pending approval gates."""
        if not self._client:
            return ToolResult("gorgon_approvals", False, None, "Not connected to Gorgon")

        try:
            result = await self._client.get_approval_status(execution_id)
            return ToolResult("gorgon_approvals", True, result)
        except Exception as e:
            return ToolResult("gorgon_approvals", False, None, f"Failed: {e}")

    async def _tool_approve(
        self,
        execution_id: str,
        token: str,
        approve: bool = True,
        reason: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Approve or reject an approval gate."""
        if not self._client:
            return ToolResult("gorgon_approve", False, None, "Not connected to Gorgon")

        try:
            result = await self._client.resume_execution(
                execution_id,
                token=token,
                approve=approve,
                approved_by="animus",
                reason=reason,
            )
            action = "Approved" if approve else "Rejected"
            return ToolResult("gorgon_approve", True, result, f"{action} gate")
        except Exception as e:
            return ToolResult("gorgon_approve", False, None, f"Failed: {e}")
