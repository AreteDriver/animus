"""Forge integration tools for the Animus CLI.

Provides Tool objects that call the Forge HTTP API on localhost:8000.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from animus.tools import Tool, ToolResult

FORGE_BASE = "http://localhost:8000/v1"


def _forge_request(path: str, method: str = "GET", body: bytes | None = None, timeout: int = 10, token: str = "") -> dict[str, Any]:
    """Make a request to Forge and return parsed JSON."""
    effective_token = token or os.environ.get("FORGE_TOKEN", "")
    req = urllib.request.Request(f"{FORGE_BASE}{path}", data=body, method=method)
    req.add_header("Accept", "application/json")
    if body:
        req.add_header("Content-Type", "application/json")
    if effective_token:
        req.add_header("Authorization", f"Bearer {effective_token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _tool_forge_health(params: dict) -> ToolResult:
    try:
        req = urllib.request.Request("http://localhost:8000/health", method="GET")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return ToolResult(
            tool_name="forge_health",
            success=True,
            output=f"Forge status: {data.get('status', 'unknown')} at {data.get('timestamp', 'N/A')}",
        )
    except Exception as e:
        return ToolResult(
            tool_name="forge_health",
            success=False,
            output=None,
            error=f"Forge unreachable on localhost:8000 — {e}",
        )


def _tool_forge_list_workflows(params: dict) -> ToolResult:
    token = params.get("token", "")
    try:
        data = _forge_request("/workflows", method="GET", token=token)
        workflows = data if isinstance(data, list) else data.get("workflows", [])
        lines = [f"Registered workflows: {len(workflows)}"]
        for wf in workflows[:20]:
            lines.append(f"  - {wf.get('name', 'unnamed')} (id: {wf.get('id', 'N/A')})")
        return ToolResult(
            tool_name="forge_list_workflows",
            success=True,
            output="\n".join(lines),
        )
    except Exception as e:
        return ToolResult(
            tool_name="forge_list_workflows",
            success=False,
            output=None,
            error=str(e),
        )


def _tool_forge_run_workflow(params: dict) -> ToolResult:
    workflow_id = params.get("workflow_id", "")
    token = params.get("token", "")
    if not workflow_id:
        return ToolResult(
            tool_name="forge_run_workflow",
            success=False,
            output=None,
            error="Missing required parameter: workflow_id",
        )
    try:
        body = json.dumps({"workflow_id": workflow_id}).encode()
        data = _forge_request("/workflows/execute", method="POST", body=body, timeout=300, token=token)
        status = data.get("status", "unknown")
        return ToolResult(
            tool_name="forge_run_workflow",
            success=status in ("success", "complete"),
            output=json.dumps(data, indent=2),
        )
    except Exception as e:
        return ToolResult(
            tool_name="forge_run_workflow",
            success=False,
            output=None,
            error=str(e),
        )


def _tool_forge_auth_login(params: dict) -> ToolResult:
    user_id = params.get("user_id", "demo")
    pw = params.get("password", "demo")
    try:
        body = json.dumps({"user_id": user_id, "password": pw}).encode()
        data = _forge_request("/auth/login", method="POST", body=body)
        token = data.get("access_token", "")
        if token:
            # Store token in env var so subsequent tools pick it up automatically
            os.environ["FORGE_TOKEN"] = token
        return ToolResult(
            tool_name="forge_auth_login",
            success=bool(token),
            output=f"Token: {token}\n(exported to FORGE_TOKEN env var for this session)",
        )
    except Exception as e:
        return ToolResult(
            tool_name="forge_auth_login",
            success=False,
            output=None,
            error=str(e),
        )


FORGE_TOOLS: list[Tool] = [
    Tool(
        name="forge_health",
        description="Check if the Forge workflow orchestrator is running and healthy",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_tool_forge_health,
        category="forge",
    ),
    Tool(
        name="forge_list_workflows",
        description="List all workflows registered in Forge",
        parameters={
            "type": "object",
            "properties": {
                "token": {
                    "type": "string",
                    "description": "Bearer token for Forge authentication (optional)",
                }
            },
            "required": [],
        },
        handler=_tool_forge_list_workflows,
        category="forge",
    ),
    Tool(
        name="forge_run_workflow",
        description="Execute a Forge workflow by its ID",
        parameters={
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The workflow ID to execute (e.g., architect-ADL-20260706-21b516)",
                },
                "token": {
                    "type": "string",
                    "description": "Bearer token for Forge authentication (optional)",
                }
            },
            "required": ["workflow_id"],
        },
        handler=_tool_forge_run_workflow,
        category="forge",
    ),
    Tool(
        name="forge_auth_login",
        description="Log in to Forge with demo credentials and get an access token",
        parameters={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "Username (default: demo)"},
                "password": {"type": "string", "description": "Password (default: demo)"},
            },
            "required": [],
        },
        handler=_tool_forge_auth_login,
        category="forge",
    ),
]
