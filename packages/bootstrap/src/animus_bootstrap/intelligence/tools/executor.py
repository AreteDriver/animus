"""Core tool executor — registration, execution, and history tracking."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from animus_bootstrap.intelligence.tools.permissions import (
    PermissionLevel,
    ToolApprovalRequired,
    ToolPermissionManager,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """A callable tool the LLM can invoke."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[..., Coroutine[Any, Any, str]]
    category: str = "general"
    permission: str = "auto"  # "auto" | "approve" | "deny"


@dataclass
class ToolResult:
    """Result of executing a tool."""

    id: str
    tool_name: str
    success: bool
    output: str
    duration_ms: float
    timestamp: datetime
    arguments: dict[str, Any] = field(default_factory=dict)


class ToolExecutor:
    """Manages tool registration and execution."""

    def __init__(
        self,
        max_calls_per_turn: int = 5,
        timeout_seconds: float = 30.0,
        permission_manager: ToolPermissionManager | None = None,
    ) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._history: list[ToolResult] = []
        self._max_calls = max_calls_per_turn
        self._timeout = timeout_seconds
        self._permissions = permission_manager or ToolPermissionManager()

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool. Raises ValueError if name already taken."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        # Sync permission from tool definition
        if tool.permission != "auto":
            self._permissions.set_permission(tool.name, PermissionLevel(tool.permission))

    def unregister(self, name: str) -> None:
        """Remove a tool by name. No-op if not found."""
        self._tools.pop(name, None)
        self._permissions.remove_override(name)

    def list_tools(self) -> list[ToolDefinition]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a tool by name, or None if not found."""
        return self._tools.get(name)

    def get_schemas(self) -> list[dict]:
        """Return JSON schemas for all tools in Anthropic tool_use format."""
        schemas: list[dict] = []
        for tool in self._tools.values():
            schemas.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
            )
        return schemas

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a single tool by name with arguments."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                id=str(uuid.uuid4()),
                tool_name=name,
                success=False,
                output=f"Unknown tool: {name}",
                duration_ms=0.0,
                timestamp=datetime.now(UTC),
                arguments=arguments,
            )

        # Check permission
        try:
            allowed = self._permissions.check(name)
        except ToolApprovalRequired:
            return ToolResult(
                id=str(uuid.uuid4()),
                tool_name=name,
                success=False,
                output=f"Tool '{name}' requires user approval",
                duration_ms=0.0,
                timestamp=datetime.now(UTC),
                arguments=arguments,
            )

        if not allowed:
            return ToolResult(
                id=str(uuid.uuid4()),
                tool_name=name,
                success=False,
                output=f"Tool '{name}' is denied by permission policy",
                duration_ms=0.0,
                timestamp=datetime.now(UTC),
                arguments=arguments,
            )

        # Execute with timeout
        start = time.monotonic()
        try:
            output = await asyncio.wait_for(tool.handler(**arguments), timeout=self._timeout)
            duration_ms = (time.monotonic() - start) * 1000
            result = ToolResult(
                id=str(uuid.uuid4()),
                tool_name=name,
                success=True,
                output=output,
                duration_ms=duration_ms,
                timestamp=datetime.now(UTC),
                arguments=arguments,
            )
        except TimeoutError:
            duration_ms = (time.monotonic() - start) * 1000
            result = ToolResult(
                id=str(uuid.uuid4()),
                tool_name=name,
                success=False,
                output=f"Tool '{name}' timed out after {self._timeout}s",
                duration_ms=duration_ms,
                timestamp=datetime.now(UTC),
                arguments=arguments,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.exception("Tool '%s' raised an exception", name)
            result = ToolResult(
                id=str(uuid.uuid4()),
                tool_name=name,
                success=False,
                output=f"Tool '{name}' failed: {exc}",
                duration_ms=duration_ms,
                timestamp=datetime.now(UTC),
                arguments=arguments,
            )

        self._history.append(result)
        return result

    async def execute_batch(self, tool_calls: list[dict]) -> list[ToolResult]:
        """Execute multiple tool calls sequentially, respecting max_calls_per_turn."""
        results: list[ToolResult] = []
        for call in tool_calls[: self._max_calls]:
            name = call.get("name", "")
            arguments = call.get("arguments", {})
            result = await self.execute(name, arguments)
            results.append(result)
        return results

    def get_history(self, limit: int = 50) -> list[ToolResult]:
        """Return recent execution history (most recent last)."""
        return self._history[-limit:]

    def clear_history(self) -> None:
        """Clear execution history."""
        self._history.clear()
