"""Permission management for tool execution."""

from __future__ import annotations

from enum import StrEnum


class PermissionLevel(StrEnum):
    """Permission levels controlling tool execution behaviour."""

    AUTO = "auto"  # Execute without asking
    APPROVE = "approve"  # Ask user before executing
    DENY = "deny"  # Never execute


class ToolApprovalRequired(Exception):
    """Raised when a tool requires user approval before execution."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' requires user approval before execution")


class ToolPermissionManager:
    """Manages per-tool permission levels."""

    def __init__(self, default: PermissionLevel = PermissionLevel.AUTO) -> None:
        self._default = default
        self._overrides: dict[str, PermissionLevel] = {}

    def set_permission(self, tool_name: str, level: PermissionLevel) -> None:
        """Set an explicit permission level for a specific tool."""
        self._overrides[tool_name] = level

    def get_permission(self, tool_name: str) -> PermissionLevel:
        """Get the effective permission level for a tool."""
        return self._overrides.get(tool_name, self._default)

    def remove_override(self, tool_name: str) -> None:
        """Remove a per-tool override, reverting to the default level."""
        self._overrides.pop(tool_name, None)

    def list_overrides(self) -> dict[str, PermissionLevel]:
        """Return all per-tool permission overrides."""
        return dict(self._overrides)

    def check(self, tool_name: str) -> bool:
        """Check whether a tool can auto-execute.

        Returns True if the tool can execute automatically (AUTO).
        Returns False if the tool is denied (DENY).
        Raises ToolApprovalRequired if the tool needs user approval (APPROVE).
        """
        level = self.get_permission(tool_name)
        if level == PermissionLevel.APPROVE:
            raise ToolApprovalRequired(tool_name)
        return level == PermissionLevel.AUTO
