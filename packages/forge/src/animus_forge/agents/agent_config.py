"""Per-agent configuration for isolated sub-agent execution.

Each agent can have its own workspace, tool access, model routing,
and resource limits. This enables true multi-agent isolation where
agents operate independently without sharing state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an isolated sub-agent.

    Controls workspace isolation, tool access, model routing, and
    resource limits per agent. When not specified, agents inherit
    the supervisor's shared configuration.

    Attributes:
        role: Agent role name (e.g. "builder", "tester").
        workspace: Optional isolated workspace directory. If set,
            the agent's tool registry is rooted here instead of
            the global project root.
        allowed_tools: Tool allow list. If set, only these tools
            are available. None = all tools available.
        denied_tools: Tool deny list. Deny wins over allow.
        model: Optional model override for this agent.
        provider_type: Optional provider override ("anthropic", "openai", "ollama").
        max_tool_iterations: Max tool loop iterations for this agent.
        timeout_seconds: Max execution time before the agent is cancelled.
        max_output_chars: Max characters in agent output before truncation.
        enable_shell: Whether this agent can run shell commands.
        budget_pool: Optional per-agent token budget (deducted from global).
        metadata: Arbitrary key-value metadata for the agent.
    """

    role: str
    workspace: Path | None = None
    allowed_tools: list[str] | None = None
    denied_tools: list[str] | None = None
    model: str | None = None
    provider_type: str | None = None
    max_tool_iterations: int = 8
    timeout_seconds: float = 300.0
    max_output_chars: int = 50000
    enable_shell: bool = False
    budget_pool: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_effective_tools(self, available_tools: list[str]) -> list[str]:
        """Compute effective tool list after allow/deny filtering.

        Args:
            available_tools: Full list of tools from the registry.

        Returns:
            Filtered list of tool names this agent can use.
        """
        tools = available_tools
        if self.allowed_tools is not None:
            tools = [t for t in tools if t in self.allowed_tools]
        if self.denied_tools:
            tools = [t for t in tools if t not in self.denied_tools]
        return tools


# Default configs for built-in roles
DEFAULT_AGENT_CONFIGS: dict[str, AgentConfig] = {
    "builder": AgentConfig(
        role="builder",
        enable_shell=True,
        allowed_tools=None,  # Full access
        max_tool_iterations=12,
        timeout_seconds=600.0,
    ),
    "tester": AgentConfig(
        role="tester",
        enable_shell=True,
        allowed_tools=[
            "read_file",
            "list_files",
            "search_code",
            "run_command",
            "get_project_structure",
        ],
        max_tool_iterations=10,
        timeout_seconds=300.0,
    ),
    "reviewer": AgentConfig(
        role="reviewer",
        enable_shell=False,
        allowed_tools=["read_file", "list_files", "search_code", "get_project_structure"],
        denied_tools=["write_file", "edit_file", "run_command"],
        max_tool_iterations=8,
        timeout_seconds=180.0,
    ),
    "analyst": AgentConfig(
        role="analyst",
        enable_shell=False,
        allowed_tools=["read_file", "list_files", "search_code", "get_project_structure"],
        denied_tools=["write_file", "edit_file", "run_command"],
        max_tool_iterations=8,
        timeout_seconds=180.0,
    ),
    "planner": AgentConfig(
        role="planner",
        allowed_tools=[],  # Text-only
        timeout_seconds=120.0,
    ),
    "architect": AgentConfig(
        role="architect",
        allowed_tools=[],  # Text-only
        timeout_seconds=120.0,
    ),
    "documenter": AgentConfig(
        role="documenter",
        allowed_tools=[],  # Text-only
        timeout_seconds=120.0,
    ),
}


def get_agent_config(role: str, overrides: dict[str, AgentConfig] | None = None) -> AgentConfig:
    """Get agent config for a role, with optional overrides.

    Args:
        role: Agent role name.
        overrides: Optional dict of role -> AgentConfig overrides.

    Returns:
        AgentConfig for the role.
    """
    if overrides and role in overrides:
        return overrides[role]
    return DEFAULT_AGENT_CONFIGS.get(role, AgentConfig(role=role))
