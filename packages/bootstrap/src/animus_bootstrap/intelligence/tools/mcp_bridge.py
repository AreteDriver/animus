"""MCP tool bridge — discovers and wraps MCP server tools as ToolDefinitions."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from animus_bootstrap.intelligence.tools.executor import ToolDefinition

logger = logging.getLogger(__name__)


class MCPToolBridge:
    """Discovers and wraps MCP server tools as ToolDefinitions."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        self._config_path = Path(config_path) if config_path else None
        self._servers: dict[str, dict] = {}

    async def discover_servers(self) -> list[str]:
        """Scan MCP config JSON for available servers.

        Reads self._config_path (JSON file with mcpServers key)
        and returns a list of server names.
        """
        if self._config_path is None or not self._config_path.exists():
            return []

        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read MCP config %s: %s", self._config_path, exc)
            return []

        servers = data.get("mcpServers", {})
        if not isinstance(servers, dict):
            return []

        self._servers = servers
        names = list(servers.keys())
        logger.info("Discovered %d MCP servers: %s", len(names), names)
        return names

    async def import_tools(self, server_name: str) -> list[ToolDefinition]:
        """Import tools from an MCP server config.

        Stub — would start MCP server process and list tools.
        For now returns empty list with a log message.
        """
        if server_name not in self._servers:
            logger.warning("Unknown MCP server: %s", server_name)
            return []

        logger.info(
            "MCP tool import for '%s' is not yet implemented — returning empty list",
            server_name,
        )
        return []

    async def call_tool(self, server: str, tool: str, args: dict) -> str:
        """Execute a tool on an MCP server.

        Stub — would communicate via stdio/SSE.
        """
        raise NotImplementedError("MCP tool execution not yet implemented")
