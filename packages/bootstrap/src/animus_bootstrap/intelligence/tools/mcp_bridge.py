"""MCP tool bridge — discovers and wraps MCP server tools as ToolDefinitions.

Implements the Model Context Protocol (MCP) client side:
- Reads server configs from a JSON file (mcpServers key)
- Starts server processes via stdio transport
- Discovers tools via JSON-RPC (initialize → tools/list)
- Wraps each tool as a ToolDefinition with a handler that calls tools/call
- Manages server process lifecycle
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from animus_bootstrap.intelligence.tools.executor import ToolDefinition

logger = logging.getLogger(__name__)

_JSONRPC_VERSION = "2.0"
_MCP_PROTOCOL_VERSION = "2024-11-05"
_DEFAULT_TIMEOUT = 30.0


class MCPServerConnection:
    """A live connection to an MCP server via stdio."""

    def __init__(self, name: str, process: asyncio.subprocess.Process) -> None:
        self.name = name
        self.process = process
        self._request_id = 0
        self._lock = asyncio.Lock()

    async def send_request(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request and wait for the response."""
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError(f"MCP server '{self.name}' has no stdio pipes")

        async with self._lock:
            self._request_id += 1
            request = {
                "jsonrpc": _JSONRPC_VERSION,
                "id": self._request_id,
                "method": method,
            }
            if params is not None:
                request["params"] = params

            payload = json.dumps(request) + "\n"
            self.process.stdin.write(payload.encode())
            await self.process.stdin.drain()

            # Read response line
            try:
                line = await asyncio.wait_for(
                    self.process.stdout.readline(), timeout=_DEFAULT_TIMEOUT
                )
            except TimeoutError:
                raise TimeoutError(f"MCP server '{self.name}' timed out on {method}") from None

            if not line:
                raise RuntimeError(f"MCP server '{self.name}' closed stdout")

            try:
                response = json.loads(line.decode())
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"MCP server '{self.name}' returned invalid JSON: {exc}"
                ) from exc

        if "error" in response:
            err = response["error"]
            raise RuntimeError(f"MCP server '{self.name}' error: {err.get('message', err)}")

        return response.get("result", {})

    async def close(self) -> None:
        """Terminate the server process."""
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except TimeoutError:
                self.process.kill()
            logger.info("MCP server '%s' terminated", self.name)


class MCPToolBridge:
    """Discovers and wraps MCP server tools as ToolDefinitions."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        self._config_path = Path(config_path) if config_path else None
        self._servers: dict[str, dict] = {}
        self._connections: dict[str, MCPServerConnection] = {}

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

    async def _start_server(self, name: str) -> MCPServerConnection:
        """Start an MCP server process and initialize the connection."""
        config = self._servers[name]
        command = config.get("command", "")
        args = config.get("args", [])
        env = config.get("env")

        if not command:
            raise ValueError(f"MCP server '{name}' has no command configured")

        try:
            process = await asyncio.create_subprocess_exec(
                command,
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"MCP server command not found: {command}") from None

        conn = MCPServerConnection(name, process)

        # Send initialize request
        try:
            await conn.send_request(
                "initialize",
                {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "animus-bootstrap",
                        "version": "0.4.0",
                    },
                },
            )
        except Exception:
            await conn.close()
            raise

        self._connections[name] = conn
        logger.info("MCP server '%s' started (PID %d)", name, process.pid)
        return conn

    async def import_tools(self, server_name: str) -> list[ToolDefinition]:
        """Start an MCP server and import its tools as ToolDefinitions.

        Each tool gets a handler that calls through to the MCP server
        via JSON-RPC tools/call.
        """
        if server_name not in self._servers:
            logger.warning("Unknown MCP server: %s", server_name)
            return []

        try:
            conn = await self._start_server(server_name)
        except Exception as exc:
            logger.warning("Failed to start MCP server '%s': %s", server_name, exc)
            return []

        # List tools
        try:
            result = await conn.send_request("tools/list")
        except Exception as exc:
            logger.warning("Failed to list tools from MCP server '%s': %s", server_name, exc)
            await conn.close()
            self._connections.pop(server_name, None)
            return []

        tools_list = result.get("tools", [])
        definitions: list[ToolDefinition] = []

        for tool_spec in tools_list:
            tool_name = tool_spec.get("name", "")
            if not tool_name:
                continue

            # Prefix with server name to avoid collisions
            full_name = f"mcp_{server_name}_{tool_name}"
            description = tool_spec.get("description", f"MCP tool: {tool_name}")
            input_schema = tool_spec.get(
                "inputSchema",
                {
                    "type": "object",
                    "properties": {},
                },
            )

            # Create a closure that captures the server name and tool name
            def _make_handler(srv: str, tname: str) -> Any:
                async def handler(**kwargs: Any) -> str:
                    return await self.call_tool(srv, tname, kwargs)

                return handler

            definitions.append(
                ToolDefinition(
                    name=full_name,
                    description=description,
                    parameters=input_schema,
                    handler=_make_handler(server_name, tool_name),
                    category=f"mcp:{server_name}",
                )
            )

        logger.info(
            "Imported %d tools from MCP server '%s'",
            len(definitions),
            server_name,
        )
        return definitions

    async def call_tool(self, server: str, tool: str, args: dict) -> str:
        """Execute a tool on an MCP server via JSON-RPC."""
        conn = self._connections.get(server)
        if conn is None:
            return f"MCP server '{server}' is not connected"

        try:
            result = await conn.send_request(
                "tools/call",
                {"name": tool, "arguments": args},
            )
        except Exception as exc:
            return f"MCP tool call failed: {exc}"

        # Extract text content from result
        content = result.get("content", [])
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(texts) if texts else json.dumps(result)
        return json.dumps(result)

    async def close(self) -> None:
        """Shut down all connected MCP servers."""
        for _, conn in list(self._connections.items()):
            await conn.close()
        self._connections.clear()
        logger.info("All MCP server connections closed")
