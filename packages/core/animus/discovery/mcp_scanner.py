"""MCPScanner: discovers tools from MCP (Model Context Protocol) servers.

Supports two modes:
- SSE transport: connects to running MCP servers via HTTP
- stdio transport: spawns MCP server processes locally

Discovered tools are converted to Animus Tool schemas with JSON Schema parameters.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from animus.logging import get_logger

logger = get_logger("discovery.mcp_scanner")


@dataclass
class MCPToolSpec:
    """Raw tool specification from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str
    server_url: str | None = None
    transport: str = "sse"  # sse, stdio, websocket

    def to_animus_schema(self) -> dict:
        """Convert MCP input schema to Animus-compatible JSON Schema."""
        return {
            "type": "object",
            "properties": self.input_schema.get("properties", {}),
            "required": self.input_schema.get("required", []),
        }


class MCPScanner:
    """Scans MCP servers and extracts available tools.

    Usage:
        scanner = MCPScanner()
        specs = scanner.scan_server("http://localhost:3000/sse")
        for spec in specs:
            registry.register(Tool(name=spec.name, ...))
    """

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self._discovered_servers: dict[str, list[MCPToolSpec]] = {}

    def scan_server(self, url: str, server_name: str | None = None) -> list[MCPToolSpec]:
        """Scan a single MCP server via SSE transport.

        Args:
            url: MCP server SSE endpoint (e.g., http://localhost:3000/sse)
            server_name: Human-readable name for the server.

        Returns:
            List of discovered tool specs.
        """
        specs: list[MCPToolSpec] = []
        name = server_name or url.split("//")[-1].split("/")[0]

        try:
            import requests
        except ImportError:
            logger.warning("requests not installed; falling back to stdio discovery only")
            return specs

        try:
            # MCP servers expose a /tools endpoint for listing available tools
            tools_url = url.rstrip("/") + "/tools"
            resp = requests.get(tools_url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            for tool_data in data.get("tools", []):
                spec = MCPToolSpec(
                    name=tool_data.get("name", "unknown"),
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                    server_name=name,
                    server_url=url,
                    transport="sse",
                )
                specs.append(spec)
                logger.debug(f"Discovered MCP tool: {spec.name} from {name}")

        except Exception as e:
            logger.warning(f"Failed to scan MCP server {url}: {e}")

        self._discovered_servers[name] = specs
        logger.info(f"MCP scan complete: {len(specs)} tools from {name}")
        return specs

    def scan_stdio_server(self, command: list[str], server_name: str) -> list[MCPToolSpec]:
        """Discover tools from a stdio-based MCP server.

        Spawns the server process, sends an initialize request, then lists tools.

        Args:
            command: Command to spawn the MCP server (e.g., ["python", "-m", "mcp_server"])
            server_name: Human-readable name for the server.

        Returns:
            List of discovered tool specs.
        """
        specs: list[MCPToolSpec] = []

        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Send initialize request
            init_req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "animus-discovery", "version": "1.0.0"},
                },
            }
            proc.stdin.write(json.dumps(init_req) + "\n")
            proc.stdin.flush()

            # Read initialize response
            line = proc.stdout.readline()
            if not line:
                logger.warning(f"MCP server {server_name} returned no response to init")
                proc.terminate()
                return specs

            # Send tools/list request
            list_req = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
            }
            proc.stdin.write(json.dumps(list_req) + "\n")
            proc.stdin.flush()

            # Read tools response with timeout
            start = time.time()
            while time.time() - start < self.timeout:
                line = proc.stdout.readline()
                if not line:
                    break
                try:
                    resp = json.loads(line)
                    if resp.get("id") == 2 and "result" in resp:
                        result = resp["result"]
                        for tool_data in result.get("tools", []):
                            spec = MCPToolSpec(
                                name=tool_data.get("name", "unknown"),
                                description=tool_data.get("description", ""),
                                input_schema=tool_data.get("inputSchema", {}),
                                server_name=server_name,
                                transport="stdio",
                            )
                            specs.append(spec)
                        break
                except json.JSONDecodeError:
                    continue

            proc.terminate()

        except Exception as e:
            logger.warning(f"Failed to scan stdio MCP server {server_name}: {e}")

        self._discovered_servers[server_name] = specs
        logger.info(f"MCP stdio scan complete: {len(specs)} tools from {server_name}")
        return specs

    def scan_local_servers(self, ports: list[int] | None = None) -> list[MCPToolSpec]:
        """Scan localhost for MCP servers on common ports.

        Args:
            ports: List of ports to scan. Defaults to common MCP ports.

        Returns:
            Combined list of discovered tool specs.
        """
        if ports is None:
            ports = [3000, 8080, 8081, 9000, 9001]

        all_specs: list[MCPToolSpec] = []
        for port in ports:
            url = f"http://localhost:{port}/sse"
            specs = self.scan_server(url, server_name=f"localhost:{port}")
            all_specs.extend(specs)

        return all_specs

    def get_all_discovered(self) -> dict[str, list[MCPToolSpec]]:
        """Get all discovered tools grouped by server."""
        return dict(self._discovered_servers)
