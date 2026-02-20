"""MCP protocol client for tool execution.

Wraps the `mcp` Python SDK to call tools on registered MCP servers
via stdio or SSE transport.  Gracefully degrades when the SDK is
not installed (optional dependency).
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from typing import Any

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client

    HAS_MCP_SDK = True
except ImportError:
    HAS_MCP_SDK = False


class MCPClientError(Exception):
    """Error raised when an MCP tool call fails."""


# ---------------------------------------------------------------------------
# Async transport helpers
# ---------------------------------------------------------------------------


async def _call_tool_stdio(
    command: str,
    args: list[str],
    env: dict[str, str] | None,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Call a tool on an MCP server via stdio transport.

    Args:
        command: Executable command for the server
        args: Command-line arguments
        env: Optional environment variables
        tool_name: Name of the tool to invoke
        arguments: Tool-specific arguments

    Returns:
        Normalized result dict ``{"content": str, "is_error": bool}``
    """
    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            content = _extract_content(result)
            return {"content": content, "is_error": bool(result.isError)}


async def _call_tool_sse(
    url: str,
    headers: dict[str, str] | None,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Call a tool on an MCP server via SSE transport.

    Args:
        url: SSE endpoint URL
        headers: Optional HTTP headers (auth, etc.)
        tool_name: Name of the tool to invoke
        arguments: Tool-specific arguments

    Returns:
        Normalized result dict ``{"content": str, "is_error": bool}``
    """
    sse_kwargs: dict[str, Any] = {"url": url}
    if headers:
        sse_kwargs["headers"] = headers
    async with sse_client(**sse_kwargs) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            content = _extract_content(result)
            return {"content": content, "is_error": bool(result.isError)}


def _extract_content(result: Any) -> str:
    """Extract text content from an MCP CallToolResult."""
    if not result.content:
        return ""
    parts: list[str] = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
        else:
            parts.append(str(block))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public sync entry-point
# ---------------------------------------------------------------------------


def call_mcp_tool(
    server_type: str,
    server_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    headers: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Call a tool on an MCP server (sync wrapper).

    Dispatches to the correct transport (stdio / sse) based on *server_type*.

    Args:
        server_type: ``"stdio"`` or ``"sse"``
        server_url: For stdio: ``"command arg1 arg2"``; for SSE: the HTTP URL
        tool_name: Name of the tool to invoke
        arguments: Tool-specific arguments
        headers: Optional HTTP headers (SSE only)
        env: Optional environment variables (stdio only)

    Returns:
        ``{"content": str, "is_error": bool}``

    Raises:
        MCPClientError: On SDK missing, unsupported type, or connection failure
    """
    if not HAS_MCP_SDK:
        raise MCPClientError("MCP SDK not installed. Install with: pip install 'gorgon[mcp]'")

    try:
        if server_type == "stdio":
            parts = shlex.split(server_url)
            command = parts[0]
            args = parts[1:] if len(parts) > 1 else []
            return asyncio.run(_call_tool_stdio(command, args, env, tool_name, arguments))
        elif server_type == "sse":
            return asyncio.run(_call_tool_sse(server_url, headers, tool_name, arguments))
        else:
            raise MCPClientError(f"Unsupported MCP server type: {server_type}")
    except MCPClientError:
        raise
    except Exception as exc:
        raise MCPClientError(f"MCP tool call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Tool / resource discovery
# ---------------------------------------------------------------------------


async def _discover_stdio(
    command: str,
    args: list[str],
    env: dict[str, str] | None,
) -> dict[str, Any]:
    """Discover tools and resources on a stdio MCP server."""
    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            resources_result = await session.list_resources()
            return _normalize_discovery(tools_result, resources_result)


async def _discover_sse(
    url: str,
    headers: dict[str, str] | None,
) -> dict[str, Any]:
    """Discover tools and resources on an SSE MCP server."""
    sse_kwargs: dict[str, Any] = {"url": url}
    if headers:
        sse_kwargs["headers"] = headers
    async with sse_client(**sse_kwargs) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            resources_result = await session.list_resources()
            return _normalize_discovery(tools_result, resources_result)


def _normalize_discovery(tools_result: Any, resources_result: Any) -> dict[str, Any]:
    """Normalize MCP list_tools / list_resources results to plain dicts."""
    tools = []
    for tool in getattr(tools_result, "tools", []) or []:
        tools.append(
            {
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "inputSchema": getattr(tool, "inputSchema", {}) or {},
            }
        )

    resources = []
    for res in getattr(resources_result, "resources", []) or []:
        resources.append(
            {
                "uri": str(getattr(res, "uri", "")),
                "name": getattr(res, "name", "") or "",
                "mimeType": getattr(res, "mimeType", None),
                "description": getattr(res, "description", None),
            }
        )

    return {"tools": tools, "resources": resources}


def discover_tools(
    server_type: str,
    server_url: str,
    headers: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Discover tools and resources on an MCP server (sync wrapper).

    Args:
        server_type: ``"stdio"`` or ``"sse"``
        server_url: For stdio: ``"command arg1 arg2"``; for SSE: the HTTP URL
        headers: Optional HTTP headers (SSE only)
        env: Optional environment variables (stdio only)

    Returns:
        ``{"tools": [...], "resources": [...]}``

    Raises:
        MCPClientError: On SDK missing, unsupported type, or connection failure
    """
    if not HAS_MCP_SDK:
        raise MCPClientError("MCP SDK not installed. Install with: pip install 'gorgon[mcp]'")

    try:
        if server_type == "stdio":
            parts = shlex.split(server_url)
            command = parts[0]
            args = parts[1:] if len(parts) > 1 else []
            return asyncio.run(_discover_stdio(command, args, env))
        elif server_type == "sse":
            return asyncio.run(_discover_sse(server_url, headers))
        else:
            raise MCPClientError(f"Unsupported MCP server type: {server_type}")
    except MCPClientError:
        raise
    except Exception as exc:
        raise MCPClientError(f"MCP discovery failed: {exc}") from exc
