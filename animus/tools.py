"""
Animus Tool Framework

Provides tool definitions, registry, and built-in tools for agentic capabilities.
"""

import asyncio
import fnmatch
import glob as glob_module
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from animus.logging import get_logger

logger = get_logger("tools")

# Security config - initialized by create_tool_registry()
_security_config = None


def _set_security_config(config) -> None:
    """Set the security configuration for tools."""
    global _security_config
    _security_config = config


def _validate_path(path: str) -> tuple[bool, str | None]:
    """
    Validate a file path against security rules.

    Returns:
        (is_valid, error_message)
    """
    if _security_config is None:
        return True, None  # No config = no restrictions (dev mode)

    resolved = Path(path).expanduser().resolve()

    # Check blocked paths
    for blocked in _security_config.blocked_paths:
        blocked_resolved = Path(blocked).expanduser()
        # Handle glob patterns in blocked paths
        if "*" in blocked:
            if fnmatch.fnmatch(str(resolved), str(blocked_resolved)):
                return False, f"Access denied: path matches blocked pattern '{blocked}'"
        elif resolved == blocked_resolved or blocked_resolved in resolved.parents:
            return False, f"Access denied: path is blocked"

    # Check if within allowed paths
    in_allowed = False
    for allowed in _security_config.allowed_paths:
        allowed_resolved = Path(allowed).expanduser().resolve()
        if resolved == allowed_resolved or allowed_resolved in resolved.parents:
            in_allowed = True
            break

    if not in_allowed:
        return False, f"Access denied: path not in allowed directories"

    return True, None


def _validate_command(command: str) -> tuple[bool, str | None]:
    """
    Validate a shell command against security rules.

    Returns:
        (is_valid, error_message)
    """
    if _security_config is None:
        return True, None

    if not _security_config.command_enabled:
        return False, "Command execution is disabled"

    # Check command blocklist
    for pattern in _security_config.command_blocklist:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"Command blocked by security policy"

    return True, None


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_name: str
    success: bool
    output: Any
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }

    def to_context(self) -> str:
        """Format for injection into model context."""
        if self.success:
            return f"[Tool: {self.tool_name}]\n{self.output}"
        else:
            return f"[Tool: {self.tool_name} - ERROR]\n{self.error}"


@dataclass
class Tool:
    """Definition of an executable tool."""

    name: str
    description: str
    parameters: dict  # JSON Schema for parameters
    handler: Callable[[dict], ToolResult]
    requires_approval: bool = False
    category: str = "general"

    def get_schema(self) -> dict:
        """Get JSON Schema representation for model context."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_approval": self.requires_approval,
        }


class ToolRegistry:
    """
    Registry for managing and executing tools.

    Provides tool registration, lookup, and execution with error handling.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        logger.debug("ToolRegistry initialized")

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> bool:
        """Unregister a tool. Returns True if removed."""
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_schema(self) -> list[dict]:
        """Get JSON Schema for all tools (for model context)."""
        return [tool.get_schema() for tool in self._tools.values()]

    def get_schema_text(self) -> str:
        """Get formatted text description of all tools."""
        lines = ["Available tools:"]
        for tool in self._tools.values():
            lines.append(f"\n- {tool.name}: {tool.description}")
            if tool.parameters.get("properties"):
                lines.append("  Parameters:")
                for param, spec in tool.parameters["properties"].items():
                    required = param in tool.parameters.get("required", [])
                    req_marker = " (required)" if required else ""
                    lines.append(
                        f"    - {param}: {spec.get('description', spec.get('type', 'any'))}{req_marker}"
                    )
        return "\n".join(lines)

    def execute(self, name: str, params: dict) -> ToolResult:
        """
        Execute a tool by name with given parameters.

        Args:
            name: Tool name
            params: Parameters to pass to the tool

        Returns:
            ToolResult with success/failure and output
        """
        tool = self.get(name)
        if not tool:
            logger.warning(f"Tool not found: {name}")
            return ToolResult(
                tool_name=name,
                success=False,
                output=None,
                error=f"Tool '{name}' not found",
            )

        try:
            logger.debug(f"Executing tool: {name} with params: {params}")
            result = tool.handler(params)
            logger.debug(f"Tool {name} completed: success={result.success}")
            return result
        except Exception as e:
            logger.error(f"Tool {name} failed with exception: {e}")
            return ToolResult(
                tool_name=name,
                success=False,
                output=None,
                error=str(e),
            )

    async def execute_async(self, name: str, params: dict) -> ToolResult:
        """Async wrapper for tool execution."""
        return await asyncio.to_thread(self.execute, name, params)


# =============================================================================
# Built-in Tools
# =============================================================================


def _tool_get_datetime(params: dict) -> ToolResult:
    """Get current date and time."""
    format_str = params.get("format", "%Y-%m-%d %H:%M:%S")
    try:
        now = datetime.now()
        formatted = now.strftime(format_str)
        return ToolResult(
            tool_name="get_datetime",
            success=True,
            output=formatted,
        )
    except Exception as e:
        return ToolResult(
            tool_name="get_datetime",
            success=False,
            output=None,
            error=str(e),
        )


def _tool_read_file(params: dict) -> ToolResult:
    """Read contents of a local file."""
    path = params.get("path")
    if not path:
        return ToolResult(
            tool_name="read_file",
            success=False,
            output=None,
            error="Missing required parameter: path",
        )

    # Security validation
    is_valid, error = _validate_path(path)
    if not is_valid:
        logger.warning(f"Path validation failed for '{path}': {error}")
        return ToolResult(
            tool_name="read_file",
            success=False,
            output=None,
            error=error,
        )

    try:
        file_path = Path(path).expanduser()
        if not file_path.exists():
            return ToolResult(
                tool_name="read_file",
                success=False,
                output=None,
                error=f"File not found: {path}",
            )

        if not file_path.is_file():
            return ToolResult(
                tool_name="read_file",
                success=False,
                output=None,
                error=f"Not a file: {path}",
            )

        # Limit file size to prevent memory issues
        max_size = params.get("max_size", 100_000)  # 100KB default
        if file_path.stat().st_size > max_size:
            return ToolResult(
                tool_name="read_file",
                success=False,
                output=None,
                error=f"File too large (>{max_size} bytes)",
            )

        content = file_path.read_text()
        return ToolResult(
            tool_name="read_file",
            success=True,
            output=content,
        )
    except Exception as e:
        return ToolResult(
            tool_name="read_file",
            success=False,
            output=None,
            error=str(e),
        )


def _tool_list_files(params: dict) -> ToolResult:
    """List files matching a pattern."""
    pattern = params.get("pattern", "*")
    directory = params.get("directory", ".")

    # Security validation
    is_valid, error = _validate_path(directory)
    if not is_valid:
        logger.warning(f"Path validation failed for '{directory}': {error}")
        return ToolResult(
            tool_name="list_files",
            success=False,
            output=None,
            error=error,
        )

    try:
        base_path = Path(directory).expanduser()
        if not base_path.exists():
            return ToolResult(
                tool_name="list_files",
                success=False,
                output=None,
                error=f"Directory not found: {directory}",
            )

        full_pattern = str(base_path / pattern)
        matches = glob_module.glob(full_pattern, recursive=True)

        # Limit results
        max_results = params.get("max_results", 100)
        matches = matches[:max_results]

        # Format output
        result_list = []
        for match in sorted(matches):
            p = Path(match)
            prefix = "d" if p.is_dir() else "f"
            result_list.append(f"[{prefix}] {match}")

        return ToolResult(
            tool_name="list_files",
            success=True,
            output="\n".join(result_list) if result_list else "No matches found",
        )
    except Exception as e:
        return ToolResult(
            tool_name="list_files",
            success=False,
            output=None,
            error=str(e),
        )


def _tool_run_command(params: dict) -> ToolResult:
    """Execute a shell command (requires approval)."""
    command = params.get("command")
    if not command:
        return ToolResult(
            tool_name="run_command",
            success=False,
            output=None,
            error="Missing required parameter: command",
        )

    # Security validation
    is_valid, error = _validate_command(command)
    if not is_valid:
        logger.warning(f"Command validation failed for '{command}': {error}")
        return ToolResult(
            tool_name="run_command",
            success=False,
            output=None,
            error=error,
        )

    try:
        timeout = params.get("timeout", 30)
        if _security_config:
            timeout = min(timeout, _security_config.command_timeout_seconds)

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"

        return ToolResult(
            tool_name="run_command",
            success=result.returncode == 0,
            output=output,
            error=f"Exit code: {result.returncode}" if result.returncode != 0 else None,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool_name="run_command",
            success=False,
            output=None,
            error=f"Command timed out after {timeout} seconds",
        )
    except Exception as e:
        return ToolResult(
            tool_name="run_command",
            success=False,
            output=None,
            error=str(e),
        )


def _tool_web_search(params: dict) -> ToolResult:
    """Search the web using DuckDuckGo Instant Answer API."""
    query = params.get("query")
    if not query:
        return ToolResult(
            tool_name="web_search",
            success=False,
            output=None,
            error="Missing required parameter: query",
        )

    try:
        import urllib.parse
        import urllib.request

        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1"

        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())

        # Extract relevant information
        results = []

        # Abstract (main answer)
        if data.get("Abstract"):
            results.append(f"**Summary**: {data['Abstract']}")
            if data.get("AbstractSource"):
                results.append(f"Source: {data['AbstractSource']}")

        # Related topics
        if data.get("RelatedTopics"):
            results.append("\n**Related:**")
            for topic in data["RelatedTopics"][:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(f"- {topic['Text'][:200]}")

        if not results:
            return ToolResult(
                tool_name="web_search",
                success=True,
                output=f"No instant answer found for '{query}'. Try a more specific query.",
            )

        return ToolResult(
            tool_name="web_search",
            success=True,
            output="\n".join(results),
        )
    except Exception as e:
        return ToolResult(
            tool_name="web_search",
            success=False,
            output=None,
            error=str(e),
        )


# Tool definitions
BUILTIN_TOOLS = [
    Tool(
        name="get_datetime",
        description="Get the current date and time",
        parameters={
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "strftime format string (default: %Y-%m-%d %H:%M:%S)",
                }
            },
            "required": [],
        },
        handler=_tool_get_datetime,
        category="utility",
    ),
    Tool(
        name="read_file",
        description="Read the contents of a local file",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
                "max_size": {
                    "type": "integer",
                    "description": "Maximum file size in bytes (default: 100000)",
                },
            },
            "required": ["path"],
        },
        handler=_tool_read_file,
        category="filesystem",
    ),
    Tool(
        name="list_files",
        description="List files in a directory matching a glob pattern",
        parameters={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Base directory (default: current directory)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (default: *). Use ** for recursive.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 100)",
                },
            },
            "required": [],
        },
        handler=_tool_list_files,
        category="filesystem",
    ),
    Tool(
        name="run_command",
        description="Execute a shell command. Use with caution.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                },
            },
            "required": ["command"],
        },
        handler=_tool_run_command,
        requires_approval=True,
        category="system",
    ),
    Tool(
        name="web_search",
        description="Search the web for information using DuckDuckGo",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
            },
            "required": ["query"],
        },
        handler=_tool_web_search,
        category="web",
    ),
]


def create_default_registry(security_config=None) -> ToolRegistry:
    """Create a ToolRegistry with all built-in tools registered.

    Args:
        security_config: Optional ToolsSecurityConfig for tool validation.
    """
    if security_config:
        _set_security_config(security_config)
        logger.info("Tools security config loaded")

    registry = ToolRegistry()
    for tool in BUILTIN_TOOLS:
        registry.register(tool)
    logger.info(f"Created default registry with {len(BUILTIN_TOOLS)} tools")
    return registry


def create_memory_tools(memory_layer) -> list[Tool]:
    """
    Create memory-related tools that require a MemoryLayer instance.

    Args:
        memory_layer: MemoryLayer instance to use for memory operations

    Returns:
        List of Tool objects for memory operations
    """
    from animus.memory import MemoryType

    def _tool_search_memory(params: dict) -> ToolResult:
        """Search memories."""
        query = params.get("query")
        if not query:
            return ToolResult(
                tool_name="search_memory",
                success=False,
                output=None,
                error="Missing required parameter: query",
            )

        try:
            limit = params.get("limit", 5)
            tags = params.get("tags")
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",")]

            memories = memory_layer.recall(query, tags=tags, limit=limit)

            if not memories:
                return ToolResult(
                    tool_name="search_memory",
                    success=True,
                    output=f"No memories found for '{query}'",
                )

            results = []
            for mem in memories:
                tags_str = f" [tags: {', '.join(mem.tags)}]" if mem.tags else ""
                results.append(f"- {mem.content[:200]}...{tags_str}")

            return ToolResult(
                tool_name="search_memory",
                success=True,
                output="\n".join(results),
            )
        except Exception as e:
            return ToolResult(
                tool_name="search_memory",
                success=False,
                output=None,
                error=str(e),
            )

    def _tool_save_memory(params: dict) -> ToolResult:
        """Save a new memory."""
        content = params.get("content")
        if not content:
            return ToolResult(
                tool_name="save_memory",
                success=False,
                output=None,
                error="Missing required parameter: content",
            )

        try:
            tags = params.get("tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",")]

            memory_type_str = params.get("type", "semantic")
            memory_type = MemoryType(memory_type_str)

            memory = memory_layer.remember(content, memory_type=memory_type, tags=tags)

            return ToolResult(
                tool_name="save_memory",
                success=True,
                output=f"Saved memory with ID: {memory.id[:8]}",
            )
        except Exception as e:
            return ToolResult(
                tool_name="save_memory",
                success=False,
                output=None,
                error=str(e),
            )

    return [
        Tool(
            name="search_memory",
            description="Search through stored memories",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 5)",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags to filter by",
                    },
                },
                "required": ["query"],
            },
            handler=_tool_search_memory,
            category="memory",
        ),
        Tool(
            name="save_memory",
            description="Save new information to memory",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to remember",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags",
                    },
                    "type": {
                        "type": "string",
                        "description": "Memory type: episodic, semantic, procedural",
                    },
                },
                "required": ["content"],
            },
            handler=_tool_save_memory,
            category="memory",
        ),
    ]
