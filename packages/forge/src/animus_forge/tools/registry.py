"""Tool registry for LLM-accessible tool execution.

Bridges FilesystemTools into a format consumable by LLM tool_use APIs
(Anthropic, OpenAI, Ollama). Each tool has a JSON Schema definition
and an execute() handler.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from animus_forge.tools.filesystem import FilesystemTools
from animus_forge.tools.safety import PathValidator

logger = logging.getLogger(__name__)

# Dedicated audit logger — writes structured JSON to forge_audit handler
audit_logger = logging.getLogger("forge.tool_audit")

# Maximum output length returned to the LLM to avoid context bloat
MAX_TOOL_OUTPUT_CHARS = 8000


@dataclass
class ToolDefinition:
    """A tool that can be called by an LLM."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Any = field(repr=False)  # Callable[[dict], str]

    def to_anthropic(self) -> dict:
        """Convert to Anthropic tool_use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_ollama(self) -> dict:
        """Convert to Ollama /api/chat tools format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ForgeToolRegistry:
    """Registry of tools available to Forge sub-agents.

    Wraps FilesystemTools and shell execution into LLM-callable tools
    with JSON Schema parameter definitions.
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        enable_shell: bool = False,
        allowed_commands: list[str] | None = None,
        require_write_approval: bool = False,
        budget_manager: Any = None,
        budget_tokens_per_call: int = 100,
    ):
        """Initialize the tool registry.

        Args:
            project_root: Root directory for filesystem operations.
            enable_shell: Whether to enable the run_command tool.
            allowed_commands: Whitelist of allowed shell commands (if enable_shell).
            require_write_approval: If True, write_file returns a proposal
                instead of writing directly.
            budget_manager: Optional BudgetManager for per-call budget tracking.
            budget_tokens_per_call: Estimated tokens charged per tool call.
        """
        self._tools: dict[str, ToolDefinition] = {}
        self._fs: FilesystemTools | None = None
        self._project_root = Path(project_root) if project_root else Path.cwd()
        self._enable_shell = enable_shell
        self._allowed_commands = set(allowed_commands or [
            "python", "python3", "pytest", "ruff", "git", "ls", "cat", "grep", "find",
            "pip", "poetry", "cargo", "npm", "node",
        ])
        self._require_write_approval = require_write_approval
        self._pending_writes: list[dict] = []
        self._budget_manager = budget_manager
        self._budget_tokens_per_call = budget_tokens_per_call

        self._register_filesystem_tools()
        if enable_shell:
            self._register_shell_tool()

    def _get_fs(self) -> FilesystemTools:
        """Lazy-init FilesystemTools."""
        if self._fs is None:
            validator = PathValidator(project_path=self._project_root)
            self._fs = FilesystemTools(validator=validator)
        return self._fs

    def _register_filesystem_tools(self) -> None:
        """Register all filesystem tools."""
        self.register(ToolDefinition(
            name="read_file",
            description="Read a file's content. Returns numbered lines.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to project root)"},
                    "start_line": {"type": "integer", "description": "Start line (1-indexed, optional)"},
                    "end_line": {"type": "integer", "description": "End line (1-indexed, optional)"},
                },
                "required": ["path"],
            },
            handler=self._handle_read_file,
        ))

        self.register(ToolDefinition(
            name="list_files",
            description="List files in a directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: '.')"},
                    "pattern": {"type": "string", "description": "Glob pattern to filter"},
                    "recursive": {"type": "boolean", "description": "List recursively (default: false)"},
                },
                "required": [],
            },
            handler=self._handle_list_files,
        ))

        self.register(ToolDefinition(
            name="search_code",
            description="Search for a regex pattern in project files.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search (default: '.')"},
                    "file_pattern": {"type": "string", "description": "Glob to filter files (e.g. '*.py')"},
                },
                "required": ["pattern"],
            },
            handler=self._handle_search_code,
        ))

        self.register(ToolDefinition(
            name="get_project_structure",
            description="Get a tree overview of the project structure.",
            parameters={
                "type": "object",
                "properties": {
                    "max_depth": {"type": "integer", "description": "Max tree depth (default: 3)"},
                },
                "required": [],
            },
            handler=self._handle_get_structure,
        ))

        self.register(ToolDefinition(
            name="write_file",
            description="Write content to a file. Creates the file if it doesn't exist.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to project root)"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
            handler=self._handle_write_file,
        ))

    def _register_shell_tool(self) -> None:
        """Register the shell command tool."""
        self.register(ToolDefinition(
            name="run_command",
            description=(
                "Run a shell command in the project directory. "
                f"Allowed commands: {', '.join(sorted(self._allowed_commands))}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"},
                },
                "required": ["command"],
            },
            handler=self._handle_run_command,
        ))

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    @property
    def tools(self) -> list[ToolDefinition]:
        """All registered tools."""
        return list(self._tools.values())

    def to_anthropic_tools(self) -> list[dict]:
        """Export all tools in Anthropic API format."""
        return [t.to_anthropic() for t in self._tools.values()]

    def to_ollama_tools(self) -> list[dict]:
        """Export all tools in Ollama API format."""
        return [t.to_ollama() for t in self._tools.values()]

    def execute(self, tool_name: str, arguments: dict, agent_id: str = "") -> str:
        """Execute a tool by name with the given arguments.

        Emits a structured audit log entry for every call. Enforces budget
        limits when a budget_manager is configured.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments dict.
            agent_id: Calling agent identifier (for audit/budget).

        Returns:
            String result (truncated if too long).
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            self._emit_audit(tool_name, arguments, agent_id, False, "unknown_tool")
            return f"Error: Unknown tool '{tool_name}'"

        # Budget gate — reject if budget exhausted
        if self._budget_manager is not None:
            if not self._budget_manager.can_allocate(
                self._budget_tokens_per_call, agent_id=agent_id or None,
            ):
                self._emit_audit(
                    tool_name, arguments, agent_id, False, "budget_exceeded",
                )
                return (
                    f"Error: Budget exceeded — cannot execute {tool_name}. "
                    f"Remaining: {self._budget_manager.remaining} tokens."
                )

        start = time.monotonic()
        try:
            result = tool.handler(arguments)
            duration_ms = (time.monotonic() - start) * 1000

            if len(result) > MAX_TOOL_OUTPUT_CHARS:
                result = result[:MAX_TOOL_OUTPUT_CHARS] + "\n... (truncated)"

            # Record budget usage
            if self._budget_manager is not None:
                try:
                    self._budget_manager.record_usage(
                        agent_id=agent_id or "tool",
                        tokens=self._budget_tokens_per_call,
                        operation=f"tool:{tool_name}",
                    )
                except Exception:
                    pass  # Budget recording is advisory

            self._emit_audit(
                tool_name, arguments, agent_id, True, duration_ms=duration_ms,
            )
            return result
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            logger.warning("Tool %s execution failed: %s", tool_name, e)
            self._emit_audit(
                tool_name, arguments, agent_id, False,
                error=str(e), duration_ms=duration_ms,
            )
            return f"Error executing {tool_name}: {e}"

    def _emit_audit(
        self,
        tool_name: str,
        arguments: dict,
        agent_id: str,
        success: bool,
        error: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        """Emit a structured audit log entry for a tool execution."""
        # Sanitize arguments — don't log full file content
        safe_args = dict(arguments)
        if "content" in safe_args:
            content = safe_args["content"]
            safe_args["content"] = (
                f"({len(content)} chars)" if isinstance(content, str) else "(binary)"
            )

        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "tool_execution",
            "tool": tool_name,
            "agent_id": agent_id,
            "arguments": safe_args,
            "success": success,
            "duration_ms": round(duration_ms, 2),
        }
        if error:
            entry["error"] = error

        try:
            audit_logger.info(json.dumps(entry, default=str))
        except Exception:
            pass  # Audit logging must never break execution

    # --- Tool handlers ---

    def _handle_read_file(self, args: dict) -> str:
        path = args.get("path", "")
        start = args.get("start_line")
        end = args.get("end_line")
        result = self._get_fs().read_file(path, start_line=start, end_line=end)
        return f"File: {result.path} ({result.line_count} lines, {result.size_bytes} bytes)\n{result.content}"

    def _handle_list_files(self, args: dict) -> str:
        path = args.get("path", ".")
        pattern = args.get("pattern")
        recursive = args.get("recursive", False)
        result = self._get_fs().list_files(path=path, pattern=pattern, recursive=recursive)
        lines = [f"Directory: {result.path} ({result.total_files} files, {result.total_dirs} dirs)"]
        for entry in result.entries:
            prefix = "d " if entry.is_dir else "f "
            size = f" ({entry.size_bytes}B)" if entry.size_bytes else ""
            lines.append(f"  {prefix}{entry.path}{size}")
        if result.truncated:
            lines.append("  ... (truncated)")
        return "\n".join(lines)

    def _handle_search_code(self, args: dict) -> str:
        pattern = args.get("pattern", "")
        path = args.get("path", ".")
        file_pattern = args.get("file_pattern")
        result = self._get_fs().search_code(
            pattern=pattern, path=path, file_pattern=file_pattern, max_results=50,
        )
        lines = [f"Search: '{pattern}' — {result.total_matches} matches in {result.files_searched} files"]
        for m in result.matches:
            lines.append(f"  {m.path}:{m.line_number}: {m.line_content}")
        if result.truncated:
            lines.append("  ... (truncated)")
        return "\n".join(lines)

    def _handle_get_structure(self, args: dict) -> str:
        depth = args.get("max_depth", 3)
        result = self._get_fs().get_structure(max_depth=depth)
        return f"Project: {result.root_path} ({result.total_files} files, {result.total_dirs} dirs)\n{result.tree}"

    def _handle_write_file(self, args: dict) -> str:
        path = args.get("path", "")
        content = args.get("content", "")

        if self._require_write_approval:
            self._pending_writes.append({
                "path": path,
                "content": content,
                "timestamp": datetime.now(UTC).isoformat(),
            })
            return (
                f"Write to {path} ({len(content)} bytes) queued for approval. "
                f"Total pending writes: {len(self._pending_writes)}. "
                "Continue with your task — writes will be applied after review."
            )

        resolved = self._project_root / path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {path}"

    @property
    def pending_writes(self) -> list[dict]:
        """Get pending write proposals (when require_write_approval is True)."""
        return list(self._pending_writes)

    def approve_write(self, index: int) -> str:
        """Approve and apply a pending write by index.

        Args:
            index: Index into pending_writes list.

        Returns:
            Result message.
        """
        if index < 0 or index >= len(self._pending_writes):
            return f"Error: Invalid write index {index}"

        write = self._pending_writes.pop(index)
        path = write["path"]
        content = write["content"]
        resolved = self._project_root / path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Approved and written {len(content)} bytes to {path}"

    def approve_all_writes(self) -> list[str]:
        """Approve and apply all pending writes.

        Returns:
            List of result messages.
        """
        results = []
        while self._pending_writes:
            write = self._pending_writes.pop(0)
            path = write["path"]
            content = write["content"]
            resolved = self._project_root / path
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            results.append(f"Written {len(content)} bytes to {path}")
        return results

    def reject_all_writes(self) -> int:
        """Reject and discard all pending writes.

        Returns:
            Number of writes discarded.
        """
        count = len(self._pending_writes)
        self._pending_writes.clear()
        return count

    def _handle_run_command(self, args: dict) -> str:
        command = args.get("command", "")
        timeout = args.get("timeout", 30)

        # Validate command against whitelist
        cmd_parts = command.split()
        if not cmd_parts:
            return "Error: Empty command"

        base_cmd = Path(cmd_parts[0]).name
        if base_cmd not in self._allowed_commands:
            return f"Error: Command '{base_cmd}' not in allowed list: {sorted(self._allowed_commands)}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self._project_root),
            )
            output = result.stdout
            if result.returncode != 0:
                output += f"\nSTDERR:\n{result.stderr}" if result.stderr else ""
                output += f"\nExit code: {result.returncode}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            return f"Error running command: {e}"
