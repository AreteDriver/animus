"""System tools — shell command execution with approval gate."""

from __future__ import annotations

import asyncio
import logging

from animus_bootstrap.intelligence.tools.executor import ToolDefinition

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0


async def _shell_exec(command: str, timeout: float = _DEFAULT_TIMEOUT) -> str:
    """Execute a shell command and return stdout + stderr."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        return f"Command timed out after {timeout}s: {command}"
    except OSError as exc:
        return f"Failed to execute command: {exc}"

    parts: list[str] = []
    if stdout:
        parts.append(stdout.decode(errors="replace"))
    if stderr:
        parts.append(f"[stderr] {stderr.decode(errors='replace')}")
    if not parts:
        parts.append(f"(exit code {proc.returncode})")
    return "\n".join(parts)


def get_system_tools() -> list[ToolDefinition]:
    """Return system tool definitions."""
    return [
        ToolDefinition(
            name="shell_exec",
            description=(
                "Execute a shell command and return its output. "
                "Requires user approval before execution."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                },
                "required": ["command"],
            },
            handler=_shell_exec,
            category="system",
            permission="approve",
        ),
    ]
