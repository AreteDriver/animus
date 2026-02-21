"""Filesystem tools — sandboxed file read/write."""

from __future__ import annotations

import logging
from pathlib import Path

from animus_bootstrap.intelligence.tools.executor import ToolDefinition

logger = logging.getLogger(__name__)

# Default sandbox roots (user home)
_DEFAULT_ALLOWED_ROOTS: list[str] = [str(Path.home())]


def _validate_path(path: str, allowed_roots: list[str] | None = None) -> Path:
    """Resolve a path and verify it falls under an allowed root directory.

    Raises ValueError if the path escapes the sandbox.
    """
    roots = allowed_roots or _DEFAULT_ALLOWED_ROOTS
    resolved = Path(path).expanduser().resolve()
    for root in roots:
        root_resolved = Path(root).expanduser().resolve()
        try:
            resolved.relative_to(root_resolved)
            return resolved
        except ValueError:
            continue
    raise ValueError(
        f"Path '{path}' (resolved: {resolved}) is outside allowed directories: {roots}"
    )


async def _file_read(path: str, allowed_roots: list[str] | None = None) -> str:
    """Read file content from a sandboxed path."""
    try:
        validated = _validate_path(path, allowed_roots)
    except ValueError as exc:
        return f"Permission denied: {exc}"

    try:
        content = validated.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"File not found: {validated}"
    except PermissionError:
        return f"OS permission denied: {validated}"
    except IsADirectoryError:
        return f"Path is a directory, not a file: {validated}"
    except OSError as exc:
        return f"Read error: {exc}"

    return content


async def _file_write(path: str, content: str, allowed_roots: list[str] | None = None) -> str:
    """Write content to a file at a sandboxed path."""
    try:
        validated = _validate_path(path, allowed_roots)
    except ValueError as exc:
        return f"Permission denied: {exc}"

    try:
        validated.parent.mkdir(parents=True, exist_ok=True)
        validated.write_text(content, encoding="utf-8")
    except PermissionError:
        return f"OS permission denied: {validated}"
    except OSError as exc:
        return f"Write error: {exc}"

    return f"Successfully wrote {len(content)} bytes to {validated}"


def get_filesystem_tools(
    allowed_roots: list[str] | None = None,
) -> list[ToolDefinition]:
    """Return filesystem tool definitions with optional custom sandbox roots."""
    roots = allowed_roots

    async def file_read_handler(path: str) -> str:
        return await _file_read(path, roots)

    async def file_write_handler(path: str, content: str) -> str:
        return await _file_write(path, content, roots)

    return [
        ToolDefinition(
            name="file_read",
            description="Read the contents of a file. Path must be under allowed directories.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                },
                "required": ["path"],
            },
            handler=file_read_handler,
            category="filesystem",
        ),
        ToolDefinition(
            name="file_write",
            description="Write content to a file. Path must be under allowed directories.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
            handler=file_write_handler,
            category="filesystem",
        ),
    ]
