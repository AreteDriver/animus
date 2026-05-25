"""ReadFile tool — spec §4.2."""

from __future__ import annotations

from pathlib import Path

from ..types import ToolError
from .base import resolve_within_root


def read_file(
    path: str,
    project_root: Path,
    offset: int = 0,
    limit: int | None = None,
) -> str:
    """Read a file's text contents.

    Args:
        path: Path inside `project_root` (relative or absolute).
        project_root: Repository / project root the path must stay inside.
        offset: Character offset to start reading from. Default 0.
        limit: Max characters to return. `None` reads to EOF.

    Returns:
        The requested slice of file contents as a UTF-8 string.

    Raises:
        ToolError: on path escape, missing file, or non-UTF-8 binary content.
    """
    target = resolve_within_root(path, project_root)
    if not target.exists():
        raise ToolError(f"file not found: {path}")
    if not target.is_file():
        raise ToolError(f"not a regular file: {path}")
    try:
        contents = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ToolError(f"binary file: {path}") from e

    if offset < 0:
        raise ToolError(f"negative offset not allowed: {offset}")
    if limit is None:
        return contents[offset:]
    if limit < 0:
        raise ToolError(f"negative limit not allowed: {limit}")
    return contents[offset : offset + limit]
