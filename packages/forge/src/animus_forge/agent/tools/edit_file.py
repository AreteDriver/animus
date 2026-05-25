"""EditFile tool — spec §4.4. Single-occurrence replace."""

from __future__ import annotations

from pathlib import Path

from ..types import ToolError
from .base import resolve_within_root


def edit_file(path: str, old: str, new: str, project_root: Path) -> str:
    """Replace exactly one occurrence of `old` with `new` in `path`.

    Requires `old` to match exactly once. Multiple matches force the
    caller to refine — silent multi-replace is a footgun the spec
    explicitly avoids.

    Args:
        path: Path inside `project_root`. Must exist.
        old: Exact substring to find (one occurrence required).
        new: Replacement string.
        project_root: Repository / project root the path must stay inside.

    Returns:
        `"replaced 1 occurrence"` on success.

    Raises:
        ToolError: on path escape, missing file, zero matches, multiple
            matches, or non-UTF-8 binary content.
    """
    target = resolve_within_root(path, project_root)
    if not target.exists():
        raise ToolError(f"file not found: {path}")
    if not target.is_file():
        raise ToolError(f"not a regular file: {path}")
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ToolError(f"binary file: {path}") from e

    count = content.count(old)
    if count == 0:
        raise ToolError("string not found")
    if count > 1:
        raise ToolError(f"multiple matches ({count}); refine old")

    target.write_text(content.replace(old, new), encoding="utf-8")
    return "replaced 1 occurrence"
