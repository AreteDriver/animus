"""Shared helpers for the starter tools — path validation primarily."""

from __future__ import annotations

from pathlib import Path

from ..types import ToolError


def resolve_within_root(path: str, project_root: Path) -> Path:
    """Resolve `path` and confirm it is inside `project_root`.

    Symlink-aware: a symlink targeting a path outside the root counts as
    escape. Non-existent paths are allowed (writers create new files); their
    resolved-parent must still be inside the root.

    Raises:
        ToolError("path escapes project root"): if the resolved path is
            outside `project_root` after following symlinks.
    """
    target = (project_root / path).expanduser() if not Path(path).is_absolute() else Path(path)
    try:
        resolved_target = target.resolve(strict=False)
        resolved_root = project_root.resolve(strict=True)
    except (OSError, RuntimeError) as e:
        raise ToolError(f"path resolution failed: {e}") from e

    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as e:
        raise ToolError("path escapes project root") from e
    return resolved_target
