"""WriteFile tool — spec §4.3."""

from __future__ import annotations

from pathlib import Path

from .base import resolve_within_root


def write_file(path: str, content: str, project_root: Path) -> str:
    """Write `content` to `path`, creating parent directories as needed.

    Overwrites without warning in v0 (spec §4.3 — no confirmation step).
    The agent loop's identity-guard hook runs BEFORE this tool, blocking
    writes to identity-file roots (spec R14).

    Args:
        path: Path inside `project_root`.
        content: UTF-8 string to write.
        project_root: Repository / project root the path must stay inside.

    Returns:
        Human-readable summary, e.g. `"wrote 42 bytes to foo.txt"`.

    Raises:
        ToolError: on path escape.
    """
    target = resolve_within_root(path, project_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    n_bytes = len(content.encode("utf-8"))
    return f"wrote {n_bytes} bytes to {path}"
