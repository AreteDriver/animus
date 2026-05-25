"""Grep tool — spec §4.6."""

from __future__ import annotations

import re
from pathlib import Path

from ..types import ToolError
from .base import resolve_within_root

_RESULT_CAP = 100  # spec §4.6


def grep(
    pattern: str,
    project_root: Path,
    path: str = ".",
    glob: str = "**/*",
    case_insensitive: bool = False,
) -> list[dict]:
    """Search files matching `glob` under `path` for `pattern`.

    Args:
        pattern: Python `re` regex.
        project_root: Repository / project root the search must stay inside.
        path: Subdirectory under `project_root`. Default ".".
        glob: Path glob relative to `path`. Default "**/*".
        case_insensitive: If True, applies `re.IGNORECASE`.

    Returns:
        Up to 100 hit dicts `{"file": str, "line": int, "text": str}` plus
        a sentinel `{"_truncated": True}` when more matches existed.

    Raises:
        ToolError: on path escape or invalid regex.
    """
    base = resolve_within_root(path, project_root)
    if not base.exists():
        raise ToolError(f"path not found: {path}")

    try:
        rx = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
    except re.error as e:
        raise ToolError(f"invalid regex: {e}") from e

    project_resolved = project_root.resolve(strict=True)
    results: list[dict] = []

    candidates = [base] if base.is_file() else sorted(base.glob(glob))
    for candidate in candidates:
        if len(results) >= _RESULT_CAP:
            break
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        try:
            rel = candidate.resolve(strict=True).relative_to(project_resolved)
        except (ValueError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                results.append({"file": str(rel), "line": lineno, "text": line})
                if len(results) >= _RESULT_CAP:
                    break

    if len(results) >= _RESULT_CAP:
        # Try to detect whether more matches existed beyond the cap.
        # Cheap heuristic: if we hit the cap, append the sentinel.
        results.append({"_truncated": True})
    return results
