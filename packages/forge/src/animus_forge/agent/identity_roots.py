"""Identity-file roots — paths the built-in identity guard refuses to write.

Per spec RD4 + R14 + C13: the agent's `identity_guard` pre-tool-use hook
denies `WriteFile`/`EditFile` on any path under these roots, fail-closed
and runs before all user-supplied hooks. Override requires an explicit
user `Allow` from a separately-installed hook (per spec R26).

This module exports `IDENTITY_ROOTS` so the list is greppable, testable,
and editable in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

# Roots are repo-relative; the guard resolves them against the project
# root at hook-invocation time. Keeping them repo-relative makes the
# guard portable across user homes and worktrees.
IDENTITY_ROOTS: tuple[str, ...] = (
    "packages/core/animus/CORE_VALUES.md",
    "packages/core/animus/identity",
    "packages/bootstrap/src/animus_bootstrap/persona",
)


def is_identity_path(target: Path, project_root: Path) -> bool:
    """True if `target` is inside any IDENTITY_ROOTS entry under `project_root`.

    Path resolution is symlink-aware: a symlink pointing into an identity
    root is treated as an identity path. Non-existent paths are checked by
    string prefix against the resolved project root.
    """
    try:
        resolved_target = target.resolve(strict=False)
        resolved_root = project_root.resolve(strict=True)
    except (OSError, RuntimeError):
        # If we can't even resolve the project root, fail closed.
        return True

    for rel in IDENTITY_ROOTS:
        root_path = (resolved_root / rel).resolve(strict=False)
        try:
            resolved_target.relative_to(root_path)
            return True
        except ValueError:
            continue
        # Also catch the exact-match case for a file root (CORE_VALUES.md).
        if resolved_target == root_path:
            return True
    return False
