"""Built-in identity-file deny hook — spec R14 + RD4 + A16.

Denies `WriteFile` and `EditFile` on any path inside `IDENTITY_ROOTS`.
Fail-closed: if path resolution fails, deny. Runs before all user hooks
and before the P1-P9 stub. A user `Allow` after this hook's `Deny` does
NOT override (spec R26).
"""

from __future__ import annotations

from pathlib import Path

from ...identity_roots import is_identity_path
from ...types import AgentContext, Allow, Deny, HookDecision

_GUARDED_TOOLS = frozenset({"WriteFile", "EditFile"})


def pre_tool_use(context: AgentContext, tool: str, args: dict) -> HookDecision:
    """Deny if `tool` is a write/edit and target path is under IDENTITY_ROOTS."""
    if tool not in _GUARDED_TOOLS:
        return Allow()

    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        # Malformed args — let the tool itself raise its own ToolError.
        return Allow()

    try:
        target = (context.project_root / raw_path).expanduser()
        if Path(raw_path).is_absolute():
            target = Path(raw_path)
    except (OSError, RuntimeError):
        return Deny(f"identity guard: path resolution failed for {raw_path!r}")

    if is_identity_path(target, context.project_root):
        return Deny(
            f"identity-file write blocked by built-in guard: {raw_path} "
            "(see agent.identity_roots.IDENTITY_ROOTS)"
        )
    return Allow()
