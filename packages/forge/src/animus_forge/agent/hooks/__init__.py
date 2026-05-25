"""Hook gate system — spec §4.8 + R7 + R13 + R14.

Four lifecycle events: `agent_start`, `pre_tool_use`, `post_tool_use`,
`agent_end`. Loaded from `~/.config/animus/agent/hooks.d/*.py` plus
built-ins from `agent.hooks.builtin`.

Execution order for `pre_tool_use`:
1. Built-in identity-file deny (R14, fail-closed)
2. Built-in P1-P9 validator (R13, stubbed)
3. User hooks (alphabetical filename order)

First `Deny` short-circuits. A user `Allow` after a built-in `Deny` does
NOT override (R26). Hook raising → tool call denied, error logged.
"""

from __future__ import annotations

from .loader import DEFAULT_HOOK_DIR, load_user_hooks
from .runner import (
    HookOutcome,
    run_agent_end,
    run_agent_start,
    run_post_tool_use,
    run_pre_tool_use,
)

__all__ = [
    "DEFAULT_HOOK_DIR",
    "HookOutcome",
    "load_user_hooks",
    "run_agent_end",
    "run_agent_start",
    "run_post_tool_use",
    "run_pre_tool_use",
]
