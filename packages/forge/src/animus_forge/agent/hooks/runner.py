"""Hook orchestration — runs built-ins + user hooks in spec §4.8 order."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType

from ..types import AgentContext, Allow, Deny, HookDecision, Receipt, ToolResult
from .builtin import identity_guard, p1_p9_stub

logger = logging.getLogger(__name__)


@dataclass
class HookOutcome:
    """Result of running a `pre_tool_use` chain."""

    allowed: bool
    deny_reason: str = ""
    hook_errors: list[str] = field(default_factory=list)


def _builtin_pre_tool_use() -> list[tuple[str, Callable]]:
    """Return built-in pre_tool_use hooks in the spec §4.8 order.

    Order matters: identity_guard runs first (fail-closed), P1-P9 second.
    User hooks come after (alphabetically) in `run_pre_tool_use`.
    """
    return [
        ("agent.hooks.builtin.identity_guard", identity_guard.pre_tool_use),
        ("agent.hooks.builtin.p1_p9_stub", p1_p9_stub.pre_tool_use),
    ]


def run_pre_tool_use(
    context: AgentContext,
    tool: str,
    args: dict,
    user_hooks: list[ModuleType],
) -> HookOutcome:
    """Run pre_tool_use in spec §4.8 order; first Deny short-circuits.

    Hook raising → tool call denied (fail-closed), error captured for
    the receipt's hook_errors list.
    """
    chain: list[tuple[str, Callable]] = list(_builtin_pre_tool_use())
    for module in user_hooks:
        fn = getattr(module, "pre_tool_use", None)
        if callable(fn):
            chain.append((module.__name__, fn))

    errors: list[str] = []
    for name, fn in chain:
        try:
            decision: HookDecision = fn(context, tool, args)
        except Exception as e:  # noqa: BLE001 — hook isolation
            err = f"{name} raised in pre_tool_use: {e}"
            logger.warning(err)
            errors.append(err)
            return HookOutcome(allowed=False, deny_reason=err, hook_errors=errors)
        if isinstance(decision, Deny):
            return HookOutcome(allowed=False, deny_reason=decision.reason, hook_errors=errors)
        if not isinstance(decision, Allow):
            err = (
                f"{name} returned {type(decision).__name__}; "
                "expected Allow or Deny — treating as Deny (fail-closed)"
            )
            logger.warning(err)
            errors.append(err)
            return HookOutcome(allowed=False, deny_reason=err, hook_errors=errors)
    return HookOutcome(allowed=True, hook_errors=errors)


def _run_fire_and_forget(
    name: str,
    fn: Callable,
    args: tuple,
    errors: list[str],
) -> None:
    """Invoke a non-decision hook; capture exceptions for hook_errors."""
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001 — hook isolation
        err = f"{name} raised: {e}"
        logger.warning(err)
        errors.append(err)


def run_agent_start(context: AgentContext, user_hooks: list[ModuleType]) -> list[str]:
    """Fire `agent_start` on every user hook module that defines it."""
    errors: list[str] = []
    for module in user_hooks:
        fn = getattr(module, "agent_start", None)
        if callable(fn):
            _run_fire_and_forget(module.__name__, fn, (context,), errors)
    return errors


def run_post_tool_use(
    context: AgentContext,
    tool: str,
    args: dict,
    result: ToolResult,
    user_hooks: list[ModuleType],
) -> list[str]:
    """Fire `post_tool_use` on every user hook module that defines it."""
    errors: list[str] = []
    for module in user_hooks:
        fn = getattr(module, "post_tool_use", None)
        if callable(fn):
            _run_fire_and_forget(module.__name__, fn, (context, tool, args, result), errors)
    return errors


def run_agent_end(
    context: AgentContext, receipt: Receipt, user_hooks: list[ModuleType]
) -> list[str]:
    """Fire `agent_end` on every user hook module that defines it."""
    errors: list[str] = []
    for module in user_hooks:
        fn = getattr(module, "agent_end", None)
        if callable(fn):
            _run_fire_and_forget(module.__name__, fn, (context, receipt), errors)
    return errors
