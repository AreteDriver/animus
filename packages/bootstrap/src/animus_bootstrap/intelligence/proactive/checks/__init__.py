"""Built-in proactive checks."""

from __future__ import annotations

from animus_bootstrap.intelligence.proactive.checks.calendar import get_calendar_check
from animus_bootstrap.intelligence.proactive.checks.morning_brief import (
    get_morning_brief_check,
)
from animus_bootstrap.intelligence.proactive.checks.reflection import get_reflection_check
from animus_bootstrap.intelligence.proactive.checks.tasks import get_task_nudge_check
from animus_bootstrap.intelligence.proactive.engine import ProactiveCheck


def get_builtin_checks() -> list[ProactiveCheck]:
    """Return all built-in proactive checks."""
    return [
        get_morning_brief_check(),
        get_task_nudge_check(),
        get_calendar_check(),
        get_reflection_check(),
    ]


__all__ = [
    "get_builtin_checks",
    "get_calendar_check",
    "get_morning_brief_check",
    "get_reflection_check",
    "get_task_nudge_check",
]
