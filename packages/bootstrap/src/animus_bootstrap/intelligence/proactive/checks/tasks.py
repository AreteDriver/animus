"""Task nudge proactive check."""

from __future__ import annotations

from animus_bootstrap.intelligence.proactive.engine import ProactiveCheck


async def task_nudge_checker() -> str | None:
    """Check for overdue or stale tasks.

    Stub -- would query task system. Returns None (nothing to report).
    """
    return None


def get_task_nudge_check() -> ProactiveCheck:
    """Return a ProactiveCheck configured for task nudges."""
    return ProactiveCheck(
        name="task_nudge",
        schedule="0 */2 * * *",
        checker=task_nudge_checker,
        channels=["webchat"],
        priority="low",
    )
