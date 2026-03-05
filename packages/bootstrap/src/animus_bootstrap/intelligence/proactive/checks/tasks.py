"""Task nudge proactive check."""

from __future__ import annotations

from animus_bootstrap.intelligence.proactive.engine import ProactiveCheck

# Task store reference — set at runtime
_task_store = None


def set_task_store(store: object) -> None:
    """Wire the persistent task store for nudge checks."""
    global _task_store  # noqa: PLW0603
    _task_store = store


async def task_nudge_checker() -> str | None:
    """Check for overdue or upcoming tasks and return a nudge if any.

    Returns None if nothing to report.
    """
    if _task_store is None:
        return None

    parts = []

    overdue = _task_store.get_overdue()
    if overdue:
        lines = [
            f"  - {t['name']} (due: {t['due_date']}, priority: {t['priority']})" for t in overdue
        ]
        parts.append(f"Overdue tasks ({len(overdue)}):\n" + "\n".join(lines))

    upcoming = _task_store.get_upcoming(hours=24)
    if upcoming:
        lines = [
            f"  - {t['name']} (due: {t['due_date']}, priority: {t['priority']})" for t in upcoming
        ]
        parts.append(f"Upcoming tasks ({len(upcoming)}):\n" + "\n".join(lines))

    if not parts:
        return None

    return "\n\n".join(parts)


def get_task_nudge_check() -> ProactiveCheck:
    """Return a ProactiveCheck configured for task nudges."""
    return ProactiveCheck(
        name="task_nudge",
        schedule="0 */2 * * *",
        checker=task_nudge_checker,
        channels=["webchat"],
        priority="low",
    )
