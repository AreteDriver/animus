"""Morning brief proactive check."""

from __future__ import annotations

from datetime import UTC, datetime

from animus_bootstrap.intelligence.proactive.engine import ProactiveCheck


async def morning_brief_checker() -> str | None:
    """Generate a morning brief message.

    In production would query calendar, weather, tasks.
    For now returns a template greeting with the date.
    """
    now = datetime.now(UTC)
    return f"Good morning! Today is {now.strftime('%A, %B %d, %Y')}."


def get_morning_brief_check() -> ProactiveCheck:
    """Return a ProactiveCheck configured for morning briefs."""
    return ProactiveCheck(
        name="morning_brief",
        schedule="0 7 * * *",
        checker=morning_brief_checker,
        channels=["webchat"],
        priority="normal",
    )
