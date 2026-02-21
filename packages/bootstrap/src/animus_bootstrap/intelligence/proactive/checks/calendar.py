"""Calendar reminder proactive check."""

from __future__ import annotations

from animus_bootstrap.intelligence.proactive.engine import ProactiveCheck


async def calendar_reminder_checker() -> str | None:
    """Check for upcoming calendar events.

    Stub -- would query Google Calendar API. Returns None.
    """
    return None


def get_calendar_check() -> ProactiveCheck:
    """Return a ProactiveCheck configured for calendar reminders (disabled by default)."""
    return ProactiveCheck(
        name="calendar_reminder",
        schedule="every 15m",
        checker=calendar_reminder_checker,
        channels=["webchat"],
        priority="normal",
        enabled=False,
    )
