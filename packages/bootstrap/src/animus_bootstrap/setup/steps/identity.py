"""Identity step — collect user name, timezone, and locale."""

from __future__ import annotations

import datetime

from rich.console import Console
from rich.prompt import Confirm, Prompt


def run_identity(console: Console) -> dict[str, str]:
    """Prompt the user for identity information.

    Auto-detects the system timezone and asks the user to confirm or
    override it. Collects the user's preferred name and locale.

    Args:
        console: Rich Console instance for terminal output.

    Returns:
        Dictionary with keys ``name``, ``timezone``, and ``locale``.
    """
    console.print()
    console.print("[bold cyan]Identity[/bold cyan]")
    console.print()

    name = Prompt.ask("[bold]What should Animus call you?[/bold]", console=console)

    # Auto-detect timezone from system clock
    detected_tz = _detect_timezone()
    console.print(f"  Detected timezone: [cyan]{detected_tz}[/cyan]")

    if Confirm.ask("  Use this timezone?", console=console, default=True):
        timezone = detected_tz
    else:
        timezone = Prompt.ask("  Enter your timezone (e.g. America/New_York)", console=console)

    locale = Prompt.ask("[bold]Locale[/bold]", console=console, default="en_US")

    return {"name": name, "timezone": timezone, "locale": locale}


def _detect_timezone() -> str:
    """Detect the system timezone without extra dependencies.

    Uses ``datetime.datetime.now().astimezone().tzinfo`` to get the
    current timezone name. Falls back to ``"UTC"`` if detection fails.

    Returns:
        Timezone name string (e.g. ``"EST"`` or ``"UTC+05:30"``).
    """
    try:
        tz = datetime.datetime.now().astimezone().tzinfo
        if tz is not None:
            tz_name = str(tz)
            if tz_name:
                return tz_name
    except (ValueError, OSError):
        pass
    return "UTC"
