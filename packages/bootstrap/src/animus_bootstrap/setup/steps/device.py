"""Device step — identify this machine's name and role."""

from __future__ import annotations

import socket

from rich.console import Console
from rich.prompt import IntPrompt, Prompt

_ROLES = {
    1: "primary",
    2: "secondary",
    3: "mobile",
}


def run_device(console: Console) -> dict[str, str]:
    """Prompt for the machine name and its role in the Animus network.

    Auto-detects the system hostname as the default machine name.
    Offers three roles: Primary, Secondary, and Mobile.

    Args:
        console: Rich Console instance for terminal output.

    Returns:
        Dictionary with keys ``machine_name`` and ``role``.
    """
    console.print()
    console.print("[bold cyan]Device Configuration[/bold cyan]")
    console.print()

    default_hostname = socket.gethostname()
    machine_name = Prompt.ask(
        "[bold]Machine name[/bold]", console=console, default=default_hostname
    )

    console.print()
    console.print("  [bold]1.[/bold] [cyan]Primary[/cyan] \u2014 Main workstation")
    console.print("  [bold]2.[/bold] [cyan]Secondary[/cyan] \u2014 Additional machine")
    console.print("  [bold]3.[/bold] [cyan]Mobile[/cyan] \u2014 Laptop or mobile device")
    console.print()

    choice = IntPrompt.ask(
        "[bold]Select role[/bold]",
        console=console,
        default=1,
        choices=["1", "2", "3"],
    )
    role = _ROLES[choice]
    console.print(f"  Role: [cyan]{role}[/cyan]")

    console.print()
    console.print("  [dim]Note: Multi-device sync is planned for a future release.[/dim]")

    return {"machine_name": machine_name, "role": role}
