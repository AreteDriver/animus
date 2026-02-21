"""Forge step — detect or configure the Forge orchestration engine."""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Confirm, Prompt

from animus_bootstrap.setup.validators import test_forge_connection

_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 8000


def run_forge(console: Console) -> dict[str, bool | str | int]:
    """Auto-detect or manually configure a Forge connection.

    Attempts to reach Forge at ``localhost:8000``. If found, confirms
    with the user. If not found, offers manual configuration or skip.

    Args:
        console: Rich Console instance for terminal output.

    Returns:
        Dictionary with keys ``enabled``, ``host``, ``port``, and ``api_key``.
    """
    console.print()
    console.print("[bold cyan]Forge Orchestration Engine[/bold cyan]")
    console.print()

    # Auto-detect
    with console.status("[cyan]Scanning for Forge...[/cyan]"):
        detected = test_forge_connection(_DEFAULT_HOST, _DEFAULT_PORT)

    if detected:
        console.print(
            f"  [green]\u2713[/green] Forge detected at "
            f"[bold]{_DEFAULT_HOST}:{_DEFAULT_PORT}[/bold]"
        )
        if Confirm.ask("  Use this instance?", console=console, default=True):
            api_key = Prompt.ask(
                "  Forge API key (leave blank if none)", console=console, default=""
            )
            return {
                "enabled": True,
                "host": _DEFAULT_HOST,
                "port": _DEFAULT_PORT,
                "api_key": api_key,
            }

    else:
        console.print("  [dim]Forge not detected at localhost:8000[/dim]")

    # Manual entry or skip
    console.print()
    if Confirm.ask("  Configure Forge manually?", console=console, default=False):
        host = Prompt.ask("  Forge host", console=console, default=_DEFAULT_HOST)
        port_str = Prompt.ask("  Forge port", console=console, default=str(_DEFAULT_PORT))
        try:
            port = int(port_str)
        except ValueError:
            console.print(f"  [yellow]Invalid port '{port_str}', using {_DEFAULT_PORT}[/yellow]")
            port = _DEFAULT_PORT

        api_key = Prompt.ask("  Forge API key (leave blank if none)", console=console, default="")
        return {"enabled": True, "host": host, "port": port, "api_key": api_key}

    console.print("  [dim]Skipping Forge. You can configure it later.[/dim]")
    return {"enabled": False, "host": _DEFAULT_HOST, "port": _DEFAULT_PORT, "api_key": ""}
