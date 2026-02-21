"""Sovereignty step — telemetry, data locality, and privacy settings."""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table


def run_sovereignty(console: Console) -> dict[str, bool]:
    """Confirm data sovereignty settings and display storage summary.

    Telemetry is off by default. The user acknowledges that all data
    remains local and that Animus is single-user by design.

    Args:
        console: Rich Console instance for terminal output.

    Returns:
        Dictionary with keys ``telemetry`` and ``data_local_only``.
    """
    console.print()
    console.print("[bold cyan]Data Sovereignty[/bold cyan]")
    console.print()

    # Telemetry
    telemetry = Confirm.ask(
        "[bold]Enable anonymous usage telemetry?[/bold]",
        console=console,
        default=False,
    )
    if not telemetry:
        console.print("  [green]\u2713[/green] Telemetry disabled. No data leaves your machine.")
    else:
        console.print("  [yellow]Telemetry enabled.[/yellow] Only anonymous usage stats are sent.")

    # Data locality
    console.print()
    data_local_only = Confirm.ask(
        "[bold]Confirm: all data stays local to this machine?[/bold]",
        console=console,
        default=True,
    )

    # Single-user acknowledgement
    console.print()
    console.print("  [dim]Animus is single-user by design. One identity per installation.[/dim]")

    # Data storage summary
    console.print()
    table = Table(title="Data Storage Summary", border_style="cyan")
    table.add_column("Data", style="bold")
    table.add_column("Location", style="cyan")
    table.add_column("Format")

    table.add_row("Configuration", "~/.config/animus/config.toml", "TOML")
    table.add_row("Memory DB", "~/.local/share/animus/memory.db", "SQLite")
    table.add_row("Episodic memory", "~/.local/share/animus/episodes/", "JSON")
    table.add_row("Workflow logs", "~/.local/share/animus/workflows/", "SQLite")
    table.add_row("API keys", "~/.config/animus/config.toml (0600)", "Encrypted at rest")

    console.print(table)

    return {"telemetry": telemetry, "data_local_only": data_local_only}
