"""Welcome step — display branding and sovereignty pledge."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.text import Text


def run_welcome(console: Console) -> bool:
    """Display the Animus welcome screen and sovereignty pledge.

    Shows the project name, tagline, and a clear statement about data
    ownership. The user must confirm to proceed.

    Args:
        console: Rich Console instance for terminal output.

    Returns:
        True if the user confirms, False if they decline.
    """
    console.print()

    title = Text("ANIMUS", style="bold cyan", justify="center")
    tagline = Text(
        "Your personal AI \u2014 sovereign, persistent, local-first",
        style="italic",
        justify="center",
    )

    header = Text()
    header.append_text(title)
    header.append("\n")
    header.append_text(tagline)

    console.print(Panel(header, border_style="cyan", padding=(1, 4)))
    console.print()

    console.print("[bold]Sovereignty Pledge[/bold]", style="cyan")
    console.print()
    console.print("  [green]\u2713[/green] Your data stays on your machine. Always.")
    console.print("  [green]\u2713[/green] Nothing is ever phoned home or shared.")
    console.print("  [green]\u2713[/green] Your AI, your data, your rules.")
    console.print()

    return Confirm.ask("[bold]Ready to set up Animus?[/bold]", console=console, default=True)
