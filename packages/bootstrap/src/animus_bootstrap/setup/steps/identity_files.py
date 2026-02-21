"""Identity files step — generate initial identity files during wizard setup."""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Prompt


def run_identity_files(console: Console) -> dict[str, object]:
    """Generate identity files from wizard-collected data.

    Displays what files will be generated, asks for optional "About Me"
    text, and returns the context dict needed for template rendering.
    The actual file generation is handled later in the wizard's
    ``_generate_identity_files()`` method.

    Args:
        console: Rich Console instance for terminal output.

    Returns:
        Dictionary with keys ``about`` and ``generate_identity_files``.
    """
    console.print()
    console.print("[bold cyan]Identity Files[/bold cyan]")
    console.print()
    console.print("  Animus uses identity files to understand who you are")
    console.print("  and how to communicate with you. These will be created:")
    console.print()

    files = [
        ("CORE_VALUES.md", "Foundational values — immutable, human-edit only"),
        ("IDENTITY.md", "Who you are — name, background, role"),
        ("CONTEXT.md", "Current projects and priorities"),
        ("GOALS.md", "Short and long-term goals"),
        ("PREFERENCES.md", "Communication style preferences"),
        ("LEARNED.md", "What Animus learns about you over time"),
    ]
    for name, desc in files:
        console.print(f"    [green]{name:20s}[/green] {desc}")

    console.print()
    about = Prompt.ask(
        "[bold]Tell Animus about yourself[/bold] (optional, press Enter to skip)",
        console=console,
        default="",
    )

    return {"about": about, "generate_identity_files": True}
