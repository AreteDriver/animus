"""Setup commands — init, version, completion, tui."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.panel import Panel

from ..helpers import console

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _show_completion_instructions(shell: str) -> None:
    """Show manual completion installation instructions."""
    instructions = {
        "bash": """
# Add to ~/.bashrc:
eval "$(_GORGON_COMPLETE=bash_source gorgon)"

# Or generate and source a file:
_GORGON_COMPLETE=bash_source gorgon > ~/.gorgon-complete.bash
echo 'source ~/.gorgon-complete.bash' >> ~/.bashrc
""",
        "zsh": """
# Add to ~/.zshrc:
eval "$(_GORGON_COMPLETE=zsh_source gorgon)"

# Or for faster startup, add to ~/.zshrc:
autoload -Uz compinit && compinit
eval "$(_GORGON_COMPLETE=zsh_source gorgon)"
""",
        "fish": """
# Add to ~/.config/fish/completions/gorgon.fish:
_GORGON_COMPLETE=fish_source gorgon > ~/.config/fish/completions/gorgon.fish
""",
    }

    console.print(
        Panel(
            f"[bold]Shell Completion Setup for {shell.upper()}[/bold]\n\n"
            f"{instructions.get(shell, instructions['bash'])}\n"
            "[dim]After adding, restart your shell or source the config file.[/dim]",
            title="Installation Instructions",
            border_style="cyan",
        )
    )


# ---------------------------------------------------------------------------
# Commands (registered by main.py)
# ---------------------------------------------------------------------------


def tui() -> None:
    """Launch the Gorgon TUI - unified AI terminal interface."""
    from animus_forge.tui.app import GorgonApp

    tui_app = GorgonApp()
    tui_app.run()


def init(
    name: str = typer.Argument(
        ...,
        help="Name for the new workflow",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: <name>.json)",
    ),
):
    """Create a new workflow template."""
    workflow_id = name.lower().replace(" ", "_").replace("-", "_")
    output_path = output or Path(f"{workflow_id}.json")

    if output_path.exists():
        overwrite = typer.confirm(f"{output_path} already exists. Overwrite?")
        if not overwrite:
            raise typer.Abort()

    template = {
        "id": workflow_id,
        "name": name,
        "description": f"Workflow: {name}",
        "variables": {"input": "default value"},
        "steps": [
            {
                "id": "step_1",
                "type": "transform",
                "action": "format",
                "params": {"template": "Processing: {{input}}"},
                "next_step": "step_2",
            },
            {
                "id": "step_2",
                "type": "claude_code",
                "action": "execute_agent",
                "params": {
                    "role": "assistant",
                    "task": "Analyze the input and provide insights",
                },
                "next_step": None,
            },
        ],
    }

    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)

    console.print(f"[green]✓ Created workflow template:[/green] {output_path}")
    console.print("\nNext steps:")
    console.print(f"  1. Edit {output_path} to customize your workflow")
    console.print(f"  2. Validate: [cyan]gorgon validate {output_path}[/cyan]")
    console.print(f"  3. Run: [cyan]gorgon run {output_path}[/cyan]")


def version_cmd() -> None:
    """Show Gorgon version (use --version instead)."""
    from animus_forge.cli.main import __version__

    console.print(f"[bold cyan]gorgon[/bold cyan] version {__version__}")
    console.print("[dim]Your personal army of AI agents[/dim]")


def completion(
    shell: str = typer.Argument(
        None,
        help="Shell type (bash, zsh, fish). Auto-detected if not provided.",
    ),
    install: bool = typer.Option(
        False,
        "--install",
        "-i",
        help="Install completion to shell config file.",
    ),
):
    """Show or install shell completion.

    Examples:

        # Show completion script for current shell
        gorgon completion

        # Show completion script for specific shell
        gorgon completion bash

        # Install completion (adds to shell config)
        gorgon completion --install
    """
    import os

    # Auto-detect shell
    if not shell:
        shell_path = os.environ.get("SHELL", "")
        if "zsh" in shell_path:
            shell = "zsh"
        elif "fish" in shell_path:
            shell = "fish"
        else:
            shell = "bash"

    if install:
        # Use Typer's built-in completion installation
        console.print(f"[cyan]Installing completion for {shell}...[/cyan]")
        try:
            # Typer installs completion via --install-completion
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "typer",
                    "gorgon",
                    "--install-completion",
                    shell,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                console.print(f"[green]Completion installed for {shell}![/green]")
                console.print("[dim]Restart your shell or source your config file.[/dim]")
            else:
                # Fall back to manual instructions
                _show_completion_instructions(shell)
        except Exception:
            _show_completion_instructions(shell)
    else:
        _show_completion_instructions(shell)
