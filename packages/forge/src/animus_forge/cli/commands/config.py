"""Config commands — view and manage configuration."""

from __future__ import annotations

import json

import typer
from rich.panel import Panel
from rich.table import Table

from ..helpers import console

config_app = typer.Typer(help="View and manage configuration")


@config_app.command("show")
def config_show(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Show current configuration."""
    try:
        from animus_forge.config import get_config

        config = get_config()
        config_dict = config.model_dump()
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        config_dict = {}

    # Mask sensitive values
    masked = {}
    for key, value in config_dict.items():
        if any(s in key.lower() for s in ["key", "secret", "password", "token"]):
            masked[key] = "****" if value else None
        else:
            masked[key] = value

    if json_output:
        print(json.dumps(masked, indent=2, default=str))
        return

    console.print(Panel("[bold]Animus Forge Configuration[/bold]", border_style="blue"))

    table = Table(show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    for key, value in sorted(masked.items()):
        display_value = str(value) if value is not None else "[dim]not set[/dim]"
        if display_value == "****":
            display_value = "[green]****[/green]"
        table.add_row(key, display_value)

    console.print(table)


@config_app.command("path")
def config_path() -> None:
    """Show configuration file paths."""
    import os
    from pathlib import Path

    from animus_forge.config.settings import _YAML_SEARCH_PATHS, _find_yaml_config

    console.print("[bold]Configuration Sources[/bold]\n")

    # animus-forge.yaml
    yaml_path = _find_yaml_config()
    if yaml_path:
        console.print(f"[green]✓[/green] animus-forge.yaml: {yaml_path.absolute()}")
    else:
        console.print("[yellow]○[/yellow] animus-forge.yaml: not found")

    # .env file
    env_path = Path(".env")
    if env_path.exists():
        console.print(f"[green]✓[/green] .env: {env_path.absolute()}")
    else:
        console.print("[yellow]○[/yellow] .env: not found")

    # Environment variables
    forge_vars = {k: v for k, v in os.environ.items() if k.startswith("FORGE_")}
    if forge_vars:
        console.print(f"\n[bold]Environment Variables ({len(forge_vars)}):[/bold]")
        for key in sorted(forge_vars.keys()):
            console.print(f"  {key}")
    else:
        console.print("\n[dim]No FORGE_* environment variables set[/dim]")

    # YAML search paths
    console.print("\n[bold]YAML Search Paths:[/bold]")
    for p in _YAML_SEARCH_PATHS:
        marker = "[green]✓[/green]" if p.is_file() else "[dim]○[/dim]"
        console.print(f"  {marker} {p}")


@config_app.command("env")
def config_env() -> None:
    """Show required environment variables."""
    env_vars = [
        ("ANTHROPIC_API_KEY", "Anthropic/Claude API key", True),
        ("OPENAI_API_KEY", "OpenAI API key", True),
        ("GITHUB_TOKEN", "GitHub personal access token", False),
        ("NOTION_TOKEN", "Notion API token", False),
        ("FORGE_LOG_LEVEL", "Log level (DEBUG, INFO, WARNING, ERROR)", False),
        ("FORGE_BUDGET_LIMIT", "Token budget limit", False),
        ("FORGE_WORKFLOWS_DIR", "Workflows directory path", False),
    ]

    import os

    console.print("[bold]Environment Variables[/bold]\n")

    table = Table()
    table.add_column("Variable", style="cyan")
    table.add_column("Description")
    table.add_column("Required")
    table.add_column("Status")

    for var, desc, required in env_vars:
        value = os.environ.get(var)
        if value:
            status = "[green]✓ set[/green]"
        elif required:
            status = "[red]✗ missing[/red]"
        else:
            status = "[dim]not set[/dim]"

        req = "[yellow]yes[/yellow]" if required else "no"
        table.add_row(var, desc, req, status)

    console.print(table)
