"""Admin commands — dashboard, plugins, logs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from ..helpers import console, get_tracker

# ---------------------------------------------------------------------------
# Dashboard command (registered by main.py)
# ---------------------------------------------------------------------------


def dashboard(
    port: int = typer.Option(8501, "--port", "-p", help="Port to run dashboard on"),
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser automatically"),
):
    """Launch the Gorgon web dashboard."""
    import webbrowser

    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"

    if not dashboard_path.exists():
        console.print(f"[red]Dashboard not found at:[/red] {dashboard_path}")
        raise typer.Exit(1)

    url = f"http://{host}:{port}"
    console.print("[cyan]Starting Gorgon Dashboard...[/cyan]")
    console.print(f"[bold]URL:[/bold] {url}")
    console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")

    if not no_browser:
        webbrowser.open(url)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(dashboard_path),
                "--server.port",
                str(port),
                "--server.address",
                host,
                "--server.headless",
                "true",
            ],
        )
        if result.returncode != 0:
            raise typer.Exit(result.returncode)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")


# ---------------------------------------------------------------------------
# Plugins sub-app
# ---------------------------------------------------------------------------

plugins_app = typer.Typer(help="Manage plugins")


@plugins_app.command("list")
def plugins_list(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """List installed plugins."""
    try:
        from animus_forge.plugins import PluginManager

        manager = PluginManager()
        plugins = manager.list_plugins()
    except Exception as e:
        console.print(f"[red]Error loading plugins:[/red] {e}")
        plugins = []

    if json_output:
        print(json.dumps([p.to_dict() for p in plugins], indent=2))
        return

    if not plugins:
        console.print("[yellow]No plugins installed[/yellow]")
        console.print("\n[dim]Plugins extend Gorgon with custom step types and integrations.[/dim]")
        return

    table = Table(title="Installed Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Description")
    table.add_column("Status")

    for plugin in plugins:
        status = "[green]active[/green]" if plugin.enabled else "[dim]disabled[/dim]"
        table.add_row(
            plugin.name,
            plugin.version,
            plugin.description[:40] if plugin.description else "-",
            status,
        )

    console.print(table)


@plugins_app.command("info")
def plugins_info(
    name: str = typer.Argument(..., help="Plugin name"),
):
    """Show detailed plugin information."""
    try:
        from animus_forge.plugins import PluginManager

        manager = PluginManager()
        plugin = manager.get_plugin(name)

        if not plugin:
            console.print(f"[red]Plugin not found:[/red] {name}")
            raise typer.Exit(1)

        console.print(
            Panel(
                f"[bold]{plugin.name}[/bold] v{plugin.version}\n\n"
                f"{plugin.description or 'No description'}\n\n"
                f"[dim]Author:[/dim] {plugin.author or 'Unknown'}\n"
                f"[dim]Status:[/dim] {'Enabled' if plugin.enabled else 'Disabled'}",
                title="Plugin Info",
                border_style="cyan",
            )
        )

        if plugin.step_types:
            console.print("\n[bold]Step Types:[/bold]")
            for step_type in plugin.step_types:
                console.print(f"  • {step_type}")

        if plugin.hooks:
            console.print("\n[bold]Hooks:[/bold]")
            for hook in plugin.hooks:
                console.print(f"  • {hook}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Logs command (registered by main.py)
# ---------------------------------------------------------------------------


def logs(
    workflow: str = typer.Option(None, "--workflow", "-w", help="Filter by workflow"),
    execution: str = typer.Option(None, "--execution", "-e", help="Filter by execution ID"),
    level: str = typer.Option(
        "INFO", "--level", "-l", help="Minimum log level (DEBUG, INFO, WARNING, ERROR)"
    ),
    tail: int = typer.Option(50, "--tail", "-n", help="Number of recent entries"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """View workflow execution logs."""
    import time

    try:
        tracker = get_tracker()
        if not tracker:
            console.print("[yellow]Tracker not available[/yellow]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error accessing logs:[/red] {e}")
        raise typer.Exit(1)

    def format_log_entry(entry: dict) -> str:
        """Format a single log entry."""
        ts = entry.get("timestamp", "")[:19]
        lvl = entry.get("level", "INFO")
        msg = entry.get("message", "")
        wf = entry.get("workflow_id", "")
        ex = entry.get("execution_id", "")[:8] if entry.get("execution_id") else ""

        level_colors = {
            "DEBUG": "dim",
            "INFO": "blue",
            "WARNING": "yellow",
            "ERROR": "red",
        }
        color = level_colors.get(lvl, "white")

        if wf:
            return f"[dim]{ts}[/dim] [{color}]{lvl:7}[/{color}] [{wf}:{ex}] {msg}"
        return f"[dim]{ts}[/dim] [{color}]{lvl:7}[/{color}] {msg}"

    def get_logs_data():
        """Fetch logs from tracker."""
        try:
            return tracker.get_logs(
                workflow_id=workflow,
                execution_id=execution,
                level=level,
                limit=tail,
            )
        except AttributeError:
            # Tracker doesn't have get_logs, use dashboard data
            data = tracker.get_dashboard_data()
            return data.get("recent_logs", [])

    logs_data = get_logs_data()

    if json_output:
        print(json.dumps(logs_data, indent=2, default=str))
        return

    if not logs_data:
        console.print("[yellow]No logs found[/yellow]")
        if not follow:
            return

    # Display existing logs
    for entry in logs_data:
        console.print(format_log_entry(entry))

    # Follow mode
    if follow:
        console.print("\n[dim]Following logs... (Ctrl+C to stop)[/dim]\n")
        seen = set(str(e) for e in logs_data)
        try:
            while True:
                time.sleep(2)
                new_logs = get_logs_data()
                for entry in new_logs:
                    entry_str = str(entry)
                    if entry_str not in seen:
                        seen.add(entry_str)
                        console.print(format_log_entry(entry))
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped following logs[/dim]")
