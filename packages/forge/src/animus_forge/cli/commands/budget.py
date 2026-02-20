"""Budget commands — view and manage token budgets."""

from __future__ import annotations

import json

import typer
from rich.panel import Panel
from rich.table import Table

from ..helpers import console

budget_app = typer.Typer(help="View budget and usage")


@budget_app.command("status")
def budget_status(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Show current budget status."""
    try:
        from animus_forge.budget import BudgetManager

        manager = BudgetManager()
        stats = manager.get_stats()
    except Exception as e:
        console.print(f"[red]Error getting budget:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(stats, indent=2, default=str))
        return

    used_pct = (stats["used"] / stats["total_budget"] * 100) if stats["total_budget"] > 0 else 0
    status_color = "green" if used_pct < 75 else "yellow" if used_pct < 90 else "red"

    console.print(
        Panel(
            f"Total Budget: [bold]{stats['total_budget']:,}[/bold] tokens\n"
            f"Used: [bold]{stats['used']:,}[/bold] tokens ([{status_color}]{used_pct:.1f}%[/{status_color}])\n"
            f"Remaining: [bold]{stats['remaining']:,}[/bold] tokens\n"
            f"Operations: [bold]{stats['total_operations']}[/bold]",
            title="Budget Status",
            border_style="blue",
        )
    )

    if stats.get("agents"):
        console.print("\n[dim]Usage by Agent:[/dim]")
        for agent_id, usage in stats["agents"].items():
            console.print(f"  {agent_id}: {usage:,} tokens")


@budget_app.command("history")
def budget_history(
    agent: str = typer.Option(None, "--agent", "-a", help="Filter by agent"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum entries"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Show budget usage history."""
    try:
        from animus_forge.budget import BudgetManager

        manager = BudgetManager()
        history = manager.get_usage_history(agent)[:limit]
    except Exception as e:
        console.print(f"[red]Error getting history:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps([h.__dict__ for h in history], indent=2, default=str))
        return

    if not history:
        console.print("[yellow]No usage history[/yellow]")
        return

    table = Table(title="Usage History")
    table.add_column("Time", style="dim")
    table.add_column("Agent", style="cyan")
    table.add_column("Tokens", justify="right")
    table.add_column("Operation")

    for record in history:
        time_str = str(record.timestamp)[:19] if record.timestamp else "-"
        table.add_row(
            time_str,
            record.agent_id,
            f"{record.tokens:,}",
            record.operation[:30] if record.operation else "-",
        )

    console.print(table)


@budget_app.command("daily")
def budget_daily(
    days: int = typer.Option(7, "--days", "-d", help="Number of days to show"),
    agent: str = typer.Option(None, "--agent", "-a", help="Filter by agent role"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Show daily token spend from task history."""
    try:
        from animus_forge.db import get_task_store

        rows = get_task_store().get_daily_budget(days=days, agent_role=agent)
    except Exception as e:
        console.print(f"[red]Error getting daily budget:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(rows, indent=2, default=str))
        return

    if not rows:
        console.print("[yellow]No daily budget data[/yellow]")
        return

    table = Table(title="Daily Token Spend")
    table.add_column("Date", style="dim")
    table.add_column("Agent", style="cyan")
    table.add_column("Tasks", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right", style="green")

    for row in rows:
        table.add_row(
            row.get("date", "-"),
            row.get("agent_role", "-"),
            str(row.get("task_count", 0)),
            f"{row.get('total_tokens', 0):,}",
            f"${row.get('total_cost_usd', 0):.4f}",
        )

    console.print(table)


@budget_app.command("reset")
def budget_reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Reset budget tracking."""
    if not force:
        if not typer.confirm("Reset all budget tracking? This cannot be undone."):
            raise typer.Abort()

    try:
        from animus_forge.budget import BudgetManager

        manager = BudgetManager()
        manager.reset()
        console.print("[green]✓ Budget tracking reset[/green]")
    except Exception as e:
        console.print(f"[red]Error resetting budget:[/red] {e}")
        raise typer.Exit(1)
