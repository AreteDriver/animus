"""History commands — view task history and agent performance."""

from __future__ import annotations

import typer
from rich.table import Table

from ..helpers import console

history_app = typer.Typer(help="View task history and analytics")


@history_app.command("list")
def history_list(
    status: str | None = typer.Option(None, "--status", "-s", help="Filter by status"),
    agent: str | None = typer.Option(None, "--agent", "-a", help="Filter by agent role"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of tasks to show"),
):
    """Show recent task history."""
    try:
        from animus_forge.db import get_task_store

        store = get_task_store()
        tasks = store.query_tasks(status=status, agent_role=agent, limit=limit)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not tasks:
        console.print("[dim]No tasks found.[/dim]")
        return

    table = Table(title="Task History")
    table.add_column("ID", style="dim")
    table.add_column("Workflow")
    table.add_column("Status")
    table.add_column("Agent")
    table.add_column("Duration")
    table.add_column("Cost")
    table.add_column("Completed")

    for t in tasks:
        dur = f"{t['duration_ms']}ms" if t.get("duration_ms") else "-"
        cost = f"${t['cost_usd']:.4f}" if t.get("cost_usd") else "-"
        status_style = "green" if t["status"] == "completed" else "red"
        table.add_row(
            str(t["id"]),
            t["workflow_id"][:20],
            f"[{status_style}]{t['status']}[/{status_style}]",
            t.get("agent_role") or "-",
            dur,
            cost,
            str(t.get("completed_at", ""))[:16],
        )

    console.print(table)


@history_app.command("stats")
def history_stats(
    agent: str | None = typer.Option(None, "--agent", "-a", help="Show stats for specific agent"),
):
    """Show agent performance statistics."""
    try:
        from animus_forge.db import get_task_store

        stats = get_task_store().get_agent_stats(agent_role=agent)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not stats:
        console.print("[dim]No agent stats found.[/dim]")
        return

    table = Table(title="Agent Performance")
    table.add_column("Agent")
    table.add_column("Tasks", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Avg Duration", justify="right")

    for s in stats:
        table.add_row(
            s["agent_role"],
            str(s["total_tasks"]),
            str(s["successful_tasks"]),
            str(s["failed_tasks"]),
            f"{s['success_rate']:.1f}%",
            f"{s['total_tokens']:,}",
            f"${s['total_cost_usd']:.4f}",
            f"{s['avg_duration_ms']:.0f}ms",
        )

    console.print(table)


@history_app.command("budget")
def history_budget(
    days: int = typer.Option(7, "--days", "-d", help="Number of days to show"),
):
    """Show daily budget breakdown."""
    try:
        from animus_forge.db import get_task_store

        logs = get_task_store().get_daily_budget(days=days)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not logs:
        console.print("[dim]No budget data found.[/dim]")
        return

    table = Table(title=f"Budget — Last {days} Days")
    table.add_column("Date")
    table.add_column("Agent")
    table.add_column("Tasks", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")

    for log in logs:
        table.add_row(
            log["date"],
            log.get("agent_role") or "-",
            str(log["task_count"]),
            f"{log['total_tokens']:,}",
            f"${log['total_cost_usd']:.4f}",
        )

    console.print(table)
