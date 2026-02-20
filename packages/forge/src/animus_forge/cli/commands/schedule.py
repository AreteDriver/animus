"""Schedule commands — manage scheduled workflows."""

from __future__ import annotations

import json

import typer
from rich.table import Table

from ..helpers import console

schedule_app = typer.Typer(help="Manage scheduled workflows")


@schedule_app.command("list")
def schedule_list(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """List all scheduled workflows."""
    try:
        from animus_forge.workflow import WorkflowScheduler

        scheduler = WorkflowScheduler()
        schedules = scheduler.list()
    except Exception as e:
        console.print(f"[red]Error loading schedules:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps([s.__dict__ for s in schedules], indent=2, default=str))
        return

    if not schedules:
        console.print("[yellow]No scheduled workflows[/yellow]")
        return

    table = Table(title="Scheduled Workflows")
    table.add_column("ID", style="cyan")
    table.add_column("Workflow")
    table.add_column("Schedule")
    table.add_column("Status")
    table.add_column("Next Run")

    for s in schedules:
        schedule_str = s.cron_expression if s.cron_expression else f"every {s.interval_seconds}s"
        status_color = "green" if s.status.value == "active" else "yellow"
        next_run = str(s.next_run_time)[:19] if s.next_run_time else "-"
        table.add_row(
            s.schedule_id[:12],
            s.workflow_path,
            schedule_str,
            f"[{status_color}]{s.status.value}[/{status_color}]",
            next_run,
        )

    console.print(table)


@schedule_app.command("add")
def schedule_add(
    workflow: str = typer.Argument(..., help="Workflow path or ID"),
    cron: str = typer.Option(None, "--cron", "-c", help="Cron expression"),
    interval: int = typer.Option(None, "--interval", "-i", help="Interval in seconds"),
    name: str = typer.Option(None, "--name", "-n", help="Schedule name"),
):
    """Add a new scheduled workflow."""
    if not cron and not interval:
        console.print("[red]Must specify --cron or --interval[/red]")
        raise typer.Exit(1)

    try:
        from animus_forge.workflow import ScheduleConfig, WorkflowScheduler

        scheduler = WorkflowScheduler()

        config = ScheduleConfig(
            workflow_path=workflow,
            name=name or f"Schedule for {workflow}",
            cron_expression=cron,
            interval_seconds=interval,
        )
        result = scheduler.add(config)
        scheduler.start()

        console.print(f"[green]✓ Schedule created:[/green] {result.schedule_id}")
        if cron:
            console.print(f"  Cron: {cron}")
        else:
            console.print(f"  Interval: {interval}s")
    except Exception as e:
        console.print(f"[red]Error creating schedule:[/red] {e}")
        raise typer.Exit(1)


@schedule_app.command("remove")
def schedule_remove(
    schedule_id: str = typer.Argument(..., help="Schedule ID to remove"),
):
    """Remove a scheduled workflow."""
    try:
        from animus_forge.workflow import WorkflowScheduler

        scheduler = WorkflowScheduler()

        if scheduler.remove(schedule_id):
            console.print(f"[green]✓ Schedule removed:[/green] {schedule_id}")
        else:
            console.print(f"[red]Schedule not found:[/red] {schedule_id}")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error removing schedule:[/red] {e}")
        raise typer.Exit(1)


@schedule_app.command("pause")
def schedule_pause(
    schedule_id: str = typer.Argument(..., help="Schedule ID to pause"),
):
    """Pause a scheduled workflow."""
    try:
        from animus_forge.workflow import WorkflowScheduler

        scheduler = WorkflowScheduler()

        if scheduler.pause(schedule_id):
            console.print(f"[green]✓ Schedule paused:[/green] {schedule_id}")
        else:
            console.print(f"[red]Schedule not found:[/red] {schedule_id}")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error pausing schedule:[/red] {e}")
        raise typer.Exit(1)


@schedule_app.command("resume")
def schedule_resume(
    schedule_id: str = typer.Argument(..., help="Schedule ID to resume"),
):
    """Resume a paused scheduled workflow."""
    try:
        from animus_forge.workflow import WorkflowScheduler

        scheduler = WorkflowScheduler()

        if scheduler.resume(schedule_id):
            console.print(f"[green]✓ Schedule resumed:[/green] {schedule_id}")
        else:
            console.print(f"[red]Schedule not found:[/red] {schedule_id}")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error resuming schedule:[/red] {e}")
        raise typer.Exit(1)
