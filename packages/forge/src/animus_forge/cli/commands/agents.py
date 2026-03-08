"""CLI commands for agent management.

Provides gorgon agent run/list/status/cancel/memory commands.
"""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.table import Table

agent_app = typer.Typer(help="Manage and run AI agents.")
console = Console()


@agent_app.command("run")
def agent_run(
    role: str = typer.Argument(help="Agent role (builder, tester, reviewer, planner, analyst)."),
    task: str = typer.Argument(help="Task description for the agent."),
    provider: str = typer.Option(
        "ollama", "--provider", "-p", help="Provider: ollama, anthropic, openai."
    ),
    tools: bool = typer.Option(True, "--tools/--no-tools", help="Enable/disable tool execution."),
    context: str = typer.Option("", "--context", "-c", help="Additional context for the agent."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Run an agent task and display the result."""
    from animus_forge.agents.provider_wrapper import create_agent_provider
    from animus_forge.agents.task_runner import AgentTaskRunner

    try:
        agent_provider = create_agent_provider(provider)
    except Exception as e:
        console.print(f"[red]Failed to create provider: {e}[/red]")
        raise typer.Exit(1)

    tool_registry = None
    if tools:
        try:
            from animus_forge.tools.registry import ForgeToolRegistry

            tool_registry = ForgeToolRegistry()
        except Exception:
            console.print(
                "[yellow]Warning: Tool registry unavailable, running without tools.[/yellow]"
            )

    runner = AgentTaskRunner(provider=agent_provider, tool_registry=tool_registry)

    with console.status(f"[bold green]Running {role} agent...[/bold green]"):
        result = asyncio.run(runner.run(role, task, use_tools=tools, context=context))

    if json_output:
        console.print_json(json.dumps(result.to_dict()))
        return

    if result.status == "completed":
        console.print(f"\n[bold green]{role}[/bold green] completed in {result.duration_ms}ms")
        if result.tool_calls > 0:
            console.print(f"[dim]Tool calls: {result.tool_calls}[/dim]")
        console.print()
        console.print(result.output)
    else:
        console.print(f"\n[bold red]{role}[/bold red] failed: {result.error}")
        raise typer.Exit(1)


@agent_app.command("list")
def agent_list(
    status: str | None = typer.Option(None, "--status", "-s", help="Filter by status."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List agent runs from the process registry."""
    try:
        from animus_forge.agents.subagent_manager import SubAgentManager

        sam = SubAgentManager()
        runs = sam.list_runs()
    except Exception:
        runs = []

    if not runs:
        console.print("[dim]No agent runs found.[/dim]")
        return

    if status:
        runs = [r for r in runs if r.status.value == status]

    if json_output:
        console.print_json(json.dumps([r.to_dict() for r in runs]))
        return

    table = Table(title="Agent Runs")
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Agent", style="green")
    table.add_column("Status", style="bold")
    table.add_column("Duration", justify="right")
    table.add_column("Task", max_width=40)

    for run in runs:
        status_style = {
            "completed": "[green]completed[/green]",
            "running": "[yellow]running[/yellow]",
            "failed": "[red]failed[/red]",
            "pending": "[dim]pending[/dim]",
            "cancelled": "[dim]cancelled[/dim]",
            "timed_out": "[red]timed_out[/red]",
        }.get(run.status.value, run.status.value)

        table.add_row(
            run.run_id[:16],
            run.agent,
            status_style,
            f"{run.duration_ms}ms",
            run.task[:40],
        )

    console.print(table)


@agent_app.command("status")
def agent_status(
    run_id: str = typer.Argument(help="Run ID to inspect."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show detailed status of an agent run."""
    try:
        from animus_forge.agents.subagent_manager import SubAgentManager

        sam = SubAgentManager()
        run = sam.get_run(run_id)
    except Exception:
        run = None

    if run is None:
        console.print(f"[red]Run {run_id} not found.[/red]")
        raise typer.Exit(1)

    if json_output:
        console.print_json(json.dumps(run.to_dict()))
        return

    console.print(f"[bold]Run ID:[/bold]  {run.run_id}")
    console.print(f"[bold]Agent:[/bold]   {run.agent}")
    console.print(f"[bold]Status:[/bold]  {run.status.value}")
    console.print(f"[bold]Task:[/bold]    {run.task[:100]}")
    console.print(f"[bold]Duration:[/bold] {run.duration_ms}ms")

    if run.result:
        console.print(f"\n[bold green]Result:[/bold green]\n{run.result[:500]}")
    if run.error:
        console.print(f"\n[bold red]Error:[/bold red] {run.error}")
    if run.children:
        console.print(f"\n[bold]Children:[/bold] {', '.join(run.children)}")


@agent_app.command("cancel")
def agent_cancel(
    run_id: str = typer.Argument(help="Run ID to cancel."),
    cascade: bool = typer.Option(True, "--cascade/--no-cascade", help="Cancel children too."),
) -> None:
    """Cancel a running agent."""
    try:
        from animus_forge.agents.subagent_manager import SubAgentManager

        sam = SubAgentManager()
        cancelled = asyncio.run(sam.cancel(run_id, cascade=cascade))
    except Exception as e:
        console.print(f"[red]Cancel failed: {e}[/red]")
        raise typer.Exit(1)

    if cancelled:
        console.print(f"[green]Cancelled run {run_id}[/green]")
    else:
        console.print(f"[yellow]Run {run_id} was not running.[/yellow]")


@agent_app.command("memory")
def agent_memory(
    agent_id: str = typer.Argument(help="Agent ID to query memory for."),
    memory_type: str | None = typer.Option(None, "--type", "-t", help="Filter by memory type."),
    limit: int = typer.Option(10, "--limit", "-n", help="Max entries to show."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show persistent memory for an agent."""
    try:
        from animus_forge.state.agent_memory import AgentMemory

        mem = AgentMemory()
        entries = mem.recall(agent_id, memory_type=memory_type, limit=limit)
    except Exception as e:
        console.print(f"[red]Memory query failed: {e}[/red]")
        raise typer.Exit(1)

    if not entries:
        console.print(f"[dim]No memories found for {agent_id}.[/dim]")
        return

    if json_output:
        console.print_json(
            json.dumps(
                [
                    {
                        "id": e.id,
                        "type": e.memory_type,
                        "content": e.content[:200],
                        "importance": e.importance,
                    }
                    for e in entries
                ]
            )
        )
        return

    table = Table(title=f"Memories for {agent_id}")
    table.add_column("ID", style="dim", width=5)
    table.add_column("Type", style="cyan")
    table.add_column("Importance", justify="right")
    table.add_column("Content", max_width=60)

    for entry in entries:
        table.add_row(
            str(entry.id),
            entry.memory_type,
            f"{entry.importance:.1f}",
            entry.content[:60],
        )

    console.print(table)
