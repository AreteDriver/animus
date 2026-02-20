"""Memory commands — manage agent memory."""

from __future__ import annotations

import json

import typer
from rich.panel import Panel
from rich.table import Table

from ..helpers import console

memory_app = typer.Typer(help="Manage agent memory")


@memory_app.command("list")
def memory_list(
    agent: str = typer.Option(None, "--agent", "-a", help="Filter by agent ID"),
    memory_type: str = typer.Option(None, "--type", "-t", help="Filter by type"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum entries"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """List agent memories."""
    try:
        from animus_forge.state import AgentMemory

        memory = AgentMemory()

        if agent:
            memories = memory.recall(agent, memory_type=memory_type, limit=limit)
        else:
            # Get all agents' memories
            memories = memory.backend.fetchall(
                "SELECT * FROM agent_memories ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            from animus_forge.state.memory import MemoryEntry

            memories = [MemoryEntry.from_dict(m) for m in memories]
    except Exception as e:
        console.print(f"[red]Error loading memories:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps([m.to_dict() for m in memories], indent=2, default=str))
        return

    if not memories:
        console.print("[yellow]No memories found[/yellow]")
        return

    table = Table(title="Agent Memories")
    table.add_column("ID", style="dim")
    table.add_column("Agent", style="cyan")
    table.add_column("Type")
    table.add_column("Content")
    table.add_column("Importance", justify="right")

    for m in memories:
        content = m.content[:50] + "..." if len(m.content) > 50 else m.content
        table.add_row(
            str(m.id),
            m.agent_id,
            m.memory_type,
            content,
            f"{m.importance:.2f}",
        )

    console.print(table)


@memory_app.command("stats")
def memory_stats(
    agent: str = typer.Argument(..., help="Agent ID"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Show memory statistics for an agent."""
    try:
        from animus_forge.state import AgentMemory

        memory = AgentMemory()
        stats = memory.get_stats(agent)
    except Exception as e:
        console.print(f"[red]Error getting stats:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(stats, indent=2))
        return

    console.print(
        Panel(
            f"Total Memories: [bold]{stats['total_memories']}[/bold]\n"
            f"Average Importance: [bold]{stats['average_importance']:.2f}[/bold]",
            title=f"Memory Stats: {agent}",
            border_style="blue",
        )
    )

    if stats["by_type"]:
        console.print("\n[dim]By Type:[/dim]")
        for mtype, count in stats["by_type"].items():
            console.print(f"  {mtype}: {count}")


@memory_app.command("clear")
def memory_clear(
    agent: str = typer.Argument(..., help="Agent ID"),
    memory_type: str = typer.Option(None, "--type", "-t", help="Only clear this type"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Clear agent memories."""
    if not force:
        msg = f"Clear all memories for agent '{agent}'"
        if memory_type:
            msg += f" of type '{memory_type}'"
        if not typer.confirm(f"{msg}?"):
            raise typer.Abort()

    try:
        from animus_forge.state import AgentMemory

        memory = AgentMemory()
        count = memory.forget(agent, memory_type=memory_type)
        console.print(f"[green]✓ Cleared {count} memories[/green]")
    except Exception as e:
        console.print(f"[red]Error clearing memories:[/red] {e}")
        raise typer.Exit(1)
