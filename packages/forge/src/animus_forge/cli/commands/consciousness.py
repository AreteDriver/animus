"""CLI commands for the consciousness-quorum bridge."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

consciousness_app = typer.Typer(help="Consciousness bridge — reflection & coordination")
console = Console()


def _get_bridge():
    """Lazy-load a ConsciousnessBridge for CLI use."""
    from animus_forge.budget.manager import BudgetManager
    from animus_forge.coordination.consciousness_bridge import (
        ConsciousnessBridge,
        ConsciousnessConfig,
    )

    try:
        from animus_forge.agents import create_agent_provider

        provider = create_agent_provider("ollama")
    except Exception:
        try:
            from animus_forge.agents import create_agent_provider

            provider = create_agent_provider("anthropic")
        except Exception as e:
            console.print(f"[red]No LLM provider available:[/red] {e}")
            raise typer.Exit(1) from e

    from animus_forge.config import get_settings

    settings = get_settings()
    config = ConsciousnessConfig(
        enabled=True,
        min_idle_seconds=0,
        reflections_log_path=settings.base_dir / "logs" / "reflections.jsonl",
        review_queue_path=settings.base_dir / "logs" / "workflow_review_queue.jsonl",
    )
    return ConsciousnessBridge(
        provider=provider,
        budget_manager=BudgetManager(),
        config=config,
    )


@consciousness_app.command("status")
def status():
    """Show consciousness bridge status."""
    try:
        from animus_forge import api_state as state

        if state.consciousness_bridge is not None:
            data = state.consciousness_bridge.status()
        else:
            data = {
                "running": False,
                "enabled": False,
                "last_reflection": None,
                "reflection_count": 0,
                "total_tokens": 0,
                "min_idle_seconds": 300,
                "model": "N/A",
            }
    except Exception:
        data = {
            "running": False,
            "enabled": False,
            "last_reflection": None,
            "reflection_count": 0,
            "total_tokens": 0,
            "min_idle_seconds": 300,
            "model": "N/A",
        }

    table = Table(title="Consciousness Bridge")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key, value in data.items():
        table.add_row(key, str(value))
    console.print(table)


@consciousness_app.command("reflect")
def reflect():
    """Trigger a single reflection cycle immediately."""
    bridge = _get_bridge()
    console.print("[dim]Running reflection cycle...[/dim]")
    try:
        output = bridge.reflect_once()
    except Exception as e:
        console.print(f"[red]Reflection failed:[/red] {e}")
        raise typer.Exit(1) from e

    console.print(f"\n[bold]Summary:[/bold] {output.summary}")
    if output.insights:
        console.print("\n[bold]Insights:[/bold]")
        for insight in output.insights:
            console.print(f"  - {insight}")
    if output.principle_tensions:
        console.print("\n[bold yellow]Principle Tensions:[/bold yellow]")
        for tension in output.principle_tensions:
            console.print(f"  - {tension}")
    if output.workflow_patch_ids:
        console.print("\n[bold]Workflows flagged for review:[/bold]")
        for wf_id in output.workflow_patch_ids:
            console.print(f"  - {wf_id}")
    console.print(f"\n[dim]Next reflection suggested in {output.next_reflection_in}s[/dim]")


@consciousness_app.command("history")
def history(
    last: int = typer.Option(10, "--last", "-n", help="Number of records to show"),
):
    """Show recent reflection records."""
    from animus_forge.config import get_settings

    log_path = get_settings().base_dir / "logs" / "reflections.jsonl"
    if not log_path.exists():
        console.print("[dim]No reflections recorded yet.[/dim]")
        raise typer.Exit(0)

    lines = log_path.read_text().strip().splitlines()
    recent = lines[-last:] if len(lines) > last else lines

    table = Table(title=f"Last {len(recent)} Reflections")
    table.add_column("Time", style="dim")
    table.add_column("Model")
    table.add_column("Tokens", justify="right")
    table.add_column("Summary")

    for line in recent:
        try:
            record = json.loads(line)
            summary = record.get("output", {}).get("summary", "")
            if len(summary) > 80:
                summary = summary[:77] + "..."
            table.add_row(
                record.get("timestamp", "?")[:19],
                record.get("model", "?"),
                str(record.get("tokens_used", 0)),
                summary,
            )
        except json.JSONDecodeError:
            continue

    console.print(table)


@consciousness_app.command("reviews")
def reviews():
    """Show pending workflow review queue from reflections."""
    from animus_forge.config import get_settings

    queue_path = get_settings().base_dir / "logs" / "workflow_review_queue.jsonl"
    if not queue_path.exists():
        console.print("[dim]No workflow reviews pending.[/dim]")
        raise typer.Exit(0)

    lines = queue_path.read_text().strip().splitlines()
    if not lines:
        console.print("[dim]No workflow reviews pending.[/dim]")
        raise typer.Exit(0)

    table = Table(title="Pending Workflow Reviews")
    table.add_column("Workflow", style="cyan")
    table.add_column("Flagged By")
    table.add_column("Time", style="dim")
    table.add_column("Cycle", justify="right")

    for line in lines:
        try:
            record = json.loads(line)
            table.add_row(
                record.get("workflow_id", "?"),
                record.get("flagged_by", "?"),
                record.get("timestamp", "?")[:19],
                str(record.get("cycle", 0)),
            )
        except json.JSONDecodeError:
            continue

    console.print(table)
