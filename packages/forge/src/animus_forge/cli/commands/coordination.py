"""Coordination commands — health, cycles, events from Convergent Phase 4."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..helpers import console

coordination_app = typer.Typer(help="Convergent coordination tools")


def _default_db_path() -> str:
    """Return the default coordination database path."""
    return str(Path.home() / ".gorgon" / "coordination.db")


def _default_events_db_path() -> str:
    """Return the default events database path."""
    return str(Path.home() / ".gorgon" / "coordination.events.db")


@coordination_app.command("health")
def health(
    db_path: Annotated[
        str,
        typer.Option("--db", help="Path to coordination database"),
    ] = "",
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON"),
    ] = False,
) -> None:
    """Show coordination health report."""
    import json as json_mod

    try:
        from animus_forge.agents.convergence import HAS_CONVERGENT

        if not HAS_CONVERGENT:
            console.print("[yellow]Convergent not installed[/yellow]")
            raise typer.Exit(1)

        from animus_forge.agents.convergence import create_bridge, get_coordination_health

        path = db_path or _default_db_path()
        if not Path(path).exists():
            console.print(f"[yellow]No coordination database found at {path}[/yellow]")
            raise typer.Exit(1)

        bridge = create_bridge(db_path=path)
        if bridge is None:
            console.print("[red]Failed to create coordination bridge[/red]")
            raise typer.Exit(1)

        try:
            health_data = get_coordination_health(bridge)
        finally:
            bridge.close()

        if not health_data:
            console.print("[yellow]No health data available[/yellow]")
            return

        if json_output:
            print(json_mod.dumps(health_data, indent=2, default=str))
            return

        # Reconstruct dataclass for the report renderer
        from dataclasses import fields

        from convergent import CoordinationHealth, health_report

        kwargs = {}
        for f in fields(CoordinationHealth):
            if f.name in health_data:
                kwargs[f.name] = health_data[f.name]
        console.print(health_report(CoordinationHealth(**kwargs)))

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@coordination_app.command("cycles")
def cycles(
    db_path: Annotated[
        str,
        typer.Option("--db", help="Path to coordination database"),
    ] = "",
) -> None:
    """Check for dependency cycles in the intent graph."""
    try:
        from animus_forge.agents.convergence import HAS_CONVERGENT

        if not HAS_CONVERGENT:
            console.print("[yellow]Convergent not installed[/yellow]")
            raise typer.Exit(1)

        from convergent import IntentResolver, SQLiteBackend

        path = db_path or _default_db_path()
        if not Path(path).exists():
            console.print(f"[yellow]No coordination database found at {path}[/yellow]")
            raise typer.Exit(1)

        from animus_forge.agents.convergence import check_dependency_cycles

        resolver = IntentResolver(backend=SQLiteBackend(path))
        found_cycles = check_dependency_cycles(resolver)

        if not found_cycles:
            console.print("[green]No dependency cycles detected[/green]")
            return

        console.print(f"[red]Found {len(found_cycles)} cycle(s):[/red]")
        for c in found_cycles:
            console.print(f"  - {c['display']}")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@coordination_app.command("events")
def events(
    db_path: Annotated[
        str,
        typer.Option("--db", help="Path to events database"),
    ] = "",
    event_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Filter by event type"),
    ] = None,
    agent: Annotated[
        str | None,
        typer.Option("--agent", "-a", help="Filter by agent ID"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Max events to display"),
    ] = 20,
) -> None:
    """Show coordination event timeline."""
    try:
        from animus_forge.agents.convergence import HAS_CONVERGENT

        if not HAS_CONVERGENT:
            console.print("[yellow]Convergent not installed[/yellow]")
            raise typer.Exit(1)

        from convergent import EventLog, EventType, event_timeline

        path = db_path or _default_events_db_path()
        if not Path(path).exists():
            console.print(f"[yellow]No events database found at {path}[/yellow]")
            raise typer.Exit(1)

        log = EventLog(path)

        et = None
        if event_type is not None:
            try:
                et = EventType(event_type)
            except ValueError:
                valid = ", ".join(e.value for e in EventType)
                console.print(f"[red]Unknown event type: {event_type}[/red]\nValid types: {valid}")
                raise typer.Exit(1)

        results = log.query(event_type=et, agent_id=agent, limit=limit)
        log.close()

        if not results:
            console.print("[dim]No events found[/dim]")
            return

        console.print(event_timeline(results))

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
