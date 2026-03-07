"""CLI commands for YAML workflow evolution fast path."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

evolve_app = typer.Typer(help="YAML workflow evolution — propose, approve, reject patches")
console = Console()


def _get_evolution():
    from animus_forge.config import get_settings
    from animus_forge.coordination.workflow_evolution import WorkflowEvolution

    settings = get_settings()
    return WorkflowEvolution(
        workflows_dir=settings.base_dir / "workflows",
        audit_log_path=settings.base_dir / "logs" / "forge_audit.jsonl",
    )


@evolve_app.command("status")
def status():
    """List pending workflow patches awaiting approval."""
    evo = _get_evolution()
    pending = evo.list_pending()
    if not pending:
        console.print("[dim]No pending workflow patches.[/dim]")
        return
    console.print(f"[bold]{len(pending)} pending patch(es):[/bold]")
    for wf_id in pending:
        console.print(f"  - {wf_id}.pending.yaml")
    console.print("\n[dim]Use 'animus evolve approve <id>' or 'animus evolve reject <id>'[/dim]")


@evolve_app.command("list")
def list_workflows():
    """List all workflows with version and evolution status."""
    evo = _get_evolution()
    workflows = evo.list_workflows()
    if not workflows:
        console.print("[dim]No workflows found.[/dim]")
        return

    table = Table(title="Workflows")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Steps", justify="right")
    table.add_column("Last Evolved", style="dim")
    table.add_column("Pending", style="yellow")

    for wf in workflows:
        table.add_row(
            wf.get("id", "?"),
            wf.get("name", "?"),
            wf.get("version", "?"),
            str(wf.get("steps", "?")),
            wf.get("last_evolved") or "never",
            "yes" if wf.get("has_pending") else "",
        )
    console.print(table)


@evolve_app.command("approve")
def approve(
    workflow_id: str = typer.Argument(help="Workflow ID to approve"),
):
    """Approve a pending workflow patch."""
    evo = _get_evolution()
    try:
        result = evo.approve(workflow_id)
        console.print(f"[green]Approved:[/green] {workflow_id} (v{result.workflow_id})")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@evolve_app.command("reject")
def reject(
    workflow_id: str = typer.Argument(help="Workflow ID to reject"),
    reason: str = typer.Argument(default="", help="Rejection reason"),
):
    """Reject a pending workflow patch."""
    evo = _get_evolution()
    evo.reject(workflow_id, reason)
    console.print(f"[yellow]Rejected:[/yellow] {workflow_id}")
    if reason:
        console.print(f"  Reason: {reason}")


@evolve_app.command("history")
def history(
    workflow_id: str = typer.Argument(help="Workflow ID to inspect"),
):
    """Show evolution notes for a workflow."""
    evo = _get_evolution()
    notes = evo.history(workflow_id)
    if not notes:
        console.print(f"[dim]No evolution history for {workflow_id}.[/dim]")
        return

    table = Table(title=f"Evolution History: {workflow_id}")
    table.add_column("Version", style="cyan")
    table.add_column("Date")
    table.add_column("Change")
    table.add_column("Proposed By", style="dim")

    for note in notes:
        table.add_row(
            str(note.get("version", "?")),
            str(note.get("date", "?")),
            note.get("change", "?"),
            note.get("proposed_by", "?"),
        )
    console.print(table)
