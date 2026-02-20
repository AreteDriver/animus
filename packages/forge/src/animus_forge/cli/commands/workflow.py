"""Workflow commands — run, list, validate, status."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..helpers import _parse_cli_variables, console, get_tracker, get_workflow_engine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def list_workflows_table(engine) -> None:
    """Display workflows in a table."""
    workflows = engine.list_workflows()

    table = Table(title="Available Workflows")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Steps", justify="right")

    for wf in workflows:
        loaded = engine.load_workflow(wf["id"])
        steps = len(loaded.steps) if loaded else "?"
        table.add_row(
            wf["id"],
            wf.get("name", "-"),
            wf.get("description", "-")[:50],
            str(steps),
        )

    console.print(table)


def _load_workflow_from_source(workflow: str, engine) -> tuple[str, dict, Path | None]:
    """Load workflow from file or by ID.

    Returns:
        Tuple of (workflow_id, workflow_data, workflow_path_or_None)
    """
    workflow_path = Path(workflow)
    if workflow_path.exists() and workflow_path.suffix == ".json":
        try:
            with open(workflow_path) as f:
                workflow_data = json.load(f)
            workflow_id = workflow_data.get("id", workflow_path.stem)
            console.print(f"[dim]Loading workflow from:[/dim] {workflow_path}")
            return workflow_id, workflow_data, workflow_path
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in workflow file:[/red] {e}")
            raise typer.Exit(1)

    loaded = engine.load_workflow(workflow)
    if not loaded:
        console.print(f"[red]Workflow not found:[/red] {workflow}")
        console.print("\nAvailable workflows:")
        list_workflows_table(engine)
        raise typer.Exit(1)
    return workflow, loaded.model_dump(), None


def _display_workflow_preview(workflow_id: str, workflow_data: dict, variables: dict) -> None:
    """Display workflow information preview."""
    console.print(
        Panel(
            f"[bold]{workflow_data.get('name', workflow_id)}[/bold]\n"
            f"[dim]{workflow_data.get('description', 'No description')}[/dim]",
            title="Workflow",
            border_style="blue",
        )
    )

    if workflow_data.get("steps"):
        console.print(f"\n[dim]Steps:[/dim] {len(workflow_data['steps'])}")
        for step in workflow_data["steps"]:
            console.print(f"  • {step['id']} ({step['type']}:{step.get('action', 'N/A')})")

    if variables:
        console.print("\n[dim]Variables:[/dim]")
        for k, v in variables.items():
            console.print(f"  {k} = {v}")


def _output_run_results(result, json_output: bool) -> None:
    """Output workflow execution results."""
    if json_output:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
        return

    status_color = "green" if result.status == "completed" else "red"
    console.print(f"\n[{status_color}]Status: {result.status}[/{status_color}]")

    if result.step_results:
        console.print("\n[dim]Step Results:[/dim]")
        for step_id, step_result in result.step_results.items():
            status = step_result.get("status", "unknown")
            icon = "✓" if status == "success" else "✗"
            console.print(f"  {icon} {step_id}: {status}")

    if result.error:
        console.print(f"\n[red]Error:[/red] {result.error}")


def _validate_cli_required_fields(data: dict) -> tuple[list[str], list[str]]:
    """Validate required workflow fields for CLI."""
    errors = []
    warnings = []
    if "id" not in data:
        errors.append("Missing required field: id")
    if "steps" not in data:
        errors.append("Missing required field: steps")
    elif not isinstance(data["steps"], list):
        errors.append("'steps' must be a list")
    elif len(data["steps"]) == 0:
        warnings.append("Workflow has no steps")
    return errors, warnings


def _validate_cli_steps(steps: list) -> tuple[list[str], list[str], set[str]]:
    """Validate workflow steps for CLI."""
    errors = []
    warnings = []
    step_ids: set[str] = set()
    valid_types = {"claude_code", "openai", "transform", "condition"}

    for i, step in enumerate(steps):
        prefix = f"Step {i + 1}"

        if "id" not in step:
            errors.append(f"{prefix}: Missing 'id'")
        else:
            if step["id"] in step_ids:
                errors.append(f"{prefix}: Duplicate step ID '{step['id']}'")
            step_ids.add(step["id"])

        if "type" not in step:
            errors.append(f"{prefix}: Missing 'type'")
        elif step["type"] not in valid_types:
            warnings.append(f"{prefix}: Unknown step type '{step['type']}'")

        if "action" not in step:
            errors.append(f"{prefix}: Missing 'action'")

    return errors, warnings, step_ids


def _validate_cli_next_step_refs(steps: list, step_ids: set[str]) -> list[str]:
    """Validate next_step references in workflow."""
    errors = []
    for step in steps:
        if "next_step" in step and step["next_step"]:
            if step["next_step"] not in step_ids:
                errors.append(
                    f"Step '{step.get('id', '?')}': next_step '{step['next_step']}' not found"
                )
    return errors


def _output_validation_results(errors: list[str], warnings: list[str], workflow_file: Path) -> None:
    """Output validation results to console."""
    if errors:
        console.print(
            Panel(
                "\n".join(f"[red]✗[/red] {e}" for e in errors),
                title="Errors",
                border_style="red",
            )
        )

    if warnings:
        console.print(
            Panel(
                "\n".join(f"[yellow]![/yellow] {w}" for w in warnings),
                title="Warnings",
                border_style="yellow",
            )
        )

    if not errors and not warnings:
        console.print(f"[green]✓ Workflow is valid:[/green] {workflow_file}")
    elif not errors:
        console.print("\n[green]✓ Workflow is valid with warnings[/green]")
    else:
        console.print("\n[red]✗ Validation failed[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Commands (registered by main.py)
# ---------------------------------------------------------------------------


def run(
    workflow: str = typer.Argument(
        ...,
        help="Workflow ID or path to workflow JSON file",
    ),
    var: list[str] = typer.Option(
        [],
        "--var",
        "-v",
        help="Variables in key=value format (can be repeated)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and show workflow without executing",
    ),
):
    """Run a workflow by ID or from a JSON file."""
    engine = get_workflow_engine()
    variables = _parse_cli_variables(var)
    workflow_id, workflow_data, workflow_path = _load_workflow_from_source(workflow, engine)

    _display_workflow_preview(workflow_id, workflow_data, variables)

    if dry_run:
        console.print("\n[yellow]Dry run - workflow not executed[/yellow]")
        raise typer.Exit(0)

    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Executing workflow...", total=None)

        try:
            if workflow_path is None:
                wf = engine.load_workflow(workflow_id)
                wf.variables = variables
            else:
                from animus_forge.orchestrator import Workflow

                wf = Workflow(**workflow_data)
                wf.variables = variables

            result = engine.execute_workflow(wf)
            progress.update(task, description="Complete!")
        except Exception as e:
            progress.stop()
            console.print(f"\n[red]Execution failed:[/red] {e}")
            raise typer.Exit(1)

    _output_run_results(result, json_output)


def list_workflows(
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """List all available workflows."""
    engine = get_workflow_engine()
    workflows = engine.list_workflows()

    if json_output:
        print(json.dumps(workflows, indent=2))
        return

    if not workflows:
        console.print("[yellow]No workflows found[/yellow]")
        console.print(
            "\nCreate workflows in the workflows directory or use 'gorgon run <file.json>'"
        )
        return

    list_workflows_table(engine)


def validate(
    workflow_file: Path = typer.Argument(
        ...,
        help="Path to workflow JSON file",
        exists=True,
    ),
):
    """Validate a workflow JSON file."""
    try:
        with open(workflow_file) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON:[/red] {e}")
        raise typer.Exit(1)

    errors, warnings = _validate_cli_required_fields(data)

    steps = data.get("steps", [])
    if isinstance(steps, list):
        step_errors, step_warnings, step_ids = _validate_cli_steps(steps)
        errors.extend(step_errors)
        warnings.extend(step_warnings)
        errors.extend(_validate_cli_next_step_refs(steps, step_ids))

    _output_validation_results(errors, warnings, workflow_file)


def status(
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """Show orchestrator status and metrics."""
    try:
        tracker = get_tracker()
        data = tracker.get_dashboard_data()
    except Exception as e:
        console.print(f"[yellow]Metrics unavailable:[/yellow] {e}")
        data = {"summary": {}, "active_workflows": [], "recent_executions": []}

    if json_output:
        print(json.dumps(data, indent=2, default=str))
        return

    summary = data.get("summary", {})

    # Summary panel
    console.print(
        Panel(
            f"Active Workflows: [bold]{summary.get('active_workflows', 0)}[/bold]\n"
            f"Total Executions: [bold]{summary.get('total_executions', 0)}[/bold]\n"
            f"Success Rate: [bold]{summary.get('success_rate', 0):.1f}%[/bold]\n"
            f"Avg Duration: [bold]{summary.get('avg_duration_ms', 0):.0f}ms[/bold]",
            title="Gorgon Status",
            border_style="blue",
        )
    )

    # Active workflows
    active = data.get("active_workflows", [])
    if active:
        console.print("\n[bold]Active Workflows:[/bold]")
        for wf in active:
            progress = wf["completed_steps"] / wf["total_steps"] if wf["total_steps"] > 0 else 0
            console.print(
                f"  • {wf['workflow_name']} ({wf['execution_id'][:12]}...) "
                f"[dim]{wf['completed_steps']}/{wf['total_steps']} steps ({progress * 100:.0f}%)[/dim]"
            )

    # Recent executions
    recent = data.get("recent_executions", [])[:5]
    if recent:
        console.print("\n[bold]Recent Executions:[/bold]")
        table = Table(show_header=True, header_style="dim")
        table.add_column("Workflow")
        table.add_column("Status")
        table.add_column("Duration")
        table.add_column("Steps")

        for ex in recent:
            status_style = "green" if ex["status"] == "completed" else "red"
            table.add_row(
                ex["workflow_name"][:30],
                f"[{status_style}]{ex['status']}[/{status_style}]",
                f"{ex['duration_ms']:.0f}ms",
                f"{ex['completed_steps']}/{ex['total_steps']}",
            )

        console.print(table)
