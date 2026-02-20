"""Graph commands -- execute and validate visual workflow graphs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from ..helpers import console

graph_app = typer.Typer(help="Execute and validate graph workflows")


def _load_graph_file(file_path: str) -> dict:
    """Load a graph definition from a JSON or YAML file.

    Args:
        file_path: Path to the graph file.

    Returns:
        Parsed graph dictionary.

    Raises:
        typer.Exit: If the file cannot be loaded.
    """
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {file_path}")
        raise typer.Exit(1)

    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            return yaml.safe_load(content)
        except ImportError:
            console.print("[red]PyYAML is required for YAML files.[/red]")
            console.print("Install it: pip install pyyaml")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error parsing YAML:[/red] {e}")
            raise typer.Exit(1)
    else:
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing JSON:[/red] {e}")
            raise typer.Exit(1)


@graph_app.command("execute")
def graph_execute(
    file: str = typer.Argument(..., help="Path to graph JSON/YAML file"),
    var: list[str] | None = typer.Option(None, "--var", "-v", help="Variables in key=value format"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Validate without executing"),
):
    """Execute a graph workflow from a JSON or YAML file."""
    graph_data = _load_graph_file(file)

    # Parse CLI variables
    variables: dict = {}
    if var:
        for v in var:
            if "=" in v:
                key, value = v.split("=", 1)
                variables[key] = value
            else:
                console.print(f"[red]Invalid variable format: {v}[/red]")
                console.print("Use: --var key=value")
                raise typer.Exit(1)

    if dry_run:
        console.print("[yellow]Dry run -- validating only[/yellow]")
        _run_validation(graph_data, json_output)
        return

    try:
        from animus_forge.workflow.graph_executor import ReactFlowExecutor
        from animus_forge.workflow.graph_models import WorkflowGraph

        graph = WorkflowGraph.from_dict(graph_data)
        executor = ReactFlowExecutor()

        result = asyncio.run(executor.execute_async(graph, variables))
    except Exception as e:
        console.print(f"[red]Execution failed:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        out = {
            "execution_id": result.execution_id,
            "workflow_id": result.workflow_id,
            "status": result.status,
            "outputs": result.outputs,
            "total_duration_ms": result.total_duration_ms,
            "total_tokens": result.total_tokens,
            "error": result.error,
            "node_results": {
                nid: {
                    "status": nr.status.value if hasattr(nr.status, "value") else str(nr.status),
                    "duration_ms": nr.duration_ms,
                    "error": nr.error,
                }
                for nid, nr in result.node_results.items()
            },
        }
        print(json.dumps(out, indent=2, default=str))
        return

    # Display results using Rich
    status_color = "green" if result.status == "completed" else "red"
    console.print(
        Panel(
            f"Execution ID: [bold]{result.execution_id}[/bold]\n"
            f"Status: [{status_color}]{result.status}[/{status_color}]\n"
            f"Duration: [bold]{result.total_duration_ms}ms[/bold]\n"
            f"Total Tokens: [bold]{result.total_tokens:,}[/bold]"
            + (f"\nError: [red]{result.error}[/red]" if result.error else ""),
            title=f"Graph Execution: {result.workflow_id}",
            border_style="blue",
        )
    )

    if result.node_results:
        table = Table(title="Node Results")
        table.add_column("Node", style="cyan")
        table.add_column("Status")
        table.add_column("Duration", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Error")

        for nid, nr in result.node_results.items():
            status_str = nr.status.value if hasattr(nr.status, "value") else str(nr.status)
            s_color = "green" if status_str == "completed" else "red"
            table.add_row(
                nid,
                f"[{s_color}]{status_str}[/{s_color}]",
                f"{nr.duration_ms}ms",
                str(nr.tokens_used),
                nr.error or "",
            )

        console.print(table)

    if result.outputs:
        console.print("\n[dim]Outputs:[/dim]")
        for key, value in result.outputs.items():
            console.print(f"  {key}: {value}")


@graph_app.command("validate")
def graph_validate(
    file: str = typer.Argument(..., help="Path to graph JSON/YAML file"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Validate a graph workflow for structural issues."""
    graph_data = _load_graph_file(file)
    _run_validation(graph_data, json_output)


def _run_validation(graph_data: dict, json_output: bool) -> None:
    """Shared validation logic for validate command and dry-run."""
    issues: list[dict] = []

    try:
        from animus_forge.workflow.graph_models import WorkflowGraph
        from animus_forge.workflow.graph_walker import GraphWalker

        graph = WorkflowGraph.from_dict(graph_data)
    except Exception as e:
        if json_output:
            print(
                json.dumps(
                    {
                        "valid": False,
                        "issues": [{"severity": "error", "message": f"Parse error: {e}"}],
                    },
                    indent=2,
                )
            )
        else:
            console.print(f"[red]Invalid graph:[/red] {e}")
        raise typer.Exit(1)

    walker = GraphWalker(graph)

    # Cycle detection
    cycles = walker.detect_cycles()
    for cycle in cycles:
        cycle_nodes = [graph.get_node(nid) for nid in cycle]
        if not any(n and n.type == "loop" for n in cycle_nodes):
            issues.append(
                {
                    "severity": "error",
                    "message": f"Non-loop cycle: {' -> '.join(cycle)}",
                }
            )

    # Disconnected nodes
    all_connected = {e.source for e in graph.edges} | {e.target for e in graph.edges}
    for node in graph.nodes:
        if node.id not in all_connected and len(graph.nodes) > 1:
            issues.append(
                {
                    "severity": "warning",
                    "message": f"Node '{node.id}' is disconnected",
                    "node_id": node.id,
                }
            )

    # Missing node references in edges
    node_ids = {n.id for n in graph.nodes}
    for edge in graph.edges:
        if edge.source not in node_ids:
            issues.append(
                {
                    "severity": "error",
                    "message": f"Edge '{edge.id}' references missing source '{edge.source}'",
                }
            )
        if edge.target not in node_ids:
            issues.append(
                {
                    "severity": "error",
                    "message": f"Edge '{edge.id}' references missing target '{edge.target}'",
                }
            )

    # Missing start/end
    start_nodes = [n for n in graph.nodes if n.type == "start"]
    end_nodes = [n for n in graph.nodes if n.type == "end"]
    if not start_nodes and graph.nodes:
        issues.append({"severity": "warning", "message": "No explicit start node"})
    if not end_nodes and graph.nodes:
        issues.append({"severity": "warning", "message": "No explicit end node"})

    has_errors = any(i.get("severity") == "error" for i in issues)

    if json_output:
        print(
            json.dumps(
                {
                    "valid": not has_errors,
                    "issues": issues,
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edges),
                },
                indent=2,
            )
        )
        if has_errors:
            raise typer.Exit(1)
        return

    if not issues:
        console.print(
            Panel(
                f"Nodes: [bold]{len(graph.nodes)}[/bold]\nEdges: [bold]{len(graph.edges)}[/bold]",
                title="[green]Graph is valid[/green]",
                border_style="green",
            )
        )
        return

    console.print(
        Panel(
            f"Nodes: [bold]{len(graph.nodes)}[/bold]  |  "
            f"Edges: [bold]{len(graph.edges)}[/bold]  |  "
            f"Issues: [bold]{len(issues)}[/bold]",
            title="[yellow]Validation Results[/yellow]"
            if not has_errors
            else "[red]Validation Failed[/red]",
            border_style="yellow" if not has_errors else "red",
        )
    )

    table = Table(title="Issues")
    table.add_column("Severity")
    table.add_column("Message")
    table.add_column("Node", style="dim")

    for issue in issues:
        sev = issue["severity"]
        color = "red" if sev == "error" else "yellow"
        table.add_row(
            f"[{color}]{sev}[/{color}]",
            issue["message"],
            issue.get("node_id", ""),
        )

    console.print(table)

    if has_errors:
        raise typer.Exit(1)
