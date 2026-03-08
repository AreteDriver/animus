"""CLI commands for self-improvement workflows."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

self_improve_app = typer.Typer(help="Self-improvement workflows for your codebase.")


@self_improve_app.command("run")
def improve_run(
    path: str | None = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to codebase (defaults to current directory)",
    ),
    focus: str | None = typer.Option(
        None,
        "--focus",
        "-f",
        help="Focus category (e.g., 'performance', 'security', 'testing')",
    ),
    provider: str = typer.Option(
        "ollama",
        "--provider",
        help="AI provider: ollama, anthropic, or openai",
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Auto-approve all stages (skip human gates)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Analyze and plan only — don't apply changes",
    ),
) -> None:
    """Run the self-improvement workflow on a codebase."""
    from animus_forge.agents.provider_wrapper import create_agent_provider
    from animus_forge.self_improve.orchestrator import SelfImproveOrchestrator

    codebase_path = Path(path) if path else Path.cwd()

    if not codebase_path.exists():
        console.print(f"[red]Path not found: {codebase_path}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]Codebase:[/bold] {codebase_path}\n"
            f"[bold]Provider:[/bold] {provider}\n"
            f"[bold]Focus:[/bold] {focus or 'general'}\n"
            f"[bold]Auto-approve:[/bold] {auto_approve}\n"
            f"[bold]Dry run:[/bold] {dry_run}",
            title="Self-Improve",
            border_style="cyan",
        )
    )

    try:
        agent_provider = create_agent_provider(provider)
    except Exception as e:
        console.print(f"[red]Failed to create {provider} provider:[/red] {e}")
        console.print("[dim]Check your API keys or that Ollama is running.[/dim]")
        raise typer.Exit(1) from None

    # Wire tool registry for tool-equipped code generation
    tool_registry = None
    try:
        from animus_forge.tools.registry import ForgeToolRegistry

        tool_registry = ForgeToolRegistry(
            project_root=codebase_path,
            enable_shell=True,
        )
    except Exception:
        pass

    orchestrator = SelfImproveOrchestrator(
        codebase_path=codebase_path,
        provider=agent_provider,
        tool_registry=tool_registry,
    )

    with console.status("[cyan]Running self-improvement pipeline..."):
        result = asyncio.run(
            orchestrator.run(
                focus_category=focus,
                auto_approve=auto_approve or dry_run,
            )
        )

    # Display result
    if result.success:
        style = "green"
        icon = "OK"
    elif result.error:
        style = "red"
        icon = "FAILED"
    else:
        style = "yellow"
        icon = "PARTIAL"

    table = Table(border_style=style)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Status", f"[{style}]{icon}[/{style}]")
    table.add_row("Stage Reached", result.stage_reached.value)

    if result.plan:
        table.add_row("Plan", result.plan.title)
        table.add_row("Suggestions", str(len(result.plan.suggestions)))

    if result.sandbox_result:
        tests = "passed" if result.sandbox_result.tests_passed else "failed"
        table.add_row("Tests", tests)

    if result.pull_request:
        table.add_row("PR", result.pull_request.url or result.pull_request.branch)

    if result.error:
        table.add_row("Error", f"[red]{result.error}[/red]")

    if result.violations:
        table.add_row("Safety Violations", str(len(result.violations)))

    console.print()
    console.print(table)
    console.print()


@self_improve_app.command("analyze")
def improve_analyze(
    path: str | None = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to codebase (defaults to current directory)",
    ),
    focus: str | None = typer.Option(
        None,
        "--focus",
        "-f",
        help="Focus category filter",
    ),
) -> None:
    """Analyze codebase for improvement suggestions (no changes made)."""
    from animus_forge.self_improve.analyzer import CodebaseAnalyzer

    codebase_path = Path(path) if path else Path.cwd()

    with console.status("[cyan]Analyzing codebase..."):
        analyzer = CodebaseAnalyzer(codebase_path=codebase_path)
        result = analyzer.analyze()
        suggestions = result.suggestions

    if focus:
        suggestions = [s for s in suggestions if s.category == focus]

    if not suggestions:
        console.print("[dim]No improvement suggestions found.[/dim]")
        return

    table = Table(title=f"Suggestions ({len(suggestions)})", border_style="cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Category", style="bold")
    table.add_column("Files")
    table.add_column("Title")
    table.add_column("Pri", justify="center", width=3)

    for i, s in enumerate(suggestions[:20], 1):
        cat = s.category.value if hasattr(s.category, "value") else str(s.category)
        files = ", ".join(s.affected_files[:2]) if s.affected_files else "-"
        table.add_row(
            str(i),
            cat,
            files[:40],
            s.title[:50],
            str(s.priority),
        )

    console.print()
    console.print(table)
    if len(suggestions) > 20:
        console.print(f"[dim]... and {len(suggestions) - 20} more[/dim]")
    console.print()
