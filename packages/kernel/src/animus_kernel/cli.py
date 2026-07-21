"""Animus Kernel CLI — Typer entry point for kernel subcommands."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from animus_kernel.sandbox.orchestrator import SelfImproveOrchestrator, WorkflowStage
from animus_kernel.sandbox.safety import SafetyConfig

app = typer.Typer(help="Animus Kernel — autonomous builder engine")
console = Console()
logger = logging.getLogger(__name__)


def _load_provider() -> Any:
    """Load an AI provider if API keys are present, otherwise None."""
    from animus_kernel.providers import OllamaProvider, get_provider

    if os.environ.get("ANTHROPIC_API_KEY"):
        return get_provider("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        return get_provider("openai")

    # Check for Ollama availability
    from animus_kernel.config import warn_if_ollama_unreachable

    if warn_if_ollama_unreachable():
        return OllamaProvider()

    return None


@app.command("self-improve")
def self_improve(
    category: str = typer.Option(
        None,
        "--category",
        "-c",
        help="Focus category (performance, refactoring, documentation, test_coverage, code_quality, bug_fixes)",
    ),
    allow_self_targeting: bool = typer.Option(
        False,
        "--allow-self-targeting",
        help="Allow the self-improvement module to target itself (recursive mode).",
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Auto-approve all stages (requires ANIMUS_FORGE_ALLOW_AUTO_APPROVE=1).",
    ),
    recursive_depth: int = typer.Option(
        0,
        "--recursive-depth",
        help="Current recursion depth (internal use).",
        hidden=True,
    ),
    codebase_path: Path = typer.Option(
        Path("."),
        "--path",
        "-p",
        help="Path to codebase root.",
    ),
    config_path: Path = typer.Option(
        None,
        "--config",
        help="Path to safety config YAML.",
    ),
) -> None:
    """Run the self-improvement workflow on the codebase."""
    config = SafetyConfig.load(config_path) if config_path else SafetyConfig.load()
    if allow_self_targeting:
        config.allow_self_targeting = True

    provider = _load_provider()
    if provider is None:
        console.print("[yellow]No AI provider available — using static analysis only.[/yellow]")

    orchestrator = SelfImproveOrchestrator(
        codebase_path=codebase_path,
        provider=provider,
        config=config,
    )

    try:
        result = asyncio.run(
            orchestrator.run(
                focus_category=category,
                auto_approve=auto_approve,
                recursive_depth=recursive_depth,
            )
        )
    except RuntimeError as exc:
        console.print(f"[red]Blocked: {exc}[/red]")
        raise typer.Exit(1)

    # Display results
    table = Table(title="Self-Improvement Result")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Success", "✅ Yes" if result.success else "❌ No")
    table.add_row("Stage Reached", result.stage_reached.value)
    table.add_row("Recursive Depth", str(result.recursive_depth))

    if result.plan:
        table.add_row("Plan ID", result.plan.id)
        table.add_row("Plan Title", result.plan.title)
        table.add_row("Estimated Files", ", ".join(result.plan.estimated_files))

    if result.sandbox_result:
        table.add_row("Tests Passed", "✅" if result.sandbox_result.tests_passed else "❌")
        table.add_row("Lint Passed", "✅" if result.sandbox_result.lint_passed else "❌")
        if result.sandbox_result.performance_regression:
            table.add_row("Performance", "[red]REGRESSION DETECTED[/red]")

    if result.violations:
        table.add_row("Violations", str(len(result.violations)))

    if result.error:
        table.add_row("Error", f"[red]{result.error}[/red]")

    console.print(table)

    if result.success and result.stage_reached == WorkflowStage.COMPLETE and config.allow_self_targeting:
        console.print(
            "[dim]Recursive self-targeting enabled — "
            f"run again with --recursive-depth={recursive_depth + 1} to continue.[/dim]"
        )

    if not result.success:
        raise typer.Exit(1)


@app.command("analyze")
def analyze(
    category: str = typer.Option(
        None,
        "--category",
        "-c",
        help="Focus category (performance, refactoring, documentation, test_coverage, code_quality, bug_fixes)",
    ),
    allow_self_targeting: bool = typer.Option(
        False,
        "--allow-self-targeting",
        help="Allow analyzing the self-improvement module itself.",
    ),
    codebase_path: Path = typer.Option(Path("."), "--path", "-p"),
) -> None:
    """Static-only analysis of the codebase for improvement opportunities."""
    from animus_kernel.sandbox.analyzer import CodebaseAnalyzer

    analyzer = CodebaseAnalyzer(
        codebase_path=codebase_path,
        allow_self_targeting=allow_self_targeting,
    )

    from animus_kernel.sandbox.analyzer import ImprovementCategory

    categories = None
    if category:
        category_map = {c.value: c for c in ImprovementCategory}
        if category.lower() in category_map:
            categories = [category_map[category.lower()]]
        else:
            console.print(f"[red]Unknown category: {category}[/red]")
            raise typer.Exit(1)

    result = analyzer.analyze(categories=categories)

    table = Table(title=f"Codebase Analysis — {result.files_analyzed} files")
    table.add_column("Priority", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Title", style="yellow")
    table.add_column("Files", style="magenta")

    for suggestion in result.suggestions:
        table.add_row(
            str(suggestion.priority),
            suggestion.category.value,
            suggestion.title,
            ", ".join(suggestion.affected_files[:3]),
        )

    console.print(table)


def main() -> None:
    app()
