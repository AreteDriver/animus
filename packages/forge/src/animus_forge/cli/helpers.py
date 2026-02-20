"""Shared helpers for CLI modules — client factories, parsing, tracker."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.console import Console

if TYPE_CHECKING:
    from animus_forge.api_clients import ClaudeCodeClient
    from animus_forge.monitoring.tracker import ExecutionTracker
    from animus_forge.orchestrator import WorkflowEngineAdapter
    from animus_forge.workflow.executor import WorkflowExecutor

console = Console()


def get_workflow_engine() -> WorkflowEngineAdapter:
    """Lazy import workflow engine to avoid startup cost."""
    try:
        from animus_forge.orchestrator import WorkflowEngineAdapter

        return WorkflowEngineAdapter()
    except ImportError as e:
        console.print(f"[red]Missing dependencies:[/red] {e}")
        console.print("Run: pip install pydantic-settings")
        raise typer.Exit(1)


def get_claude_client() -> ClaudeCodeClient:
    """Get Claude Code client for direct agent execution."""
    try:
        from animus_forge.api_clients import ClaudeCodeClient

        client = ClaudeCodeClient()
        if not client.is_configured():
            console.print("[red]Claude not configured.[/red]")
            console.print("Set ANTHROPIC_API_KEY environment variable.")
            raise typer.Exit(1)
        return client
    except ImportError as e:
        console.print(f"[red]Missing dependencies:[/red] {e}")
        raise typer.Exit(1)


def get_workflow_executor(dry_run: bool = False) -> WorkflowExecutor:
    """Get workflow executor with checkpoint and budget managers."""
    try:
        from animus_forge.budget import BudgetManager
        from animus_forge.state.checkpoint import CheckpointManager
        from animus_forge.workflow.executor import WorkflowExecutor

        checkpoint_mgr = CheckpointManager()
        budget_mgr = BudgetManager()

        return WorkflowExecutor(
            checkpoint_manager=checkpoint_mgr,
            budget_manager=budget_mgr,
            dry_run=dry_run,
        )
    except ImportError as e:
        console.print(f"[red]Missing dependencies:[/red] {e}")
        raise typer.Exit(1)


def _create_cli_execution_manager():
    """Create a lightweight ExecutionManager for CLI --live mode.

    Returns None if dependencies unavailable (graceful degradation).
    """
    try:
        from animus_forge.executions import ExecutionManager
        from animus_forge.state.backends import SQLiteBackend
        from animus_forge.state.migrations import run_migrations

        backend = SQLiteBackend(db_path=":memory:")
        run_migrations(backend)
        return ExecutionManager(backend=backend)
    except Exception:
        return None


def get_tracker() -> ExecutionTracker | None:
    """Lazy import execution tracker."""
    try:
        from animus_forge.monitoring.tracker import get_tracker as _get_tracker

        return _get_tracker()
    except ImportError:
        return None


def _parse_cli_variables(var: list[str]) -> dict:
    """Parse CLI variables in key=value format."""
    variables = {}
    for v in var:
        if "=" in v:
            key, value = v.split("=", 1)
            variables[key] = value
        else:
            console.print(f"[red]Invalid variable format: {v}[/red]")
            console.print("Use: --var key=value")
            raise typer.Exit(1)
    return variables
