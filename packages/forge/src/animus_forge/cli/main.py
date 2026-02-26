"""Gorgon CLI - Main entry point."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

__version__ = "0.3.0"

app = typer.Typer(
    name="gorgon",
    help="Your personal army of AI agents for development workflows.",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"[bold cyan]gorgon[/bold cyan] version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
):
    """Gorgon - Multi-agent AI workflow orchestration.

    Coordinate specialized AI agents (Planner, Builder, Tester, Reviewer)
    across your development workflows.

    [bold]Quick Start:[/bold]

        gorgon init         Create a new workflow template
        gorgon run WORKFLOW Execute a workflow
        gorgon plan TASK    Plan implementation steps
        gorgon build TASK   Generate code for a task
        gorgon test TASK    Generate tests for code
        gorgon review PATH  Review code for issues
        gorgon ask QUESTION Ask a question about your code

    [bold]Shell Completion:[/bold]

        # Bash
        gorgon --install-completion bash

        # Zsh
        gorgon --install-completion zsh

        # Fish
        gorgon --install-completion fish
    """
    pass


# =============================================================================
# Register sub-app commands
# =============================================================================

from .commands.admin import plugins_app  # noqa: E402
from .commands.browser import browser_app  # noqa: E402
from .commands.budget import budget_app  # noqa: E402
from .commands.calendar_cmd import calendar_app  # noqa: E402
from .commands.config import config_app  # noqa: E402
from .commands.coordination import coordination_app  # noqa: E402
from .commands.eval_cmd import eval_app  # noqa: E402
from .commands.graph import graph_app  # noqa: E402
from .commands.history import history_app  # noqa: E402
from .commands.mcp import mcp_app  # noqa: E402
from .commands.memory import memory_app  # noqa: E402
from .commands.metrics import metrics_app  # noqa: E402
from .commands.schedule import schedule_app  # noqa: E402

app.add_typer(schedule_app, name="schedule")
app.add_typer(memory_app, name="memory")
app.add_typer(budget_app, name="budget")
app.add_typer(metrics_app, name="metrics")
app.add_typer(config_app, name="config")
app.add_typer(plugins_app, name="plugins")
app.add_typer(calendar_app, name="calendar")
app.add_typer(browser_app, name="browser")
app.add_typer(history_app, name="history")
app.add_typer(mcp_app, name="mcp")
app.add_typer(graph_app, name="graph")
app.add_typer(eval_app, name="eval")
app.add_typer(coordination_app, name="coordination")


# =============================================================================
# Register top-level commands
# =============================================================================

from .commands.admin import dashboard, logs  # noqa: E402
from .commands.dev import (  # noqa: E402
    ask,
    build,
    do_task,
    plan,
    review,
    test,
)
from .commands.setup import completion, init, tui, version_cmd  # noqa: E402
from .commands.workflow import (  # noqa: E402
    list_workflows,
    run,
    status,
    validate,
)

app.command()(run)
app.command("list")(list_workflows)
app.command()(validate)
app.command()(status)
app.command("do")(do_task)
app.command()(plan)
app.command()(build)
app.command()(test)
app.command()(review)
app.command()(ask)
app.command()(tui)
app.command()(init)
app.command("version", hidden=True)(version_cmd)
app.command()(completion)
app.command()(dashboard)
app.command()(logs)


# =============================================================================
# Backward-compatible re-exports (tests import from animus_forge.cli.main)
# =============================================================================

__all__ = [
    "app",
    "list_workflows_table",
    "detect_codebase_context",
    "format_context_for_prompt",
    "get_claude_client",
    "get_tracker",
    "get_workflow_engine",
    "get_workflow_executor",
]

from .commands.workflow import (  # noqa: E402, F401
    _display_workflow_preview,
    _load_workflow_from_source,
    _output_run_results,
    _output_validation_results,
    _validate_cli_next_step_refs,
    _validate_cli_required_fields,
    _validate_cli_steps,
    list_workflows_table,
)
from .detection import (  # noqa: E402, F401
    _detect_js_framework,
    _detect_language_and_framework,
    _detect_python_framework,
    _get_key_structure,
    _get_readme_content,
    detect_codebase_context,
    format_context_for_prompt,
)
from .helpers import (  # noqa: E402, F401
    _parse_cli_variables,
    get_claude_client,
    get_tracker,
    get_workflow_engine,
    get_workflow_executor,
)

# Keep version command accessible as 'version' for backward compat
version = version_cmd  # noqa: F811


if __name__ == "__main__":
    app()
