"""Development agent commands — do, plan, build, test, review, ask."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer
from rich.panel import Panel

from ..detection import detect_codebase_context, format_context_for_prompt
from ..helpers import console, get_claude_client, get_workflow_executor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_git_diff_context(target: str, cwd: Path) -> str:
    """Get code context from git diff."""
    try:
        diff = subprocess.run(
            ["git", "diff", target],
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if diff.returncode == 0:
            return f"\nGit diff:\n```diff\n{diff.stdout[:8000]}\n```"
    except Exception:
        pass  # Non-critical fallback: git diff unavailable, proceed without diff context
    return ""


def _get_file_context(target_path: Path) -> str:
    """Get code context from a single file."""
    try:
        return f"\nCode to review:\n```\n{target_path.read_text()[:8000]}\n```"
    except Exception:
        return ""


def _get_directory_context(target_path: Path) -> str:
    """Get code context from a directory of files."""
    files = list(target_path.rglob("*.py"))[:5]
    code_snippets = []
    for f in files:
        try:
            code_snippets.append(f"# {f}\n{f.read_text()[:2000]}")
        except Exception:
            pass  # Non-critical fallback: skip unreadable files in directory context
    if code_snippets:
        return f"\nFiles to review:\n```\n{'---'.join(code_snippets)}\n```"
    return ""


def _gather_review_code_context(target: str, context: dict) -> str:
    """Gather code context for review based on target type."""
    # Check if target is a git ref
    if target.startswith("HEAD") or target.startswith("origin/"):
        return _get_git_diff_context(target, context["path"])

    target_path = Path(target)
    if not target_path.exists():
        return ""

    if target_path.is_file():
        return _get_file_context(target_path)
    if target_path.is_dir():
        return _get_directory_context(target_path)

    return ""


# ---------------------------------------------------------------------------
# Commands (registered by main.py)
# ---------------------------------------------------------------------------


def do_task(
    task: str = typer.Argument(
        ...,
        help="Natural language description of what you want to do",
    ),
    workflow: str = typer.Option(
        "feature-build",
        "--workflow",
        "-w",
        help="Workflow to use (feature-build, bug-fix, refactor)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would happen without executing",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON",
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help="Show real-time execution progress with a Rich Live table",
    ),
):
    """Execute a development task using your agent army.

    Examples:
        gorgon do "add user authentication"
        gorgon do "fix the login bug" --workflow bug-fix
        gorgon do "refactor the database module" --workflow refactor
    """
    from animus_forge.workflow.loader import load_workflow

    # Detect codebase context
    context = detect_codebase_context()
    context_str = format_context_for_prompt(context)

    console.print(
        Panel(
            f"[bold]{task}[/bold]\n\n[dim]{context_str}[/dim]",
            title="🐍 Gorgon Task",
            border_style="cyan",
        )
    )

    # Load workflow
    workflows_dir = Path(__file__).parent.parent.parent.parent / "workflows"
    workflow_path = workflows_dir / f"{workflow}.yaml"

    if not workflow_path.exists():
        console.print(f"[red]Workflow not found:[/red] {workflow}")
        console.print(f"\nAvailable workflows in {workflows_dir}:")
        for wf in workflows_dir.glob("*.yaml"):
            console.print(f"  • {wf.stem}")
        raise typer.Exit(1)

    try:
        wf_config = load_workflow(workflow_path, validate_path=False)
    except Exception as e:
        console.print(f"[red]Failed to load workflow:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"\n[dim]Using workflow:[/dim] {wf_config.name}")
    console.print(f"[dim]Steps:[/dim] {len(wf_config.steps)}")

    if dry_run:
        console.print("\n[yellow]Dry run - showing plan without executing[/yellow]")
        for i, step in enumerate(wf_config.steps, 1):
            role = step.params.get("role", step.type)
            console.print(f"  {i}. [{step.type}] {step.id} ({role})")
        raise typer.Exit(0)

    # Execute workflow
    executor = get_workflow_executor(dry_run=False)

    inputs = {
        "feature_request": task,
        "codebase_path": context["path"],
        "task_description": task,
        "context": context_str,
    }

    console.print()
    if live:
        from rich.live import Live
        from rich.table import Table

        live_steps: dict[str, dict] = {}

        def _step_callback(event_type: str, execution_id: str, **kwargs):
            """Update live table from execution events."""
            if event_type == "status" and kwargs.get("current_step"):
                step_id = kwargs["current_step"]
                live_steps.setdefault(step_id, {})
                live_steps[step_id]["progress"] = kwargs.get("progress", 0)
            elif event_type == "log" and kwargs.get("step_id"):
                step_id = kwargs["step_id"]
                live_steps.setdefault(step_id, {})
                live_steps[step_id]["status"] = kwargs.get("message", "")
            elif event_type == "metrics":
                # Update totals on most recent step
                if live_steps:
                    last_key = list(live_steps.keys())[-1]
                    live_steps[last_key]["tokens"] = kwargs.get("total_tokens", 0)

        # Create an ExecutionManager for local CLI use
        from animus_forge.cli.helpers import _create_cli_execution_manager

        cli_em = _create_cli_execution_manager()
        if cli_em:
            cli_em.register_callback(_step_callback)
            executor.execution_manager = cli_em

        def _build_table() -> Table:
            table = Table(title="Execution Progress")
            table.add_column("Step", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Tokens", justify="right")
            for step_id, info in live_steps.items():
                status = info.get("status", "pending")
                # Truncate long status messages
                if len(status) > 60:
                    status = status[:57] + "..."
                tokens = str(info.get("tokens", ""))
                table.add_row(step_id, status, tokens)
            return table

        with Live(_build_table(), console=console, refresh_per_second=4) as live_display:
            result = executor.execute(wf_config, inputs=inputs)
            live_display.update(_build_table())
    else:
        with console.status("[bold cyan]Agents working...", spinner="dots"):
            result = executor.execute(wf_config, inputs=inputs)

    # Display results
    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
        return

    status_color = "green" if result.status == "success" else "red"
    console.print(f"\n[{status_color}]Status: {result.status}[/{status_color}]")

    if result.steps:
        console.print("\n[bold]Agent Activity:[/bold]")
        for step in result.steps:
            icon = (
                "✓"
                if step.status.value == "success"
                else "✗"
                if step.status.value == "failed"
                else "○"
            )
            role = step.output.get("role", step.step_id) if step.output else step.step_id
            tokens = step.tokens_used
            console.print(f"  {icon} {role}: {step.status.value} ({tokens:,} tokens)")

    if result.error:
        console.print(f"\n[red]Error:[/red] {result.error}")

    console.print(f"\n[dim]Total tokens: {result.total_tokens:,}[/dim]")


def plan(
    task: str = typer.Argument(
        ...,
        help="What do you want to plan?",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """Run the Planner agent to break down a task.

    Example:
        gorgon plan "add OAuth2 authentication to the API"
    """
    client = get_claude_client()
    context = detect_codebase_context()
    context_str = format_context_for_prompt(context)

    console.print(
        Panel(
            f"[bold]Planning:[/bold] {task}",
            title="🗺️ Planner Agent",
            border_style="blue",
        )
    )

    prompt = f"""Analyze and create a detailed implementation plan for:

{task}

{context_str}

Provide:
1. Task breakdown with clear steps
2. Files that need to be created or modified
3. Dependencies and order of operations
4. Potential risks and how to mitigate them
5. Success criteria for the implementation"""

    with console.status("[bold blue]Planner thinking...", spinner="dots"):
        result = client.execute_agent(
            role="planner",
            task=prompt,
            context=context_str,
        )

    if json_output:
        print(json.dumps(result, indent=2))
        return

    if result.get("success"):
        console.print("\n[bold]Implementation Plan:[/bold]\n")
        console.print(result.get("output", "No output"))
    else:
        console.print(f"\n[red]Error:[/red] {result.get('error', 'Unknown error')}")


def build(
    description: str = typer.Argument(
        ...,
        help="What to build",
    ),
    plan: str | None = typer.Option(
        None,
        "--plan",
        "-p",
        help="Path to a plan file or inline plan",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """Run the Builder agent to implement code.

    Example:
        gorgon build "user authentication module"
        gorgon build "login endpoint" --plan "1. Create route 2. Add validation"
    """
    client = get_claude_client()
    context = detect_codebase_context()
    context_str = format_context_for_prompt(context)

    console.print(
        Panel(
            f"[bold]Building:[/bold] {description}",
            title="🔨 Builder Agent",
            border_style="green",
        )
    )

    # Load plan if provided as file
    plan_text = ""
    if plan:
        plan_path = Path(plan)
        if plan_path.exists():
            plan_text = plan_path.read_text()
        else:
            plan_text = plan

    prompt = f"""Implement the following:

{description}

{context_str}
"""
    if plan_text:
        prompt += f"""
Based on this plan:
{plan_text}
"""

    prompt += """
Write production-quality code with:
- Type hints (for Python)
- Error handling
- Clear documentation
- Following existing project patterns"""

    with console.status("[bold green]Builder coding...", spinner="dots"):
        result = client.execute_agent(
            role="builder",
            task=prompt,
            context=context_str,
        )

    if json_output:
        print(json.dumps(result, indent=2))
        return

    if result.get("success"):
        console.print("\n[bold]Implementation:[/bold]\n")
        console.print(result.get("output", "No output"))
    else:
        console.print(f"\n[red]Error:[/red] {result.get('error', 'Unknown error')}")


def test(
    target: str = typer.Argument(
        ".",
        help="File or module to test",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """Run the Tester agent to create tests.

    Example:
        gorgon test src/auth/login.py
        gorgon test "the new user registration flow"
    """
    client = get_claude_client()
    context = detect_codebase_context()
    context_str = format_context_for_prompt(context)

    # Check if target is a file
    target_path = Path(target)
    code_context = ""
    if target_path.exists() and target_path.is_file():
        try:
            code_context = f"\nCode to test:\n```\n{target_path.read_text()[:5000]}\n```"
        except Exception:
            pass  # Non-critical fallback: source file unreadable, proceed without code context

    console.print(
        Panel(
            f"[bold]Testing:[/bold] {target}",
            title="🧪 Tester Agent",
            border_style="yellow",
        )
    )

    prompt = f"""Create comprehensive tests for:

{target}

{context_str}
{code_context}

Write tests that include:
- Unit tests for individual functions
- Edge cases and error conditions
- Integration tests where appropriate
- Clear test names that describe behavior
- Following the project's existing test patterns (pytest for Python)"""

    with console.status("[bold yellow]Tester analyzing...", spinner="dots"):
        result = client.execute_agent(
            role="tester",
            task=prompt,
            context=context_str,
        )

    if json_output:
        print(json.dumps(result, indent=2))
        return

    if result.get("success"):
        console.print("\n[bold]Generated Tests:[/bold]\n")
        console.print(result.get("output", "No output"))
    else:
        console.print(f"\n[red]Error:[/red] {result.get('error', 'Unknown error')}")


def review(
    target: str = typer.Argument(
        ".",
        help="File, directory, or git diff to review",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """Run the Reviewer agent for code review.

    Example:
        gorgon review src/auth/
        gorgon review HEAD~1  # Review last commit
    """
    client = get_claude_client()
    context = detect_codebase_context()
    context_str = format_context_for_prompt(context)
    code_context = _gather_review_code_context(target, context)

    console.print(
        Panel(
            f"[bold]Reviewing:[/bold] {target}",
            title="🔍 Reviewer Agent",
            border_style="magenta",
        )
    )

    prompt = f"""Review the following code:

{target}

{context_str}
{code_context}

Evaluate:
1. Code quality and readability
2. Security concerns (OWASP top 10, input validation, etc.)
3. Performance implications
4. Error handling completeness
5. Test coverage gaps
6. Adherence to project patterns

Provide:
- Approval recommendation (approved/needs_changes/rejected)
- Score (1-10)
- Specific findings with severity (critical/warning/info)
- Actionable improvement suggestions"""

    with console.status("[bold magenta]Reviewer analyzing...", spinner="dots"):
        result = client.execute_agent(
            role="reviewer",
            task=prompt,
            context=context_str,
        )

    if json_output:
        print(json.dumps(result, indent=2))
        return

    if result.get("success"):
        console.print("\n[bold]Code Review:[/bold]\n")
        console.print(result.get("output", "No output"))
    else:
        console.print(f"\n[red]Error:[/red] {result.get('error', 'Unknown error')}")


def ask(
    question: str = typer.Argument(
        ...,
        help="Question about your codebase",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
):
    """Ask a question about your codebase.

    Example:
        gorgon ask "how does the authentication system work?"
        gorgon ask "what are the main API endpoints?"
    """
    client = get_claude_client()
    context = detect_codebase_context()
    context_str = format_context_for_prompt(context)

    console.print(
        Panel(
            f"[bold]{question}[/bold]",
            title="❓ Question",
            border_style="cyan",
        )
    )

    prompt = f"""Answer this question about the codebase:

{question}

{context_str}

Provide a clear, helpful answer based on the codebase context.
If you need to reference specific files or code, mention them explicitly."""

    with console.status("[bold cyan]Thinking...", spinner="dots"):
        result = client.generate_completion(
            prompt=prompt,
            system_prompt="You are a helpful assistant analyzing a software codebase. Be concise and specific.",
        )

    if json_output:
        print(json.dumps({"question": question, "answer": result}, indent=2))
        return

    if result:
        console.print("\n[bold]Answer:[/bold]\n")
        console.print(result)
    else:
        console.print("[red]No response received[/red]")
