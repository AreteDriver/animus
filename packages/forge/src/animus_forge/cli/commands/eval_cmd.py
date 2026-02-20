"""Eval commands — agent evaluation and benchmarking."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from ..helpers import console

eval_app = typer.Typer(help="Agent evaluation and benchmarking")


@eval_app.command("run")
def eval_run(
    suite: str = typer.Argument(help="Suite name to run (e.g. 'planner')"),
    model: str | None = typer.Option(None, "--model", "-m", help="Override model name"),
    parallel: bool = typer.Option(False, "--parallel", "-p", help="Run cases in parallel"),
    output: str | None = typer.Option(None, "--output", "-o", help="Save JSON report to file"),
    mock: bool = typer.Option(False, "--mock", help="Use mock evaluator (no API calls)"),
    suites_dir: str | None = typer.Option(None, "--suites-dir", help="Custom suites directory"),
) -> None:
    """Run an evaluation suite against an agent."""
    from animus_forge.evaluation.loader import SuiteLoader

    # Load suite
    loader = SuiteLoader(Path(suites_dir) if suites_dir else None)
    try:
        eval_suite = loader.load_suite(suite)
    except FileNotFoundError:
        console.print(f"[red]Suite '{suite}' not found.[/red]")
        available = loader.list_suites()
        if available:
            names = ", ".join(s["name"] for s in available)
            console.print(f"[dim]Available: {names}[/dim]")
        raise typer.Exit(1)

    # Extract agent_role from suite tags
    agent_role = None
    for tag in eval_suite.tags:
        if tag.startswith("role:"):
            agent_role = tag[5:]
            break

    # Build evaluator
    if mock:
        from animus_forge.evaluation.base import AgentEvaluator
        from animus_forge.providers.mock_provider import MockProvider

        provider = MockProvider()
        evaluator = AgentEvaluator(
            agent_fn=lambda prompt: (
                provider.complete(
                    __import__(
                        "animus_forge.providers.base", fromlist=["CompletionRequest"]
                    ).CompletionRequest(prompt=str(prompt))
                ).content
            ),
            threshold=eval_suite.threshold,
        )
        run_mode = "mock"
    else:
        from animus_forge.evaluation.base import ProviderEvaluator

        try:
            from animus_forge.providers import get_provider

            provider = get_provider()
            if model:
                provider.config.default_model = model
        except Exception as e:
            console.print(f"[red]Failed to initialize provider: {e}[/red]")
            console.print("[dim]Use --mock for CI runs without API keys.[/dim]")
            raise typer.Exit(1)

        evaluator = ProviderEvaluator(
            provider=provider,
            threshold=eval_suite.threshold,
        )
        run_mode = "live"

    # Run
    from animus_forge.evaluation.runner import EvalRunner

    completed = [0]
    total = len(eval_suite.cases)

    def progress_callback(current: int, total_count: int, result: object) -> None:
        completed[0] = current
        status_char = "." if getattr(result, "passed", False) else "x"
        console.print(f"  [{current}/{total_count}] {status_char}", end="\r", highlight=False)

    runner = EvalRunner(evaluator, progress_callback=progress_callback)

    console.print(f"\n[bold]Running suite:[/bold] {suite} ({total} cases)")
    result = runner.run(eval_suite, parallel=parallel)
    console.print()  # clear progress line

    # Report
    from animus_forge.evaluation.reporters import ConsoleReporter

    reporter = ConsoleReporter(verbose=True)
    console.print(reporter.report(result))

    # Save JSON if requested
    if output:
        from animus_forge.evaluation.reporters import JSONReporter

        json_reporter = JSONReporter(include_outputs=True)
        Path(output).write_text(json_reporter.report(result))
        console.print(f"\n[green]Report saved to {output}[/green]")

    # Store results persistently
    try:
        from animus_forge.evaluation.store import get_eval_store

        store = get_eval_store()
        run_id = store.record_run(
            suite_name=suite,
            result=result,
            agent_role=agent_role,
            model=model or ("mock-model" if mock else None),
            run_mode=run_mode,
        )
        console.print(f"[dim]Run stored: {run_id[:8]}[/dim]")

        # Feed to OutcomeTracker
        fed = store.feed_to_outcome_tracker(run_id, workflow_id=f"eval-{suite}")
        if fed:
            console.print(f"[dim]Fed {fed} outcomes to tracker[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not store results: {e}[/yellow]")

    # Exit code based on threshold
    if result.pass_rate < eval_suite.threshold:
        raise typer.Exit(1)


@eval_app.command("list")
def eval_list(
    suites_dir: str | None = typer.Option(None, "--suites-dir", help="Custom suites directory"),
) -> None:
    """List available evaluation suites."""
    from animus_forge.evaluation.loader import SuiteLoader

    loader = SuiteLoader(Path(suites_dir) if suites_dir else None)
    suites = loader.list_suites()

    if not suites:
        console.print("[dim]No suites found.[/dim]")
        return

    table = Table(title="Evaluation Suites")
    table.add_column("Name", style="bold")
    table.add_column("Agent Role")
    table.add_column("Cases", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Description")

    for s in suites:
        table.add_row(
            s["name"],
            s.get("agent_role") or "-",
            str(s["cases_count"]),
            f"{s['threshold']:.0%}",
            (s.get("description") or "")[:60],
        )

    console.print(table)


@eval_app.command("results")
def eval_results(
    suite: str | None = typer.Argument(None, help="Filter by suite name"),
    agent: str | None = typer.Option(None, "--agent", "-a", help="Filter by agent role"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of runs to show"),
) -> None:
    """Show recent evaluation results."""
    try:
        from animus_forge.evaluation.store import get_eval_store

        store = get_eval_store()
        runs = store.query_runs(suite_name=suite, agent_role=agent, limit=limit)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not runs:
        console.print("[dim]No eval runs found.[/dim]")
        return

    table = Table(title="Evaluation Results")
    table.add_column("ID", style="dim")
    table.add_column("Suite")
    table.add_column("Agent")
    table.add_column("Mode")
    table.add_column("Cases", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Completed")

    for r in runs:
        pass_rate = r.get("pass_rate", 0)
        rate_style = "green" if pass_rate >= 0.7 else "yellow" if pass_rate >= 0.5 else "red"
        table.add_row(
            r["id"][:8],
            r["suite_name"],
            r.get("agent_role") or "-",
            r.get("run_mode", "-"),
            str(r.get("total_cases", 0)),
            str(r.get("passed", 0)),
            f"[{rate_style}]{pass_rate:.0%}[/{rate_style}]",
            f"{r.get('avg_score', 0):.2%}",
            f"{r.get('duration_ms', 0):.0f}ms",
            str(r.get("completed_at", ""))[:16],
        )

    console.print(table)
