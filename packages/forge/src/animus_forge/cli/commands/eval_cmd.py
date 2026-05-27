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
    adapter: str | None = typer.Option(
        None,
        "--adapter",
        help=(
            "Dotted path to an agent callable (e.g. "
            "'eval_suites.adapters.benchgoblins_ask:run_ask'). Takes case.input "
            "and returns the agent output string. Overrides default provider."
        ),
    ),
    suites_dir: str | None = typer.Option(None, "--suites-dir", help="Custom suites directory"),
    rubric_name: str | None = typer.Option(
        None, "--rubric", "-r", help="Apply named rubric for scoring + grading"
    ),
    rubrics_dir: str | None = typer.Option(None, "--rubrics-dir", help="Custom rubrics directory"),
    prompt_version: str | None = typer.Option(
        None, "--prompt-version", help="Tag run with prompt version (hash or semver)"
    ),
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

    # Load rubric if specified
    rubric = None
    if rubric_name:
        from animus_forge.evaluation.rubric import RubricLibrary

        try:
            library = RubricLibrary(Path(rubrics_dir) if rubrics_dir else None)
            rubric = library.load(rubric_name)
        except FileNotFoundError as e:
            console.print(f"[red]Rubric error:[/red] {e}")
            raise typer.Exit(1)

    # Extract agent_role from suite tags
    agent_role = None
    for tag in eval_suite.tags:
        if tag.startswith("role:"):
            agent_role = tag[5:]
            break

    # Reject mutually-exclusive flags up front
    if adapter and mock:
        console.print("[red]--adapter and --mock are mutually exclusive.[/red]")
        raise typer.Exit(1)

    # Build evaluator
    if adapter:
        import importlib
        import sys as _sys
        from pathlib import Path as _Path

        from animus_forge.evaluation.base import AgentEvaluator

        # Add forge package root to sys.path so `eval_suites.adapters.*`
        # imports resolve. Uses namespace packages (no __init__.py required).
        forge_root = _Path(__file__).resolve().parents[4]
        if str(forge_root) not in _sys.path:
            _sys.path.insert(0, str(forge_root))

        module_name, _, callable_name = adapter.partition(":")
        if not callable_name:
            console.print(
                f"[red]Invalid --adapter spec '{adapter}'. Use 'module.path:callable'.[/red]"
            )
            raise typer.Exit(1)
        try:
            module = importlib.import_module(module_name)
            agent_fn = getattr(module, callable_name)
        except (ImportError, AttributeError) as e:
            console.print(f"[red]Could not load adapter '{adapter}': {e}[/red]")
            raise typer.Exit(1)

        evaluator = AgentEvaluator(agent_fn=agent_fn, threshold=eval_suite.threshold)
        run_mode = f"adapter:{adapter}"
    elif mock:
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

    # Swap suite metrics for rubric-built metrics if rubric is in use
    if rubric is not None:
        provider_for_metrics = None if mock else getattr(evaluator, "provider", None)
        # Adapter-mode evaluators don't carry a .provider attr — without one,
        # LLMJudgeMetric scores 0.5 on every dim. Stand up a separate judge
        # provider from --model when the caller explicitly opted into a
        # rubric (i.e. they want real scoring).
        if provider_for_metrics is None and adapter and not mock:
            try:
                from animus_forge.providers import get_provider

                judge_provider = get_provider()
                if model:
                    judge_provider.config.default_model = model
                provider_for_metrics = judge_provider
                console.print(
                    f"[dim]Judge provider: {model or judge_provider.config.default_model}[/dim]"
                )
            except Exception as e:
                console.print(
                    f"[yellow]Warning:[/yellow] judge provider unavailable ({e}); "
                    "rubric will score 0.5 on every dim."
                )
        eval_suite.metrics = rubric.build_metrics(provider_for_metrics)
        console.print(
            f"[dim]Applying rubric: {rubric.name}@{rubric.version} ({len(rubric.dims)} dims)[/dim]"
        )

    runner = EvalRunner(evaluator, progress_callback=progress_callback)

    console.print(f"\n[bold]Running suite:[/bold] {suite} ({total} cases)")
    result = runner.run(eval_suite, parallel=parallel)
    console.print()  # clear progress line

    # Post-run: rubric banding + failure classification
    if rubric is not None:
        for case_result in result.results:
            case_result.rubric_scores = dict(case_result.metrics)
            composite = rubric.composite_score(case_result.rubric_scores)
            case_result.rubric_band = rubric.band_for(composite)

    from animus_forge.evaluation.failure_taxonomy import tag_results

    bucket_counts = tag_results(result.results)
    non_pass_buckets = {k: v for k, v in bucket_counts.items() if k != "passed"}
    if non_pass_buckets:
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(non_pass_buckets.items()))
        console.print(f"[dim]Failure buckets: {summary}[/dim]")

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
            rubric_name=rubric.name if rubric else None,
            rubric_version=rubric.version if rubric else None,
            config_hash=rubric.content_hash() if rubric else None,
            prompt_version=prompt_version,
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


@eval_app.command("compare")
def eval_compare(
    baseline: str = typer.Argument(help="Baseline run id (or 'prev', 'last', 8-char prefix)"),
    candidate: str = typer.Argument(
        "last",
        help="Candidate run id (or 'last', 'prev', 8-char prefix). Default: last",
    ),
    suite: str | None = typer.Option(
        None, "--suite", "-s", help="Resolve 'last'/'prev' within this suite"
    ),
    bootstrap_n: int = typer.Option(1000, "--bootstrap", "-n", help="Bootstrap resamples for CI"),
    seed: int | None = typer.Option(None, "--seed", help="Seed for reproducible bootstrap"),
    regression_threshold: float = typer.Option(
        0.02, "--regression", help="Score drop that flags regression"
    ),
) -> None:
    """A/B compare two eval runs with bootstrap CI on score delta.

    Examples:
        animus-forge eval compare <run-a-id> <run-b-id>
        animus-forge eval compare prev last --suite planner
        animus-forge eval compare 7b3a2c1d last
    """
    from animus_forge.evaluation.compare import compare_runs, resolve_run_id
    from animus_forge.evaluation.store import get_eval_store

    try:
        store = get_eval_store()
        baseline_id = resolve_run_id(store, baseline, suite_name=suite)
        candidate_id = resolve_run_id(store, candidate, suite_name=suite)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if baseline_id == candidate_id:
        console.print("[yellow]Baseline and candidate resolve to the same run.[/yellow]")
        raise typer.Exit(1)

    try:
        report = compare_runs(
            baseline_id,
            candidate_id,
            store=store,
            bootstrap_n=bootstrap_n,
            regression_threshold=regression_threshold,
            seed=seed,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    _render_comparison(report)

    # Exit non-zero if there's a statistically-significant regression
    if report.direction == "regression" and report.significant:
        raise typer.Exit(2)


def _render_comparison(report) -> None:  # noqa: ANN001 — Rich console rendering
    """Render a ComparisonReport to the console."""
    from animus_forge.evaluation.compare import ComparisonReport

    assert isinstance(report, ComparisonReport)  # nosec B101 — runtime guard

    base = report.baseline
    cand = report.candidate

    header = Table(
        title=f"A/B: {base.suite_name}",
        show_header=True,
        header_style="bold",
    )
    header.add_column("Field")
    header.add_column("Baseline", style="cyan")
    header.add_column("Candidate", style="magenta")

    header.add_row("run_id", base.run_id[:8], cand.run_id[:8])
    header.add_row("model", base.model or "-", cand.model or "-")
    header.add_row(
        "rubric",
        f"{base.rubric_name or '-'}@{base.rubric_version or '-'}",
        f"{cand.rubric_name or '-'}@{cand.rubric_version or '-'}",
    )
    header.add_row("prompt_version", base.prompt_version or "-", cand.prompt_version or "-")
    header.add_row("total_cases", str(base.total_cases), str(cand.total_cases))

    console.print(header)

    # Deltas
    delta_table = Table(title="Deltas (candidate - baseline)", show_header=True)
    delta_table.add_column("Metric")
    delta_table.add_column("Baseline", justify="right")
    delta_table.add_column("Candidate", justify="right")
    delta_table.add_column("Δ", justify="right")

    score_color = _delta_color(report.score_delta)
    pass_color = _delta_color(report.pass_rate_delta)
    cost_color = _delta_color(-report.cost_delta_usd)  # negative cost delta = good
    lat_color = _delta_color(-report.latency_delta_ms)  # negative latency = good

    delta_table.add_row(
        "avg_score",
        f"{base.avg_score:.3f}",
        f"{cand.avg_score:.3f}",
        f"[{score_color}]{report.score_delta:+.3f}[/{score_color}]",
    )
    delta_table.add_row(
        "pass_rate",
        f"{base.pass_rate:.0%}",
        f"{cand.pass_rate:.0%}",
        f"[{pass_color}]{report.pass_rate_delta:+.1%}[/{pass_color}]",
    )
    delta_table.add_row(
        "cost_usd",
        f"${base.total_cost_usd:.4f}",
        f"${cand.total_cost_usd:.4f}",
        f"[{cost_color}]{report.cost_delta_usd:+.4f}[/{cost_color}]",
    )
    delta_table.add_row(
        "duration_ms",
        f"{base.duration_ms:.0f}",
        f"{cand.duration_ms:.0f}",
        f"[{lat_color}]{report.latency_delta_ms:+.0f}[/{lat_color}]",
    )

    console.print(delta_table)

    # Bootstrap CI — the core statistical output
    ci_lo, ci_hi = report.score_ci_95
    sig_color = (
        "green"
        if report.significant and report.score_delta > 0
        else ("red" if report.significant else "yellow")
    )
    sig_label = "significant (excludes 0)" if report.significant else "not significant"
    console.print(
        f"\n[bold]Score Δ 95% CI[/bold]: "
        f"[{sig_color}][{ci_lo:+.3f}, {ci_hi:+.3f}][/{sig_color}] "
        f"({report.bootstrap_n} resamples, {sig_label})"
    )

    # Failure buckets (technical taxonomy)
    if report.failure_delta:
        bucket_table = Table(title="Technical failure bucket deltas", show_header=True)
        bucket_table.add_column("Bucket")
        bucket_table.add_column("Baseline", justify="right")
        bucket_table.add_column("Candidate", justify="right")
        bucket_table.add_column("Δ", justify="right")

        for bucket in sorted(report.failure_delta.keys()):
            base_count = base.failure_buckets.get(bucket, 0)
            cand_count = cand.failure_buckets.get(bucket, 0)
            delta = report.failure_delta[bucket]
            # For the 'passed' bucket, positive delta is good; for others, bad.
            good = (bucket == "passed" and delta > 0) or (bucket != "passed" and delta < 0)
            color = "green" if good else "red"
            bucket_table.add_row(
                bucket,
                str(base_count),
                str(cand_count),
                f"[{color}]{delta:+d}[/{color}]",
            )

        console.print(bucket_table)

    # Content-quality bucket deltas (F1-F8)
    if report.content_failure_delta:
        content_table = Table(
            title="Content-quality failure bucket deltas (F1-F8)",
            show_header=True,
        )
        content_table.add_column("Bucket")
        content_table.add_column("Baseline", justify="right")
        content_table.add_column("Candidate", justify="right")
        content_table.add_column("Δ", justify="right")

        for bucket in sorted(report.content_failure_delta.keys()):
            base_count = base.content_failure_buckets.get(bucket, 0)
            cand_count = cand.content_failure_buckets.get(bucket, 0)
            delta = report.content_failure_delta[bucket]
            # Fewer content failures = good
            color = "green" if delta < 0 else "red"
            content_table.add_row(
                bucket,
                str(base_count),
                str(cand_count),
                f"[{color}]{delta:+d}[/{color}]",
            )

        console.print(content_table)

    # Verdict + flags
    verdict_color = {
        "improvement": "green",
        "regression": "red",
        "no-change": "dim",
    }[report.direction]
    console.print(f"\n[bold {verdict_color}]{report.direction.upper()}[/bold {verdict_color}]")
    if report.regression_flags:
        console.print("[bold]Flags:[/bold]")
        for flag in report.regression_flags:
            console.print(f"  [yellow]![/yellow] {flag}")


def _delta_color(delta: float) -> str:
    if delta > 0:
        return "green"
    if delta < 0:
        return "red"
    return "dim"


@eval_app.command("rubrics")
def eval_rubrics(
    rubrics_dir: str | None = typer.Option(None, "--rubrics-dir", help="Custom rubrics directory"),
) -> None:
    """List available scoring rubrics."""
    from animus_forge.evaluation.rubric import RubricLibrary

    library = RubricLibrary(Path(rubrics_dir) if rubrics_dir else None)
    rubrics = library.list_rubrics()

    if not rubrics:
        console.print("[dim]No rubrics found.[/dim]")
        return

    table = Table(title="Scoring Rubrics")
    table.add_column("Name", style="bold")
    table.add_column("Version")
    table.add_column("Dims", justify="right")
    table.add_column("Description")

    for r in rubrics:
        table.add_row(
            r["name"],
            r["version"],
            str(r["dims"]),
            (r.get("description") or "")[:60],
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
