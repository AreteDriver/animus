"""CLI commands for skill evolution — metrics, experiments, versions, deprecations."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

skills_app = typer.Typer(help="Skill evolution — metrics, A/B tests, versions, deprecations")
console = Console()


def _get_db():
    from animus_forge.state.database import get_database

    return get_database()


def _get_aggregator():
    from animus_forge.skills.evolver.metrics import SkillMetricsAggregator

    return SkillMetricsAggregator(_get_db())


def _get_ab_manager():
    from animus_forge.skills.evolver.ab_test import ABTestManager

    return ABTestManager(_get_db(), _get_aggregator())


@skills_app.command("metrics")
def metrics(
    days: int = typer.Option(30, "--days", "-d", help="Look-back window in days"),
    skill: str = typer.Option("", "--skill", "-s", help="Filter by skill name"),
) -> None:
    """Show per-skill performance metrics."""
    agg = _get_aggregator()
    all_metrics = agg.get_all_skill_metrics(days=days)

    if skill:
        all_metrics = [m for m in all_metrics if skill.lower() in m.skill_name.lower()]

    if not all_metrics:
        console.print("[dim]No skill metrics recorded.[/dim]")
        return

    table = Table(title=f"Skill Metrics ({days}d)")
    table.add_column("Skill", style="cyan")
    table.add_column("Version")
    table.add_column("Invocations", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Avg Cost", justify="right")
    table.add_column("Trend", justify="center")

    for m in all_metrics:
        trend_icon = {
            "improving": "[green]↑[/]",
            "declining": "[red]↓[/]",
            "stable": "[dim]→[/]",
        }.get(m.trend, "[dim]?[/]")
        table.add_row(
            m.skill_name,
            m.skill_version or "all",
            str(m.total_invocations),
            f"{m.success_rate:.0%}",
            f"{m.avg_quality_score:.2f}",
            f"${m.avg_cost_usd:.4f}",
            trend_icon,
        )

    console.print(table)
    total_cost = sum(m.total_cost_usd for m in all_metrics)
    console.print(f"\n[dim]Total cost: ${total_cost:.2f}[/dim]")


@skills_app.command("experiments")
def experiments() -> None:
    """List active A/B test experiments."""
    ab = _get_ab_manager()
    active = ab.get_active_experiments()

    if not active:
        console.print("[dim]No active experiments.[/dim]")
        return

    table = Table(title="Active Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Skill")
    table.add_column("Control")
    table.add_column("Variant")
    table.add_column("Split", justify="right")
    table.add_column("Started")

    for exp in active:
        table.add_row(
            str(exp.get("id", ""))[:8],
            str(exp.get("skill_name", "?")),
            f"v{exp.get('control_version', '?')}",
            f"v{exp.get('variant_version', '?')}",
            f"{float(exp.get('traffic_split', 0.5)):.0%}",
            str(exp.get("start_date", ""))[:16],
        )

    console.print(table)


@skills_app.command("versions")
def versions(
    skill: str = typer.Argument(help="Skill name to inspect"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max versions to show"),
) -> None:
    """Show version history for a skill."""
    db = _get_db()
    rows = db.fetchall(
        "SELECT skill_name, version, previous_version, change_type, "
        "change_description, created_at "
        "FROM skill_versions WHERE skill_name = ? "
        "ORDER BY created_at DESC LIMIT ?",
        (skill, limit),
    )

    if not rows:
        console.print(f"[dim]No versions recorded for '{skill}'.[/dim]")
        return

    table = Table(title=f"Version History: {skill}")
    table.add_column("Version", style="cyan")
    table.add_column("From")
    table.add_column("Type")
    table.add_column("Description")
    table.add_column("Date", style="dim")

    for r in rows:
        table.add_row(
            str(r["version"]),
            str(r.get("previous_version") or "—"),
            str(r.get("change_type", "?")),
            str(r.get("change_description", ""))[:60],
            str(r.get("created_at", ""))[:16],
        )

    console.print(table)


@skills_app.command("deprecations")
def deprecations() -> None:
    """List skills in the deprecation lifecycle."""
    db = _get_db()
    rows = db.fetchall(
        "SELECT * FROM skill_deprecations ORDER BY flagged_at DESC",
        (),
    )

    if not rows:
        console.print("[dim]No skills flagged for deprecation.[/dim]")
        return

    table = Table(title="Deprecation Tracker")
    table.add_column("Skill", style="cyan")
    table.add_column("Status")
    table.add_column("Reason")
    table.add_column("Success Rate", justify="right")
    table.add_column("Replacement")
    table.add_column("Flagged", style="dim")

    for r in rows:
        status = str(r.get("status", "?"))
        color = {"flagged": "yellow", "deprecated": "red", "retired": "dim"}.get(status, "white")
        table.add_row(
            str(r.get("skill_name", "?")),
            f"[{color}]{status}[/]",
            str(r.get("reason", ""))[:50],
            f"{float(r.get('success_rate_at_flag', 0)):.0%}",
            str(r.get("replacement_skill") or "—"),
            str(r.get("flagged_at", ""))[:16],
        )

    console.print(table)


@skills_app.command("trend")
def trend(
    skill: str = typer.Argument(help="Skill name to check"),
    days: int = typer.Option(30, "--days", "-d", help="Look-back window"),
) -> None:
    """Show trend analysis for a specific skill."""
    agg = _get_aggregator()
    m = agg.get_skill_metrics(skill, days=days)

    if not m:
        console.print(f"[dim]No metrics for '{skill}'.[/dim]")
        return

    trend_val = agg.get_skill_trend(skill, days=days)
    trend_color = {"improving": "green", "declining": "red", "stable": "yellow"}.get(
        trend_val, "white"
    )

    console.print(f"[bold]{skill}[/bold] ({days}d)")
    console.print(f"  Invocations: {m.total_invocations}")
    console.print(f"  Success:     {m.success_rate:.1%}")
    console.print(f"  Quality:     {m.avg_quality_score:.2f}")
    console.print(f"  Avg Cost:    ${m.avg_cost_usd:.4f}")
    console.print(f"  Avg Latency: {m.avg_latency_ms:.0f}ms")
    console.print(f"  Total Cost:  ${m.total_cost_usd:.2f}")
    console.print(f"  Trend:       [{trend_color}]{trend_val}[/]")
