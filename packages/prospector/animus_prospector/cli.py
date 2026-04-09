"""CLI for Animus Prospector."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from animus_prospector.config import ProspectorConfig
from animus_prospector.feed import OpportunityFeed
from animus_prospector.orchestrator import Prospector

console = Console()
app = typer.Typer(
    name="animus-prospector",
    help="Autonomous crypto opportunity hunter for Animus.",
    no_args_is_help=True,
)


def _load_config() -> ProspectorConfig:
    config = ProspectorConfig()
    if config.config_file.exists():
        config = ProspectorConfig.from_yaml(config.config_file)
    config.ensure_dirs()
    return config


@app.command()
def scan() -> None:
    """Run a full opportunity scan."""
    config = _load_config()
    prospector = Prospector(config)
    report = asyncio.run(prospector.run_scan())

    console.print("\n[bold]Scan Complete[/bold]")
    console.print(f"  Scanned: {report.total_scanned}")
    console.print(f"  New: [green]{report.new_discovered}[/green]")
    console.print(f"  Auto-execute: {report.auto_execute}")
    console.print(f"  New faucets: {report.add_to_faucet}")
    console.print(f"  Flagged: {report.flagged_for_review}")
    console.print(f"  Skipped: {report.skipped}")


@app.command()
def feed(
    view: str = typer.Argument("summary", help="actionable|flagged|faucets|summary"),
) -> None:
    """View the opportunity feed."""
    config = _load_config()
    opportunity_feed = OpportunityFeed(config.data_dir)

    if view == "summary":
        summary = opportunity_feed.get_summary()
        console.print("\n[bold]Opportunity Feed[/bold]")
        console.print(f"  Total: {summary['total_opportunities']}")
        console.print(f"  Evaluated: {summary['total_evaluations']}")
        if summary["by_type"]:
            console.print("  By type:")
            for t, count in summary["by_type"].items():
                console.print(f"    {t}: {count}")
        if summary["by_action"]:
            console.print("  By action:")
            for a, count in summary["by_action"].items():
                console.print(f"    {a}: {count}")
    elif view == "actionable":
        items = opportunity_feed.get_actionable()
        _print_opportunities(items, "Actionable Opportunities")
    elif view == "flagged":
        items = opportunity_feed.get_flagged()
        _print_opportunities(items, "Flagged for Review")
    elif view == "faucets":
        faucets = opportunity_feed.get_new_faucets()
        if not faucets:
            console.print("No new faucets discovered.")
            return
        table = Table(title="New Faucets")
        table.add_column("Title")
        table.add_column("Chain")
        table.add_column("URL")
        for f in faucets:
            table.add_row(f.title[:50], f.chain.value, f.url[:60])
        console.print(table)
    else:
        console.print(f"[red]Unknown view: {view}[/red]")


def _print_opportunities(items, title: str) -> None:
    if not items:
        console.print(f"No {title.lower()}.")
        return
    table = Table(title=title)
    table.add_column("Title")
    table.add_column("Type")
    table.add_column("Chain")
    table.add_column("ROI")
    table.add_column("Risk")
    table.add_column("Action")
    for opp, ev in items:
        table.add_row(
            opp.title[:40],
            opp.opportunity_type.value,
            opp.chain.value,
            f"{ev.roi_score:.2f}" if ev else "—",
            f"{ev.scam_probability:.0%}" if ev else "—",
            ev.recommended_action.value if ev else "—",
        )
    console.print(table)


@app.command()
def health() -> None:
    """Check hunter health."""
    config = _load_config()
    prospector = Prospector(config)
    results = asyncio.run(prospector.health_check())

    table = Table(title="Hunter Health")
    table.add_column("Hunter")
    table.add_column("Status")
    for name, ok in results.items():
        status = "[green]OK[/green]" if ok else "[red]DOWN[/red]"
        table.add_row(name, status)
    console.print(table)
