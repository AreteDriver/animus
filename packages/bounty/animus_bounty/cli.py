"""CLI for Animus Bounty."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from animus_bounty.config import BountyConfig
from animus_bounty.matcher import SkillMatcher
from animus_bounty.platforms.gitcoin import GitcoinPlatform
from animus_bounty.platforms.github import GitHubPlatform
from animus_bounty.tracker import BountyTracker

console = Console()
app = typer.Typer(
    name="animus-bounty",
    help="Autonomous bounty and task claiming for Animus.",
    no_args_is_help=True,
)


def _load_config() -> BountyConfig:  # pragma: no cover
    config = BountyConfig()
    if config.config_file.exists():
        config = BountyConfig.from_yaml(config.config_file)
    config.ensure_dirs()
    return config


@app.command()
def scan() -> None:  # pragma: no cover
    """Scan all platforms for new bounties."""
    config = _load_config()
    tracker = BountyTracker(config.data_dir)
    scanners = [GitHubPlatform(config), GitcoinPlatform(config)]

    async def _run():
        for scanner in scanners:
            result = await scanner.scan()
            new_count = tracker.record_bounties(result.bounties)
            console.print(
                f"  {scanner.name}: found {result.count}, "
                f"[green]{new_count} new[/green], "
                f"{len(result.errors)} errors"
            )

    console.print("[bold]Scanning platforms...[/bold]")
    asyncio.run(_run())
    summary = tracker.get_summary()
    console.print(f"\nTotal discovered: {summary['total_discovered']}")
    console.print(f"Total earned: ${summary['total_earned_usd']}")


@app.command()
def status() -> None:  # pragma: no cover
    """Show bounty tracking summary."""
    config = _load_config()
    tracker = BountyTracker(config.data_dir)
    summary = tracker.get_summary()

    console.print("\n[bold]Bounty Tracker Summary[/bold]")
    console.print(f"  Discovered: {summary['total_discovered']}")
    console.print(f"  Claims: {summary['total_claims']}")
    console.print(f"  Matches: {summary['total_matches']}")
    console.print(f"  Earned: [green]${summary['total_earned_usd']}[/green]")
    console.print(f"  Pending: ${summary['total_pending_usd']}")
    console.print(f"  Avg match: {summary['avg_match_score']}")

    if summary["status_breakdown"]:
        console.print("  Status:")
        for s, count in summary["status_breakdown"].items():
            console.print(f"    {s}: {count}")

    if summary["platform_breakdown"]:
        console.print("  Platforms:")
        for p, count in summary["platform_breakdown"].items():
            console.print(f"    {p}: {count}")


@app.command()
def match(
    bounty_id: str = typer.Argument(..., help="Bounty ID to evaluate"),
) -> None:  # pragma: no cover
    """Evaluate a bounty against your skill profile."""
    config = _load_config()
    tracker = BountyTracker(config.data_dir)
    matcher = SkillMatcher(config)

    bounty = tracker.bounty_store.get_by_id(bounty_id)
    if bounty is None:
        console.print(f"[red]Bounty not found: {bounty_id}[/red]")
        raise typer.Exit(1)

    result = matcher.quick_match(bounty)
    tracker.record_match(result)

    table = Table(title=f"Match: {bounty.title[:50]}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Score", f"{result.match_score:.2f}")
    table.add_row("Matching", ", ".join(result.matching_skills) or "none")
    table.add_row("Missing", ", ".join(result.missing_skills) or "none")
    table.add_row("Est. Hours", f"{result.estimated_hours}")
    table.add_row("Auto-claim?", str(matcher.should_auto_claim(bounty, result)))
    console.print(table)


@app.command()
def list_bounties(
    platform: str = typer.Option("", help="Filter by platform"),
    limit: int = typer.Option(20, help="Max bounties to show"),
) -> None:  # pragma: no cover
    """List discovered bounties."""
    config = _load_config()
    tracker = BountyTracker(config.data_dir)

    bounties = tracker.bounty_store.load_all()
    if platform:
        bounties = [b for b in bounties if b.platform.value == platform]
    bounties = bounties[:limit]

    if not bounties:
        console.print("No bounties found.")
        return

    table = Table(title="Discovered Bounties")
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Platform")
    table.add_column("Reward")
    table.add_column("Difficulty")
    for b in bounties:
        table.add_row(
            b.bounty_id[:8],
            b.title[:40],
            b.platform.value,
            f"${b.reward_usd}" if b.reward_usd else "?",
            b.difficulty.value,
        )
    console.print(table)
