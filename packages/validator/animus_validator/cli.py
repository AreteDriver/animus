"""CLI for Animus Validator."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from animus_validator.config import ValidatorConfig
from animus_validator.tracker import RewardTracker

console = Console()
app = typer.Typer(
    name="animus-validator",
    help="Autonomous validator and node rewards management for Animus.",
    no_args_is_help=True,
)


def _load_config() -> ValidatorConfig:  # pragma: no cover
    config = ValidatorConfig()
    if config.config_file.exists():
        config = ValidatorConfig.from_yaml(config.config_file)
    config.ensure_dirs()
    return config


@app.command()
def positions(
    network: str = typer.Option(None, help="Filter by network"),
) -> None:  # pragma: no cover
    """List all staking positions."""
    config = _load_config()
    tracker = RewardTracker(config)
    all_positions = tracker.positions_summary()

    if network:
        all_positions = [p for p in all_positions if p["network"] == network]

    if not all_positions:
        console.print("No staking positions found.")
        return

    table = Table(title="Staking Positions")
    table.add_column("Network")
    table.add_column("Validator")
    table.add_column("Amount")
    table.add_column("APY %")
    table.add_column("Pending")
    table.add_column("Auto-Compound")

    for p in all_positions:
        table.add_row(
            p["network"],
            p["validator_address"][:16] + "...",
            f"{p['amount']:.4f}",
            f"{p['apy_pct']:.1f}%",
            f"{p['rewards_pending']:.6f}",
            "Yes" if p["auto_compound"] else "No",
        )
    console.print(table)


@app.command()
def rewards(
    days: int = typer.Option(30, help="Number of days for recent rewards"),
) -> None:  # pragma: no cover
    """Show earnings report."""
    config = _load_config()
    tracker = RewardTracker(config)
    report = tracker.yield_report()

    console.print("\n[bold]Yield Report[/bold]")
    console.print(f"  Total rewards: {report['total_rewards']:.6f}")
    console.print(f"  Total rewards (USD): ${report['total_rewards_usd']:.2f}")
    console.print(f"  Total staked: {report['total_staked']:.4f}")
    console.print(f"  Pending rewards: {report['total_pending_rewards']:.6f}")
    console.print(f"  Positions: {report['position_count']}")
    console.print(f"  Daily avg (30d): {report['daily_avg_30d']:.6f}")

    if report["earnings_by_network"]:
        console.print("  By network:")
        for net, amount in report["earnings_by_network"].items():
            console.print(f"    {net}: {amount:.6f}")


@app.command()
def discover(
    min_apy: float = typer.Option(3.0, help="Minimum APY to show"),
) -> None:  # pragma: no cover
    """Find staking opportunities."""
    from animus_validator.discovery import OpportunityDiscovery
    from animus_validator.models import Network
    from animus_validator.networks import EthereumNetwork, SolanaNetwork, SuiNetwork

    config = _load_config()
    networks = {
        Network.SUI: SuiNetwork(),
        Network.ETHEREUM: EthereumNetwork(),
        Network.SOLANA: SolanaNetwork(),
    }
    discovery = OpportunityDiscovery(config, networks)
    opps = asyncio.run(discovery.discover())

    opps = [o for o in opps if o.net_apy >= min_apy]

    if not opps:
        console.print("No opportunities above minimum APY threshold.")
        return

    table = Table(title="Staking Opportunities")
    table.add_column("Network")
    table.add_column("Validator")
    table.add_column("APY %")
    table.add_column("Net APY %")
    table.add_column("Commission %")

    for o in opps:
        table.add_row(
            o.network.value,
            o.name[:30],
            f"{o.apy_pct:.1f}%",
            f"{o.net_apy:.1f}%",
            f"{o.commission_pct:.1f}%",
        )
    console.print(table)


@app.command()
def health() -> None:  # pragma: no cover
    """Check network health."""
    from animus_validator.models import Network
    from animus_validator.monitor import NodeMonitor
    from animus_validator.networks import EthereumNetwork, SolanaNetwork, SuiNetwork

    config = _load_config()
    networks = {
        Network.SUI: SuiNetwork(),
        Network.ETHEREUM: EthereumNetwork(),
        Network.SOLANA: SolanaNetwork(),
    }
    monitor = NodeMonitor(config, networks)
    health_results = asyncio.run(monitor.network_health())

    table = Table(title="Network Health")
    table.add_column("Network")
    table.add_column("Status")
    for name, ok in health_results.items():
        status = "[green]OK[/green]" if ok else "[red]DOWN[/red]"
        table.add_row(name, status)
    console.print(table)
