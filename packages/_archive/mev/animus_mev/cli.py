"""CLI for Animus MEV."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from animus_mev import __version__
from animus_mev.config import MEVConfig

app = typer.Typer(name="animus-mev", help="Autonomous MEV detection for L2 chains.")
console = Console()


@app.command()
def status() -> None:
    """Show MEV module status."""
    config = _load_config()

    console.print(f"\n[bold]Animus MEV v{__version__}[/bold]")
    console.print(
        f"Mode: [{'red' if config.execution.live else 'green'}]"
        f"{'LIVE' if config.execution.live else 'DRY RUN'}[/]"
    )
    console.print(f"Data: {config.data_dir}")

    table = Table(title="Chains")
    table.add_column("Chain")
    table.add_column("Enabled")
    table.add_column("Min Swap USD")
    table.add_column("RPC")

    for chain_cfg in config.chains.values():
        table.add_row(
            chain_cfg.chain.value.upper(),
            "[green]Yes[/]" if chain_cfg.enabled else "[red]No[/]",
            f"${chain_cfg.min_swap_usd:,.0f}",
            chain_cfg.rpc_url[:40] + "..." if len(chain_cfg.rpc_url) > 40 else chain_cfg.rpc_url,
        )
    console.print(table)

    console.print("\n[bold]Risk Limits[/bold]")
    console.print(f"  Max gas/tx: ${config.risk.max_gas_per_tx_usd}")
    console.print(f"  Daily budget: ${config.risk.daily_gas_budget_usd}")
    console.print(f"  Min profit: ${config.risk.min_profit_usd}")
    console.print(f"  Min confidence: {config.risk.min_confidence}")
    console.print("  Sandwich execution: [red]BLOCKED[/red]")


@app.command()
def init() -> None:
    """Initialize MEV config and data directory."""
    config = MEVConfig()
    config.ensure_dirs()
    config.to_yaml(config.config_file)
    console.print(f"[green]Initialized MEV config at {config.config_file}[/green]")


@app.command()
def opportunities() -> None:
    """Show recent MEV opportunities."""
    from animus_mev.store import MEVStore

    config = _load_config()
    store = MEVStore(config.data_dir)
    recent = store.recent_opportunities(limit=20)

    if not recent:
        console.print("[yellow]No opportunities detected yet.[/yellow]")
        return

    table = Table(title="Recent MEV Opportunities")
    table.add_column("ID")
    table.add_column("Type")
    table.add_column("Chain")
    table.add_column("Est. Profit")
    table.add_column("Gas Cost")
    table.add_column("Net")
    table.add_column("Confidence")

    for opp in recent:
        net = opp.net_profit_usd
        net_color = "green" if net > 0 else "red"
        table.add_row(
            opp.opportunity_id[:8],
            opp.mev_type.value,
            opp.chain.value.upper(),
            f"${opp.estimated_profit_usd:.4f}",
            f"${opp.gas_cost_usd:.4f}",
            f"[{net_color}]${net:.4f}[/]",
            f"{opp.confidence:.0%}",
        )
    console.print(table)


@app.command()
def version() -> None:
    """Show version."""
    console.print(f"animus-mev v{__version__}")


def _load_config() -> MEVConfig:
    """Load config from default location."""
    config = MEVConfig()
    if config.config_file.exists():
        config = MEVConfig.from_yaml(config.config_file)
    config.ensure_dirs()
    return config


if __name__ == "__main__":
    app()  # pragma: no cover
