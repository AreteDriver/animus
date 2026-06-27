"""CLI for Animus Arbitrage."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from animus_arbitrage.config import ArbitrageConfig
from animus_arbitrage.detector import ArbitrageDetector
from animus_arbitrage.executor import TradeExecutor
from animus_arbitrage.risk import RiskManager
from animus_arbitrage.store import ArbitrageStore
from animus_arbitrage.tracker import PnLTracker

app = typer.Typer(name="animus-arbitrage", help="DEX arbitrage detection and execution")
console = Console()


def _load_config(config_path: Path | None = None) -> ArbitrageConfig:
    """Load configuration from file or defaults."""
    if config_path and config_path.exists():
        return ArbitrageConfig.from_yaml(config_path)
    config = ArbitrageConfig()
    if config.config_file.exists():
        return ArbitrageConfig.from_yaml(config.config_file)
    return config


@app.command()
def scan(
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    trade_size: Annotated[float, typer.Option("--size", "-s")] = 0.0,
) -> None:
    """Scan for arbitrage opportunities."""
    config = _load_config(config_path)
    config.ensure_dirs()

    detector = ArbitrageDetector(config)
    opportunities = asyncio.run(detector.scan(trade_size_usd=trade_size))

    if not opportunities:
        console.print("[yellow]No arbitrage opportunities found.[/yellow]")
        return

    table = Table(title="Arbitrage Opportunities")
    table.add_column("ID", style="dim")
    table.add_column("Pair")
    table.add_column("Buy DEX")
    table.add_column("Sell DEX")
    table.add_column("Spread %", justify="right")
    table.add_column("Profit $", justify="right")
    table.add_column("Confidence", justify="right")

    for opp in opportunities:
        table.add_row(
            opp.opportunity_id[:8],
            opp.buy_quote.pair.symbol,
            opp.buy_quote.dex.value,
            opp.sell_quote.dex.value,
            f"{opp.spread_pct:.2f}%",
            f"${opp.profit_after_gas_usd:.4f}",
            f"{opp.confidence:.2f}",
        )

    console.print(table)


@app.command()
def status(
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Show P&L and risk status."""
    config = _load_config(config_path)
    config.ensure_dirs()

    store = ArbitrageStore(config.data_dir)
    tracker = PnLTracker(store)
    risk = RiskManager(config)

    console.print(tracker.format_report())
    console.print()

    risk_status = risk.status
    console.print("[bold]Risk Status:[/bold]")
    for key, value in risk_status.items():
        console.print(f"  {key}: {value}")


@app.command(name="config")
def show_config(
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Show current configuration."""
    config = _load_config(config_path)
    console.print_json(data=config.to_dict())


@app.command()
def execute(
    opportunity_id: str,
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Execute a specific arbitrage opportunity."""
    config = _load_config(config_path)
    config.ensure_dirs()

    if not config.execution.is_live:
        console.print(
            "[red]Execution blocked: mode is DRY_RUN.[/red]\n"
            "Set execution.mode = 'live' in config to enable."
        )
        raise typer.Exit(1)

    store = ArbitrageStore(config.data_dir)
    risk = RiskManager(config)
    executor = TradeExecutor(config, risk)
    tracker = PnLTracker(store)

    opportunities = store.load_opportunities()
    target = None
    for opp in opportunities:
        if opp.opportunity_id == opportunity_id:
            target = opp
            break

    if target is None:
        console.print(f"[red]Opportunity {opportunity_id} not found.[/red]")
        raise typer.Exit(1)

    result = asyncio.run(executor.execute(target))
    tracker.record_result(result)

    if result.success:
        console.print(f"[green]Trade executed. Profit: ${result.actual_profit_usd:.4f}[/green]")
    else:
        console.print(f"[red]Trade failed: {result.error}[/red]")


if __name__ == "__main__":
    app()
