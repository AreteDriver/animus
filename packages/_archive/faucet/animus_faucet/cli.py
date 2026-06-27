"""CLI for Animus Faucet."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from animus_faucet.config import FaucetConfig
from animus_faucet.models import Chain
from animus_faucet.scheduler import FaucetScheduler
from animus_faucet.store import ClaimStore
from animus_faucet.wallet.manager import WalletManager

console = Console()
app = typer.Typer(
    name="animus-faucet",
    help="Autonomous crypto faucet farming for Animus.",
    no_args_is_help=True,
)


def _load_config() -> FaucetConfig:
    config = FaucetConfig()
    if config.faucets_file.exists():
        config = FaucetConfig.from_yaml(config.faucets_file)
    config.ensure_dirs()
    return config


@app.command()
def claim(
    faucet_name: str | None = typer.Argument(None, help="Specific faucet (omit for all)"),
    no_jitter: bool = typer.Option(False, "--no-jitter", help="Skip random delay"),
) -> None:
    """Claim from faucets now."""
    config = _load_config()
    if no_jitter:
        config.jitter_seconds = 0

    wm = WalletManager(config.data_dir)
    scheduler = FaucetScheduler(config, wm)

    if faucet_name:
        faucet = next((f for f in config.faucets if f.name == faucet_name), None)
        if faucet is None:
            console.print(f"[red]Faucet '{faucet_name}' not found[/red]")
            raise typer.Exit(1)
        result = asyncio.run(scheduler.claim_faucet(faucet))
        if result.succeeded:
            console.print(f"[green]Success[/green]: {result.amount or '?'} from {faucet_name}")
        else:
            console.print(f"[red]{result.status.value}[/red]: {result.error_detail}")
    else:
        results = asyncio.run(scheduler.run_once())
        succeeded = sum(1 for r in results if r.succeeded)
        console.print(f"\nResults: [green]{succeeded}[/green]/{len(results)} succeeded")


@app.command()
def status() -> None:
    """Show faucet health and earnings."""
    config = _load_config()
    wm = WalletManager(config.data_dir)
    scheduler = FaucetScheduler(config, wm)
    summary = scheduler.get_summary()

    console.print("\n[bold]Faucet Status[/bold]")
    console.print(f"  Total faucets: {summary['total_faucets']}")
    console.print(f"  Enabled: {summary['enabled']}")
    console.print(f"  Total earned: [green]${summary['total_earned_usd']:.4f}[/green]")
    console.print(f"  Total claims: {summary['total_claims']}")

    if summary["faucets"]:
        table = Table(title="Faucet Health")
        table.add_column("Name")
        table.add_column("Enabled")
        table.add_column("Success Rate")
        table.add_column("Earned (USD)")
        table.add_column("Last Status")

        for f in summary["faucets"]:
            if f.get("no_data"):
                table.add_row(f["name"], str(f["enabled"]), "—", "—", "—")
            else:
                total = f.get("success_count", 0) + f.get("fail_count", 0)
                rate = f"{f['success_count'] / total * 100:.0f}%" if total > 0 else "—"
                table.add_row(
                    f["name"],
                    str(f["enabled"]),
                    rate,
                    f"${f.get('total_earned_usd', 0):.4f}",
                    f.get("last_status", "—"),
                )

        console.print(table)


@app.command()
def wallet(
    action: str = typer.Argument(..., help="add|list|remove"),
    chain: str | None = typer.Argument(None, help="Chain name"),
    address: str | None = typer.Argument(None, help="Wallet address"),
) -> None:
    """Manage wallet addresses."""
    config = _load_config()
    wm = WalletManager(config.data_dir)

    if action == "list":
        wallets = wm.list_wallets()
        if not wallets:
            console.print("No wallets configured.")
            return
        table = Table(title="Wallets")
        table.add_column("Chain")
        table.add_column("Address")
        for c, addr in wallets.items():
            table.add_row(c, addr)
        console.print(table)

    elif action == "add":
        if not chain or not address:
            console.print("[red]Usage: wallet add <chain> <address>[/red]")
            raise typer.Exit(1)
        try:
            c = Chain(chain)
        except ValueError:
            console.print(f"[red]Invalid chain. Valid: {[c.value for c in Chain]}[/red]")
            raise typer.Exit(1)
        wm.set_address(c, address)
        console.print(f"[green]Added[/green] {chain} → {address}")

    elif action == "remove":
        if not chain:
            console.print("[red]Usage: wallet remove <chain>[/red]")
            raise typer.Exit(1)
        try:
            c = Chain(chain)
        except ValueError:
            console.print("[red]Invalid chain.[/red]")
            raise typer.Exit(1)
        if wm.remove_address(c):
            console.print(f"[green]Removed[/green] {chain}")
        else:
            console.print(f"[yellow]No wallet for {chain}[/yellow]")

    else:
        console.print(f"[red]Unknown action: {action}. Use add|list|remove[/red]")


@app.command()
def earnings(
    days: int = typer.Option(30, "--days", "-d", help="Report period in days"),
) -> None:
    """Show earnings report."""
    config = _load_config()
    store = ClaimStore(config.data_dir)
    claims = store.load_all()

    if days:
        from datetime import UTC, datetime, timedelta

        cutoff = datetime.now(UTC) - timedelta(days=days)
        claims = [c for c in claims if c.timestamp >= cutoff]

    succeeded = [c for c in claims if c.succeeded]
    total_usd = sum(c.amount_usd for c in succeeded if c.amount_usd)

    console.print(f"\n[bold]Earnings Report ({days} days)[/bold]")
    console.print(f"  Claims: {len(succeeded)}/{len(claims)} succeeded")
    console.print(f"  Total earned: [green]${total_usd:.4f}[/green]")

    if succeeded:
        by_faucet: dict[str, float] = {}
        for c in succeeded:
            if c.amount_usd:
                by_faucet[c.faucet_name] = by_faucet.get(c.faucet_name, 0) + c.amount_usd

        table = Table(title="By Faucet")
        table.add_column("Faucet")
        table.add_column("Earned (USD)")
        for name, earned in sorted(by_faucet.items(), key=lambda x: -x[1]):
            table.add_row(name, f"${earned:.4f}")
        console.print(table)


@app.command()
def init() -> None:
    """Initialize faucet config with starter faucets."""
    config = _load_config()

    if config.faucets_file.exists():
        console.print(f"[yellow]Config already exists: {config.faucets_file}[/yellow]")
        return

    from animus_faucet.defaults import DEFAULT_FAUCETS_YAML

    config.faucets_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config.faucets_file, "w") as f:
        f.write(DEFAULT_FAUCETS_YAML)

    console.print(f"[green]Created[/green] {config.faucets_file}")
    console.print("Edit the config to add your wallet addresses, then run:")
    console.print("  [bold]animus-faucet claim[/bold]")
