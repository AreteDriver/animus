"""CLI for Animus Referral."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from animus_referral.config import ReferralConfig
from animus_referral.manager import ReferralManager
from animus_referral.models import Account, BonusStatus, ProgramType, ReferralBonus
from animus_referral.tracker import ReferralTracker

app = typer.Typer(name="animus-referral", help="Autonomous referral program stacking")
console = Console()


def _get_manager(data_dir: Path | None = None) -> ReferralManager:
    config = ReferralConfig()
    if data_dir:
        config.data_dir = data_dir
    config.ensure_dirs()
    if config.config_file.exists():
        config = ReferralConfig.from_yaml(config.config_file)
        config.ensure_dirs()
    return ReferralManager(config)


@app.command()
def programs(
    program_type: str | None = typer.Option(None, "--type", "-t", help="Filter by type"),
    all_programs: bool = typer.Option(False, "--all", "-a", help="Include inactive"),
) -> None:
    """List known referral programs."""
    manager = _get_manager()
    ptype = ProgramType(program_type) if program_type else None
    progs = manager.get_programs(program_type=ptype, active_only=not all_programs)

    table = Table(title="Referral Programs")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Referrer $", style="yellow", justify="right")
    table.add_column("Referee $", style="yellow", justify="right")
    table.add_column("Bonus Type")
    table.add_column("Active", justify="center")

    for p in progs:
        table.add_row(
            p.name,
            p.program_type.value,
            f"${p.bonus_referrer_usd:.2f}",
            f"${p.bonus_referee_usd:.2f}",
            p.bonus_type.value,
            "[green]Yes[/green]" if p.active else "[red]No[/red]",
        )

    console.print(table)


@app.command()
def status(
    days: int = typer.Option(30, "--days", "-d", help="Report period in days"),
) -> None:
    """Show referral earnings summary."""
    manager = _get_manager()
    tracker = ReferralTracker(manager.config.data_dir)
    report = tracker.get_report(days=days)
    summary = manager.get_summary()

    console.print(f"\n[bold]Referral Summary[/bold] (last {days} days)\n")
    console.print(f"  Accounts:       {summary['total_accounts']}")
    console.print(
        f"  Programs:       {summary['active_programs']} active / {summary['total_programs']} total"
    )
    console.print(f"  Links:          {summary['total_links']}")
    console.print(f"  Signups:        {summary['total_signups']}")
    console.print(f"  Earned:         [green]${report['total_earned_usd']:.2f}[/green]")
    console.print(f"  Pending:        [yellow]${report['total_pending_usd']:.2f}[/yellow]")
    console.print(
        f"  All-time:       [bold green]${report['all_time_earned_usd']:.2f}[/bold green]"
    )
    console.print(f"  Conversion:     {report['conversion_rate']:.1%}")

    if report["top_programs"]:
        console.print("\n[bold]Top Programs:[/bold]")
        for name, earned in report["top_programs"]:
            console.print(f"  {name}: ${earned:.2f}")


@app.command()
def accounts(
    platform: str | None = typer.Option(None, "--platform", "-p", help="Filter by platform"),
) -> None:
    """List registered accounts."""
    manager = _get_manager()
    accts = manager.get_accounts(platform=platform)

    table = Table(title="Accounts")
    table.add_column("Alias", style="cyan")
    table.add_column("Platform", style="green")
    table.add_column("Referral Code")
    table.add_column("Referred By")

    for a in accts:
        table.add_row(a.alias, a.platform, a.referral_code or "-", a.referred_by or "-")

    console.print(table)


@app.command(name="add-account")
def add_account(
    alias: str = typer.Argument(help="Account alias"),
    platform: str = typer.Argument(help="Platform name"),
    referral_code: str = typer.Option("", "--code", "-c", help="Referral code"),
    referred_by: str = typer.Option("", "--referred-by", "-r", help="Referrer alias"),
    wallet: str = typer.Option("", "--wallet", "-w", help="Wallet address"),
) -> None:
    """Register an account."""
    manager = _get_manager()
    account = Account(
        alias=alias,
        platform=platform,
        referral_code=referral_code,
        referred_by=referred_by,
        wallet_address=wallet,
    )
    manager.add_account(account)
    console.print(f"[green]Added account {alias} on {platform}[/green]")


@app.command(name="gen-link")
def gen_link(
    program: str = typer.Argument(help="Program name"),
    referrer: str = typer.Argument(help="Referrer alias"),
    code: str = typer.Argument(help="Referral code"),
) -> None:
    """Generate a referral link."""
    manager = _get_manager()
    link = manager.generate_link(program, referrer, code)
    if link:
        console.print(f"[green]{link.link_url}[/green]")
    else:
        console.print(f"[red]Program '{program}' not found[/red]")


@app.command()
def graph() -> None:
    """Show the referral graph."""
    manager = _get_manager()
    g = manager.get_referral_graph()

    if not g:
        console.print("[dim]No referral relationships found[/dim]")
        return

    console.print("\n[bold]Referral Graph:[/bold]")
    for referrer, referees in g.items():
        for referee in referees:
            console.print(f"  {referrer} -> {referee}")


@app.command(name="record-bonus")
def record_bonus(
    program: str = typer.Argument(help="Program name"),
    account: str = typer.Argument(help="Account alias"),
    amount: float = typer.Argument(help="Bonus amount"),
    amount_usd: float = typer.Option(0.0, "--usd", help="USD value"),
    token: str = typer.Option("USD", "--token", "-t", help="Token name"),
    paid: bool = typer.Option(False, "--paid", help="Mark as paid immediately"),
) -> None:
    """Record a referral bonus."""
    manager = _get_manager()
    bonus = ReferralBonus(
        program_name=program,
        account_alias=account,
        amount=amount,
        amount_usd=amount_usd,
        token=token,
        status=BonusStatus.PAID if paid else BonusStatus.PENDING,
    )
    manager.record_bonus(bonus)
    status_str = "paid" if paid else "pending"
    console.print(f"[green]Recorded {amount} {token} ({status_str}) for {account}[/green]")


if __name__ == "__main__":
    app()
