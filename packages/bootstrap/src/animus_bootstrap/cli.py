"""CLI entry points for Animus Bootstrap.

Commands:
    animus-bootstrap install   — Full install: check deps, register service, run wizard
    animus-bootstrap setup     — Re-run the onboarding wizard
    animus-bootstrap start     — Start the Animus system service
    animus-bootstrap stop      — Stop the Animus system service
    animus-bootstrap status    — Show current system status
    animus-bootstrap update    — Check for and apply updates
    animus-bootstrap dashboard — Open the dashboard in a browser
    animus-bootstrap config    — Get/set individual config values
    animus-bootstrap channels  — List, enable, or disable messaging channels
"""

from __future__ import annotations

import logging
import webbrowser

import typer
from rich.console import Console
from rich.table import Table

import animus_bootstrap

console = Console()
app = typer.Typer(
    name="animus-bootstrap",
    help="Animus install daemon, onboarding wizard, and local dashboard.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Get or set configuration values.")
app.add_typer(config_app, name="config")

channels_app = typer.Typer(help="List, enable, or disable messaging channels.")
app.add_typer(channels_app, name="channels")

tools_app = typer.Typer(help="Manage registered tools.")
app.add_typer(tools_app, name="tools")

proactive_app = typer.Typer(help="Manage proactive engine.")
app.add_typer(proactive_app, name="proactive")

automations_app = typer.Typer(help="Manage automation rules.")
app.add_typer(automations_app, name="automations")

personas_app = typer.Typer(help="Manage persona profiles.")
app.add_typer(personas_app, name="personas")

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


# ------------------------------------------------------------------
# animus-bootstrap install
# ------------------------------------------------------------------


@app.command()
def install(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    skip_wizard: bool = typer.Option(False, "--skip-wizard", help="Skip onboarding wizard"),
) -> None:
    """Full install: check deps, register service, run wizard, start dashboard."""
    _setup_logging(verbose)
    from animus_bootstrap.config import ConfigManager
    from animus_bootstrap.daemon.installer import AnimusInstaller

    installer = AnimusInstaller()
    config_manager = ConfigManager()

    # Step 1: Check dependencies
    console.print("\n[bold cyan]Checking dependencies...[/bold cyan]")
    deps = installer.check_dependencies()
    for dep, found in deps.items():
        icon = "[green]OK[/green]" if found else "[red]MISSING[/red]"
        console.print(f"  {dep}: {icon}")

    # Step 2: Install missing required deps
    missing_required = {k: v for k, v in deps.items() if not v and k in ("python3", "pip")}
    if missing_required:
        console.print("\n[yellow]Installing missing dependencies...[/yellow]")
        installed = installer.install_missing_deps(deps)
        if installed:
            console.print(f"  Installed: {', '.join(installed)}")

    # Step 3: Register system service
    console.print("\n[bold cyan]Registering system service...[/bold cyan]")
    if installer.register_service():
        console.print("  [green]Service registered.[/green]")
    else:
        console.print("  [yellow]Service registration skipped (may need manual setup).[/yellow]")

    # Step 4: Run wizard if first run
    if not skip_wizard and (not config_manager.exists() or config_manager.load().animus.first_run):
        from animus_bootstrap.setup.wizard import AnimusWizard

        wizard = AnimusWizard(config_manager)
        wizard.run()
    elif not skip_wizard:
        msg = "Config already exists. Use 'animus-bootstrap setup' to re-run wizard."
        console.print(f"\n[dim]{msg}[/dim]")

    # Step 5: Start service
    console.print("\n[bold cyan]Starting Animus...[/bold cyan]")
    if installer.start_service():
        console.print("  [green]Animus service started.[/green]")
    else:
        msg = "Could not start service. Run 'animus-bootstrap dashboard' manually."
        console.print(f"  [yellow]{msg}[/yellow]")

    # Step 6: Open dashboard
    dashboard_url = f"http://localhost:{config_manager.load().services.port}"
    console.print(f"\n[bold green]Dashboard available at {dashboard_url}[/bold green]")
    try:
        webbrowser.open(dashboard_url)
    except Exception:  # noqa: BLE001
        console.print("[dim]Could not open browser automatically.[/dim]")


# ------------------------------------------------------------------
# animus-bootstrap setup
# ------------------------------------------------------------------


@app.command()
def setup(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Re-run the onboarding wizard."""
    _setup_logging(verbose)
    from animus_bootstrap.config import ConfigManager
    from animus_bootstrap.setup.wizard import AnimusWizard

    config_manager = ConfigManager()
    wizard = AnimusWizard(config_manager)
    wizard.run()


# ------------------------------------------------------------------
# animus-bootstrap start / stop
# ------------------------------------------------------------------


@app.command()
def start() -> None:
    """Start the Animus system service."""
    from animus_bootstrap.daemon.installer import AnimusInstaller

    installer = AnimusInstaller()
    if installer.start_service():
        console.print("[green]Animus service started.[/green]")
    else:
        console.print("[red]Failed to start Animus service.[/red]")
        raise typer.Exit(1)


@app.command()
def stop() -> None:
    """Stop the Animus system service."""
    from animus_bootstrap.daemon.installer import AnimusInstaller

    installer = AnimusInstaller()
    if installer.stop_service():
        console.print("[green]Animus service stopped.[/green]")
    else:
        console.print("[red]Failed to stop Animus service.[/red]")
        raise typer.Exit(1)


# ------------------------------------------------------------------
# animus-bootstrap status
# ------------------------------------------------------------------


@app.command()
def status() -> None:
    """Show current system status."""
    import httpx

    from animus_bootstrap.config import ConfigManager
    from animus_bootstrap.daemon.installer import AnimusInstaller

    installer = AnimusInstaller()
    config_manager = ConfigManager()
    config = config_manager.load()

    table = Table(title="Animus Status", border_style="cyan")
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Details", style="dim")

    # Version
    table.add_row("Version", animus_bootstrap.__version__, "")

    # Daemon
    daemon_running = installer.is_service_running()
    daemon_status = "[green]Running[/green]" if daemon_running else "[red]Stopped[/red]"
    table.add_row("Daemon", daemon_status, installer.detect_os())

    # Forge
    forge_status = "[dim]Disabled[/dim]"
    forge_detail = ""
    if config.forge.enabled:
        try:
            resp = httpx.get(
                f"http://{config.forge.host}:{config.forge.port}/health",
                timeout=2,
            )
            if resp.status_code == 200:
                forge_status = "[green]Connected[/green]"
            else:
                forge_status = "[yellow]Unhealthy[/yellow]"
            forge_detail = f"{config.forge.host}:{config.forge.port}"
        except httpx.RequestError:
            forge_status = "[red]Unreachable[/red]"
            forge_detail = f"{config.forge.host}:{config.forge.port}"
    table.add_row("Forge", forge_status, forge_detail)

    # Memory
    memory_path = config.get_memory_path()
    if memory_path.exists():
        size_mb = memory_path.stat().st_size / (1024 * 1024)
        table.add_row("Memory", f"[green]{config.memory.backend}[/green]", f"{size_mb:.1f} MB")
    else:
        table.add_row("Memory", f"[dim]{config.memory.backend}[/dim]", "No data yet")

    # Config
    config_exists = config_manager.exists()
    config_status = (
        "[green]Configured[/green]" if config_exists else "[yellow]Not configured[/yellow]"
    )
    table.add_row("Config", config_status, str(config_manager.get_config_path()))

    # Identity
    if config.identity.name:
        table.add_row("Identity", config.identity.name, config.identity.timezone)
    else:
        table.add_row("Identity", "[dim]Not set[/dim]", "")

    console.print()
    console.print(table)
    console.print()


# ------------------------------------------------------------------
# animus-bootstrap update
# ------------------------------------------------------------------


@app.command()
def update() -> None:
    """Check for and apply updates."""
    from animus_bootstrap.daemon.updater import AnimusUpdater

    updater = AnimusUpdater()
    current = updater.get_current_version()
    console.print(f"Current version: [cyan]{current}[/cyan]")

    with console.status("Checking for updates..."):
        if not updater.is_update_available():
            console.print("[green]Already up to date.[/green]")
            return

    latest = updater.get_latest_version()
    console.print(f"Update available: [cyan]{current}[/cyan] -> [green]{latest}[/green]")

    if typer.confirm("Apply update now?"):
        with console.status("Updating..."):
            if updater.apply_update():
                console.print("[green]Update applied successfully.[/green]")
            else:
                console.print("[red]Update failed. Check logs for details.[/red]")
                raise typer.Exit(1)


# ------------------------------------------------------------------
# animus-bootstrap dashboard
# ------------------------------------------------------------------


@app.command()
def dashboard() -> None:
    """Open the local dashboard in a browser and start the server."""
    from animus_bootstrap.config import ConfigManager

    config_manager = ConfigManager()
    config = config_manager.load()
    port = config.services.port
    url = f"http://localhost:{port}"

    console.print(f"Starting dashboard at [cyan]{url}[/cyan]...")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass

    from animus_bootstrap.dashboard.app import serve

    serve()


# ------------------------------------------------------------------
# animus-bootstrap config get / set
# ------------------------------------------------------------------


@config_app.command("get")
def config_get(
    key: str = typer.Argument(help="Config key in dot notation (e.g., 'api.anthropic_key')"),
) -> None:
    """Print a config value."""
    from animus_bootstrap.config import ConfigManager

    config_manager = ConfigManager()
    config = config_manager.load()
    data = config.model_dump()

    parts = key.split(".")
    value: object = data
    for part in parts:
        if isinstance(value, dict):
            if part not in value:
                console.print(f"[red]Key not found: {key}[/red]")
                raise typer.Exit(1)
            value = value[part]
        else:
            console.print(f"[red]Key not found: {key}[/red]")
            raise typer.Exit(1)

    # Mask API keys
    if "key" in key.lower() and isinstance(value, str) and value:
        masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
        console.print(masked)
    else:
        console.print(str(value))


@config_app.command("set")
def config_set(
    key: str = typer.Argument(help="Config key in dot notation (e.g., 'identity.name')"),
    value: str = typer.Argument(help="Value to set"),
) -> None:
    """Update a single config value."""
    from animus_bootstrap.config import ConfigManager

    config_manager = ConfigManager()
    config = config_manager.load()
    data = config.model_dump()

    parts = key.split(".")
    if len(parts) != 2:
        console.print("[red]Key must be in 'section.field' format (e.g., 'identity.name').[/red]")
        raise typer.Exit(1)

    section, field = parts
    if section not in data or not isinstance(data[section], dict):
        console.print(f"[red]Unknown section: {section}[/red]")
        raise typer.Exit(1)

    if field not in data[section]:
        console.print(f"[red]Unknown field: {field} in section {section}[/red]")
        raise typer.Exit(1)

    # Type coercion
    existing = data[section][field]
    if isinstance(existing, bool):
        coerced: object = value.lower() in ("true", "1", "yes")
    elif isinstance(existing, int):
        try:
            coerced = int(value)
        except ValueError:
            console.print(f"[red]Expected integer for {key}[/red]")
            raise typer.Exit(1) from None
    else:
        coerced = value

    data[section][field] = coerced
    from animus_bootstrap.config.schema import AnimusConfig

    new_config = AnimusConfig(**data)
    config_manager.save(new_config)
    console.print(f"[green]{key} = {coerced}[/green]")


# ------------------------------------------------------------------
# animus-bootstrap channels list / enable / disable
# ------------------------------------------------------------------

_ALL_CHANNELS = (
    "webchat",
    "telegram",
    "discord",
    "slack",
    "matrix",
    "signal",
    "whatsapp",
    "email",
)


@channels_app.command("list")
def channels_list() -> None:
    """Show configured channels and their status."""
    from animus_bootstrap.config import ConfigManager

    config_manager = ConfigManager()
    config = config_manager.load()

    table = Table(title="Messaging Channels", border_style="cyan")
    table.add_column("Channel", style="bold")
    table.add_column("Status")

    for name in _ALL_CHANNELS:
        channel_cfg = getattr(config.channels, name, None)
        if channel_cfg is None:
            continue
        enabled = getattr(channel_cfg, "enabled", False)
        status = "[green]Enabled[/green]" if enabled else "[dim]Disabled[/dim]"
        table.add_row(name, status)

    console.print()
    console.print(table)
    console.print()


@channels_app.command("enable")
def channels_enable(
    name: str = typer.Argument(help="Channel name to enable"),
) -> None:
    """Enable a messaging channel."""
    from animus_bootstrap.config import ConfigManager

    config_manager = ConfigManager()
    config = config_manager.load()

    if name not in _ALL_CHANNELS:
        console.print(f"[red]Unknown channel: {name}[/red]")
        console.print(f"[dim]Available: {', '.join(_ALL_CHANNELS)}[/dim]")
        raise typer.Exit(1)

    channel_cfg = getattr(config.channels, name)
    channel_cfg.enabled = True
    config_manager.save(config)
    console.print(f"[green]Channel '{name}' enabled.[/green]")


@channels_app.command("disable")
def channels_disable(
    name: str = typer.Argument(help="Channel name to disable"),
) -> None:
    """Disable a messaging channel."""
    from animus_bootstrap.config import ConfigManager

    config_manager = ConfigManager()
    config = config_manager.load()

    if name not in _ALL_CHANNELS:
        console.print(f"[red]Unknown channel: {name}[/red]")
        console.print(f"[dim]Available: {', '.join(_ALL_CHANNELS)}[/dim]")
        raise typer.Exit(1)

    channel_cfg = getattr(config.channels, name)
    channel_cfg.enabled = False
    config_manager.save(config)
    console.print(f"[yellow]Channel '{name}' disabled.[/yellow]")


# ------------------------------------------------------------------
# animus-bootstrap tools list
# ------------------------------------------------------------------


@tools_app.command("list")
def tools_list() -> None:
    """List all registered tools."""
    table = Table(title="Registered Tools", border_style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Permission", style="dim")

    console.print()
    console.print(table)
    console.print("[dim]No tools loaded. Start the intelligence layer to register tools.[/dim]")
    console.print()


# ------------------------------------------------------------------
# animus-bootstrap proactive status
# ------------------------------------------------------------------


@proactive_app.command("status")
def proactive_status() -> None:
    """Show proactive engine status."""
    from animus_bootstrap.config import ConfigManager

    config_manager = ConfigManager()
    config = config_manager.load()

    table = Table(title="Proactive Engine", border_style="cyan")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    enabled = config.proactive.enabled
    status_str = "[green]Enabled[/green]" if enabled else "[dim]Disabled[/dim]"
    table.add_row("Status", status_str)
    quiet = f"{config.proactive.quiet_hours_start} - {config.proactive.quiet_hours_end}"
    table.add_row("Quiet Hours", quiet)
    table.add_row("Timezone", config.proactive.timezone)

    checks = config.proactive.checks
    table.add_row("Checks", str(len(checks)) if checks else "0")

    console.print()
    console.print(table)
    console.print()


# ------------------------------------------------------------------
# animus-bootstrap automations list
# ------------------------------------------------------------------


@automations_app.command("list")
def automations_list() -> None:
    """List all automation rules."""
    table = Table(title="Automation Rules", border_style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Trigger")
    table.add_column("Enabled")
    table.add_column("Last Fired", style="dim")

    console.print()
    console.print(table)
    console.print("[dim]No automation rules configured.[/dim]")
    console.print()


# ------------------------------------------------------------------
# animus-bootstrap personas list
# ------------------------------------------------------------------


@personas_app.command("list")
def personas_list() -> None:
    """List all persona profiles."""
    from animus_bootstrap.config import ConfigManager

    config = ConfigManager().load()

    table = Table(title="Persona Profiles", border_style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Tone")
    table.add_column("Domains", style="dim")
    table.add_column("Default")

    # Show default persona
    cfg = config.personas
    table.add_row(
        cfg.default_name,
        cfg.default_tone,
        "General",
        "[green]Yes[/green]",
    )

    # Show profiles
    for name, profile in cfg.profiles.items():
        table.add_row(
            profile.name or name,
            profile.tone,
            ", ".join(profile.knowledge_domains) if profile.knowledge_domains else "General",
            "[dim]No[/dim]",
        )

    console.print()
    console.print(table)
    console.print()
