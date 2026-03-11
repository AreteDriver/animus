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

feedback_app = typer.Typer(help="Record and view feedback on Animus responses.")
app.add_typer(feedback_app, name="feedback")

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
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Name", style="bold")
    table.add_column("Tone")
    table.add_column("Domains", style="dim")
    table.add_column("Default")

    # Show default persona
    cfg = config.personas
    table.add_row(
        "(config)",
        cfg.default_name,
        cfg.default_tone,
        "General",
        "[green]Yes[/green]",
    )

    # Show profiles from config
    for name, profile in cfg.profiles.items():
        table.add_row(
            "(config)",
            profile.name or name,
            profile.tone,
            ", ".join(profile.knowledge_domains) if profile.knowledge_domains else "General",
            "[dim]No[/dim]",
        )

    # Show personas from storage
    storage = _get_persona_storage()
    if storage:
        for persona in storage.load_all():
            table.add_row(
                persona.id[:12],
                persona.name,
                persona.voice.tone,
                ", ".join(persona.knowledge_domains) if persona.knowledge_domains else "General",
                "[green]Yes[/green]" if persona.is_default else "[dim]No[/dim]",
            )
        storage.close()

    console.print()
    console.print(table)
    console.print()


def _get_persona_storage():
    """Get PersonaStorage instance from config path, or None if unavailable."""
    try:
        from animus_bootstrap.config import ConfigManager
        from animus_bootstrap.personas.storage import PersonaStorage

        config_path = ConfigManager().get_config_path().parent
        db_path = config_path / "personas.db"
        return PersonaStorage(db_path)
    except Exception:
        return None


@personas_app.command("add")
def personas_add(
    name: str = typer.Argument(help="Persona name"),
    description: str = typer.Option("", "--description", "-d", help="Short description"),
    tone: str = typer.Option(
        "balanced",
        "--tone",
        "-t",
        help="Voice tone (formal/casual/technical/mentor/creative/balanced)",
    ),
    domains: str = typer.Option("", "--domains", help="Comma-separated knowledge domains"),
    system_prompt: str = typer.Option("", "--prompt", "-p", help="System prompt"),
    default: bool = typer.Option(False, "--default", help="Set as default persona"),
) -> None:
    """Create a new persona profile."""
    from animus_bootstrap.personas.engine import PersonaProfile
    from animus_bootstrap.personas.voice import VoiceConfig

    valid_tones = ("formal", "casual", "technical", "mentor", "creative", "balanced")
    if tone not in valid_tones:
        console.print(f"[red]Invalid tone: {tone}. Must be one of: {', '.join(valid_tones)}[/red]")
        raise typer.Exit(1)

    voice = VoiceConfig(tone=tone)
    domain_list = [d.strip() for d in domains.split(",") if d.strip()] if domains else []

    persona = PersonaProfile(
        name=name,
        description=description or f"{name} persona",
        system_prompt=system_prompt or f"You are {name}, a personal AI assistant.",
        voice=voice,
        knowledge_domains=domain_list,
        is_default=default,
    )

    storage = _get_persona_storage()
    storage.save(persona)
    storage.close()

    console.print(f"[green]Created persona '{name}' ({persona.id[:12]}...)[/green]")
    if domain_list:
        console.print(f"  Domains: {', '.join(domain_list)}")
    if default:
        console.print("  [cyan]Set as default[/cyan]")


@personas_app.command("delete")
def personas_delete(
    name: str = typer.Argument(help="Persona name to delete"),
) -> None:
    """Delete a persona profile."""
    storage = _get_persona_storage()
    personas = storage.load_all()

    target = None
    for p in personas:
        if p.name.lower() == name.lower():
            target = p
            break

    if not target:
        storage.close()
        console.print(f"[red]Persona '{name}' not found[/red]")
        raise typer.Exit(1)

    storage.delete(target.id)
    storage.close()
    console.print(f"[yellow]Deleted persona '{target.name}'[/yellow]")


@personas_app.command("set-default")
def personas_set_default(
    name: str = typer.Argument(help="Persona name to set as default"),
) -> None:
    """Set a persona as the default."""
    storage = _get_persona_storage()
    personas = storage.load_all()

    target = None
    for p in personas:
        if p.name.lower() == name.lower():
            target = p
            break

    if not target:
        storage.close()
        console.print(f"[red]Persona '{name}' not found[/red]")
        raise typer.Exit(1)

    # Clear old defaults
    for p in personas:
        if p.is_default and p.id != target.id:
            p.is_default = False
            storage.save(p)

    target.is_default = True
    storage.save(target)
    storage.close()
    console.print(f"[green]'{target.name}' is now the default persona[/green]")


# ------------------------------------------------------------------
# animus-bootstrap reflect
# ------------------------------------------------------------------


@app.command()
def reflect(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Run a reflection cycle — analyze feedback and update LEARNED.md."""
    import asyncio

    _setup_logging(verbose)

    from animus_bootstrap.config import ConfigManager
    from animus_bootstrap.identity.manager import IdentityFileManager
    from animus_bootstrap.intelligence.feedback import FeedbackStore
    from animus_bootstrap.intelligence.proactive.checks.reflection import (
        _run_reflection,
        set_reflection_deps,
    )

    config_manager = ConfigManager()
    config = config_manager.load()

    # Wire dependencies
    identity_dir = config_manager.get_config_path().parent / "identity"
    identity_manager = IdentityFileManager(identity_dir)

    db_path = config_manager.get_config_path().parent / "feedback.db"
    feedback_store = FeedbackStore(db_path) if db_path.exists() else None

    # Try to create cognitive backend for AI-powered reflection
    cognitive_backend = None
    try:
        api_key = config.api.anthropic_key
        if api_key:
            from animus_bootstrap.gateway.backends import AnthropicBackend

            cognitive_backend = AnthropicBackend(api_key=api_key)
    except Exception:
        logger.debug("Anthropic backend unavailable, trying Ollama")

    if cognitive_backend is None:
        try:
            from animus_bootstrap.gateway.backends import OllamaBackend

            cognitive_backend = OllamaBackend(
                host=getattr(config.api, "ollama_host", "http://localhost:11434"),
                model=getattr(config.api, "ollama_model", "llama3.2"),
            )
        except Exception:
            logger.debug("Ollama backend unavailable, no cognitive backend")

    set_reflection_deps(
        identity_manager=identity_manager,
        feedback_store=feedback_store,
        cognitive_backend=cognitive_backend,
        config=config,
    )

    console.print("[cyan]Running reflection...[/cyan]")
    result = asyncio.run(_run_reflection())

    if result:
        console.print(f"[green]{result}[/green]")
    else:
        console.print("[dim]No reflection needed — no feedback data yet.[/dim]")
        console.print("[dim]Use 'animus-bootstrap feedback add' to record feedback first.[/dim]")

    if feedback_store:
        feedback_store.close()


# ------------------------------------------------------------------
# animus-bootstrap feedback add / list / stats
# ------------------------------------------------------------------


@feedback_app.command("add")
def feedback_add(
    rating: str = typer.Argument(help="Rating: 'up' (positive) or 'down' (negative)"),
    message: str = typer.Option("", "--message", "-m", help="The message/question that was asked"),
    response: str = typer.Option("", "--response", "-r", help="The response that was given"),
    comment: str = typer.Option("", "--comment", "-c", help="Why this was good/bad"),
) -> None:
    """Record feedback on an Animus response."""
    from animus_bootstrap.config import ConfigManager
    from animus_bootstrap.intelligence.feedback import FeedbackStore

    if rating not in ("up", "down"):
        console.print("[red]Rating must be 'up' or 'down'[/red]")
        raise typer.Exit(1)

    config_path = ConfigManager().get_config_path().parent
    db_path = config_path / "feedback.db"
    store = FeedbackStore(db_path)

    numeric_rating = 1 if rating == "up" else -1
    feedback_id = store.record(
        message_text=message or "(not provided)",
        response_text=response or "(not provided)",
        rating=numeric_rating,
        comment=comment,
        channel="cli",
    )
    store.close()

    icon = "[green]thumbs up[/green]" if rating == "up" else "[red]thumbs down[/red]"
    console.print(f"Recorded {icon} feedback ({feedback_id[:8]}...)")


@feedback_app.command("list")
def feedback_list(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of entries to show"),
) -> None:
    """Show recent feedback entries."""
    from animus_bootstrap.config import ConfigManager
    from animus_bootstrap.intelligence.feedback import FeedbackStore

    config_path = ConfigManager().get_config_path().parent
    db_path = config_path / "feedback.db"

    if not db_path.exists():
        console.print("[dim]No feedback recorded yet.[/dim]")
        return

    store = FeedbackStore(db_path)
    entries = store.get_recent(limit=limit)
    store.close()

    if not entries:
        console.print("[dim]No feedback entries found.[/dim]")
        return

    table = Table(title="Recent Feedback", border_style="cyan")
    table.add_column("Time", style="dim", width=16)
    table.add_column("Rating", justify="center", width=6)
    table.add_column("Message", max_width=40)
    table.add_column("Comment", style="dim", max_width=30)

    for entry in entries:
        ts = entry["timestamp"][:16].replace("T", " ")
        icon = "[green]+1[/green]" if entry["rating"] > 0 else "[red]-1[/red]"
        msg = entry["message_text"][:40]
        cmt = entry.get("comment", "")[:30]
        table.add_row(ts, icon, msg, cmt)

    console.print()
    console.print(table)
    console.print()


@feedback_app.command("stats")
def feedback_stats() -> None:
    """Show feedback statistics."""
    from animus_bootstrap.config import ConfigManager
    from animus_bootstrap.intelligence.feedback import FeedbackStore

    config_path = ConfigManager().get_config_path().parent
    db_path = config_path / "feedback.db"

    if not db_path.exists():
        console.print("[dim]No feedback recorded yet.[/dim]")
        return

    store = FeedbackStore(db_path)
    stats = store.get_stats()
    store.close()

    table = Table(title="Feedback Stats", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Total", str(stats["total"]))
    table.add_row("Positive", f"[green]{stats['positive']}[/green] ({stats['positive_pct']}%)")
    table.add_row("Negative", f"[red]{stats['negative']}[/red] ({stats['negative_pct']}%)")

    console.print()
    console.print(table)
    console.print()
