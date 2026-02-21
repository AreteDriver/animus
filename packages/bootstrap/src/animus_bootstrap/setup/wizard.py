"""Animus onboarding wizard — orchestrates all setup steps."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from animus_bootstrap.config import AnimusConfig, ConfigManager
from animus_bootstrap.config.schema import (
    ApiSection,
    ForgeSection,
    IdentitySection,
    MemorySection,
    OllamaSection,
)
from animus_bootstrap.setup.steps.api_keys import run_api_keys
from animus_bootstrap.setup.steps.channels import run_channels_step
from animus_bootstrap.setup.steps.device import run_device
from animus_bootstrap.setup.steps.forge import run_forge
from animus_bootstrap.setup.steps.identity import run_identity
from animus_bootstrap.setup.steps.identity_files import run_identity_files
from animus_bootstrap.setup.steps.memory import run_memory
from animus_bootstrap.setup.steps.sovereignty import run_sovereignty
from animus_bootstrap.setup.steps.welcome import run_welcome

_TOTAL_STEPS = 9


class AnimusWizard:
    """Interactive onboarding wizard for first-time Animus setup.

    Orchestrates all setup steps in sequence, collecting configuration
    from the user and persisting it to disk via :class:`ConfigManager`.

    Args:
        config_manager: The ConfigManager instance for reading/writing config.
    """

    def __init__(self, config_manager: ConfigManager) -> None:
        self._config_manager = config_manager
        self._console = Console()

    def run(self) -> AnimusConfig:
        """Run the full onboarding wizard.

        Executes each step in order:
          1. Welcome
          2. Identity
          3. API Keys
          4. Forge
          5. Memory
          6. Device
          7. Sovereignty
          8. Channels

        On step failure, the user can retry or skip (except API Keys,
        which is required). On completion, writes the config and shows
        a summary panel.

        Returns:
            The assembled and persisted :class:`AnimusConfig`.
        """
        console = self._console

        # Step 1: Welcome
        self._step_header(1, "Welcome")
        if not run_welcome(console):
            console.print("[dim]Setup cancelled.[/dim]")
            raise SystemExit(0)

        # Step 2: Identity
        identity_data = self._run_step(2, "Identity", run_identity, skippable=True)

        # Step 3: Identity Files
        identity_files_data = self._run_step(
            3, "Identity Files", run_identity_files, skippable=True
        )

        # Step 4: API Keys (required — not skippable)
        api_data = self._run_step(4, "API Keys", run_api_keys, skippable=False)

        # Step 5: Forge
        forge_data = self._run_step(5, "Forge", run_forge, skippable=True)

        # Step 6: Memory
        memory_data = self._run_step(6, "Memory", run_memory, skippable=True)

        # Step 7: Device
        device_data = self._run_step(7, "Device", run_device, skippable=True)

        # Step 8: Sovereignty
        sovereignty_data = self._run_step(8, "Sovereignty", run_sovereignty, skippable=True)

        # Step 9: Channels
        channels_data = self._run_step(9, "Channels", run_channels_step, skippable=True)

        # Assemble config
        config = self._build_config(
            identity_data=identity_data,
            api_data=api_data,
            forge_data=forge_data,
            memory_data=memory_data,
            device_data=device_data,
            sovereignty_data=sovereignty_data,
            channels_data=channels_data,
        )

        # Generate identity files
        if identity_files_data.get("generate_identity_files"):
            self._generate_identity_files(config, identity_files_data)

        # Write config
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Writing configuration...", total=None)
            self._config_manager.save(config)

        # Show summary
        self._show_summary(config)

        return config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_header(self, step_num: int, name: str) -> None:
        """Print a step header with progress indicator."""
        self._console.print()
        self._console.rule(f"[bold cyan]Step {step_num}/{_TOTAL_STEPS} \u2014 {name}[/bold cyan]")

    def _run_step(
        self,
        step_num: int,
        name: str,
        step_fn: object,
        *,
        skippable: bool,
    ) -> dict[str, object]:
        """Execute a step function with retry/skip on failure.

        Args:
            step_num: Current step number for display.
            name: Human-readable step name.
            step_fn: Callable that takes a Console and returns a dict.
            skippable: Whether the step can be skipped on failure.

        Returns:
            The result dict from the step, or empty dict if skipped.
        """
        console = self._console
        self._step_header(step_num, name)

        while True:
            try:
                result: dict[str, object] = step_fn(console)  # type: ignore[operator]
                return result
            except SystemExit:
                raise
            except KeyboardInterrupt:
                raise SystemExit(130) from None
            except Exception as exc:
                console.print(f"  [red]Error:[/red] {exc}")

                if skippable:
                    from rich.prompt import Confirm

                    if Confirm.ask("  Retry this step?", console=console, default=True):
                        continue
                    console.print(f"  [dim]Skipping {name}.[/dim]")
                    return {}
                else:
                    from rich.prompt import Confirm

                    if Confirm.ask("  Retry this step?", console=console, default=True):
                        continue
                    # Non-skippable steps must succeed — re-raise
                    raise

    def _build_config(
        self,
        *,
        identity_data: dict[str, object],
        api_data: dict[str, object],
        forge_data: dict[str, object],
        memory_data: dict[str, object],
        device_data: dict[str, object],
        sovereignty_data: dict[str, object],
        channels_data: dict[str, object] | None = None,
    ) -> AnimusConfig:
        """Assemble an AnimusConfig from collected step data."""
        config = self._config_manager.load()

        # Identity
        if identity_data:
            config.identity = IdentitySection(
                name=str(identity_data.get("name", "")),
                timezone=str(identity_data.get("timezone", "")),
                locale=str(identity_data.get("locale", "")),
            )

        # API Keys
        if api_data:
            config.api = ApiSection(
                anthropic_key=str(api_data.get("anthropic_key", "")),
                openai_key=str(api_data.get("openai_key", "")),
            )
            # Ollama config from provider step
            if api_data.get("ollama_enabled"):
                config.ollama = OllamaSection(
                    enabled=True,
                    host=str(api_data.get("ollama_host", "localhost")),
                    port=int(api_data.get("ollama_port", 11434)),  # type: ignore[arg-type]
                    model=str(api_data.get("ollama_model", "llama3.2")),
                )
            if api_data.get("default_backend"):
                config.gateway.default_backend = str(api_data["default_backend"])

        # Forge
        if forge_data:
            config.forge = ForgeSection(
                enabled=bool(forge_data.get("enabled", False)),
                host=str(forge_data.get("host", "localhost")),
                port=int(forge_data.get("port", 8000)),  # type: ignore[arg-type]
                api_key=str(forge_data.get("api_key", "")),
            )

        # Memory
        if memory_data:
            config.memory = MemorySection(
                backend=str(memory_data.get("backend", "sqlite")),
                path=str(memory_data.get("path", "~/.local/share/animus/memory.db")),
                max_context_tokens=int(
                    memory_data.get("max_context_tokens", 100_000)  # type: ignore[arg-type]
                ),
            )

        # Device — stored in animus section metadata (no dedicated section yet)
        # sovereignty — telemetry flag
        if sovereignty_data:
            config.animus.first_run = False

        # Channels
        if channels_data:
            raw_channels = channels_data.get("channels", {})
            if isinstance(raw_channels, dict):
                for ch_name, ch_config in raw_channels.items():
                    if isinstance(ch_config, dict) and hasattr(config.channels, ch_name):
                        channel_model = getattr(config.channels, ch_name)
                        for field, value in ch_config.items():
                            if hasattr(channel_model, field):
                                setattr(channel_model, field, value)

        return config

    def _generate_identity_files(
        self,
        config: AnimusConfig,
        identity_files_data: dict[str, object],
    ) -> None:
        """Generate identity files using templates and collected data."""
        from datetime import UTC, datetime
        from pathlib import Path

        from animus_bootstrap.identity.manager import IdentityFileManager

        identity_dir = Path(config.identity.identity_dir).expanduser()
        mgr = IdentityFileManager(identity_dir)

        context = {
            "name": config.identity.name,
            "timezone": config.identity.timezone,
            "about": identity_files_data.get("about", ""),
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        }
        mgr.generate_from_templates(context)
        self._console.print("  [green]Identity files generated.[/green]")

    def _show_summary(self, config: AnimusConfig) -> None:
        """Display a final summary panel of the configuration."""
        console = self._console

        table = Table(show_header=False, border_style="green", padding=(0, 2))
        table.add_column("Setting", style="bold")
        table.add_column("Value", style="cyan")

        table.add_row("Name", config.identity.name or "[dim]not set[/dim]")
        table.add_row("Timezone", config.identity.timezone or "[dim]not set[/dim]")
        table.add_row("Locale", config.identity.locale or "[dim]not set[/dim]")
        table.add_row("Anthropic key", _mask_key(config.api.anthropic_key))
        table.add_row("OpenAI key", _mask_key(config.api.openai_key))
        table.add_row(
            "Forge",
            f"{config.forge.host}:{config.forge.port}"
            if config.forge.enabled
            else "[dim]disabled[/dim]",
        )
        table.add_row("Memory backend", config.memory.backend)

        # Channels summary
        enabled_channels = [
            name
            for name in (
                "webchat",
                "telegram",
                "discord",
                "slack",
                "matrix",
                "signal",
                "whatsapp",
                "email",
            )
            if getattr(config.channels, name, None)
            and getattr(getattr(config.channels, name), "enabled", False)
        ]
        channels_display = ", ".join(enabled_channels) if enabled_channels else "[dim]none[/dim]"
        table.add_row("Channels", channels_display)

        table.add_row("Data path", config.animus.data_dir)
        table.add_row(
            "Config file",
            str(self._config_manager.get_config_path()),
        )

        console.print()
        console.print(
            Panel(
                table,
                title="[bold green]Setup Complete[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )
        console.print()
        console.print("[bold green]Animus is ready.[/bold green] Welcome aboard.")
        console.print()


def _mask_key(key: str) -> str:
    """Mask an API key for display, showing only the last 4 characters."""
    if not key:
        return "[dim]not set[/dim]"
    if len(key) <= 8:
        return "****" + key[-4:]
    return key[:4] + "..." + key[-4:]
