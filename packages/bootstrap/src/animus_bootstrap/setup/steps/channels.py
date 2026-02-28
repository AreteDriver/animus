"""Channels step -- configure messaging gateway channel adapters."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# Channel definitions: name -> list of required config fields.
# Channels with an empty list only need ``enabled=True``.
AVAILABLE_CHANNELS: dict[str, list[str]] = {
    "telegram": ["bot_token"],
    "discord": ["bot_token"],
    "slack": ["app_token", "bot_token"],
    "matrix": ["homeserver", "access_token"],
    "whatsapp": ["phone_number"],
    "signal": ["phone_number"],
    "email": ["smtp_host", "imap_host", "username", "password"],
    "webchat": [],
}

_CHANNEL_DESCRIPTIONS: dict[str, str] = {
    "telegram": "Telegram Bot API",
    "discord": "Discord Bot",
    "slack": "Slack App (Socket Mode)",
    "matrix": "Matrix / Element",
    "whatsapp": "WhatsApp Business API",
    "signal": "Signal Messenger",
    "email": "Email (SMTP/IMAP)",
    "webchat": "Built-in Web Chat (no config needed)",
}

_SENSITIVE_FIELDS = {"bot_token", "app_token", "access_token", "password"}


def run_channels_step(console: Console) -> dict[str, dict[str, object]]:
    """Prompt the user to select and configure messaging channels.

    Displays a panel explaining the message gateway, then asks which
    channels to enable.  For each selected channel, prompts for the
    required credentials.

    Args:
        console: Rich Console instance for terminal output.

    Returns:
        Dictionary keyed by ``"channels"`` containing per-channel config
        dicts.  Each channel dict has at minimum ``enabled: True`` and
        any required credential fields.
    """
    console.print()
    console.print(
        Panel(
            "[bold]Message Gateway Channels[/bold]\n\n"
            "Animus can communicate through multiple messaging platforms.\n"
            "Select the channels you want to enable. You can always add\n"
            "more later with [cyan]animus-bootstrap channels enable <name>[/cyan].",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # Show available channels
    console.print()
    channel_names = list(AVAILABLE_CHANNELS.keys())
    for i, name in enumerate(channel_names, 1):
        desc = _CHANNEL_DESCRIPTIONS.get(name, name)
        console.print(f"  [bold]{i}.[/bold] [cyan]{name}[/cyan] -- {desc}")

    # Multi-select: comma-separated numbers or names
    console.print()
    selection = Prompt.ask(
        "[bold]Enable channels[/bold] (comma-separated numbers or names, blank to skip)",
        console=console,
        default="",
    )

    selected = _parse_selection(selection, channel_names)

    if not selected:
        console.print("  [dim]No channels selected. You can enable them later.[/dim]")
        return {"channels": {}}

    # Collect config for each selected channel
    channels_config: dict[str, dict[str, object]] = {}

    for channel in selected:
        console.print()
        console.print(f"  [bold cyan]{channel}[/bold cyan] configuration:")
        fields = AVAILABLE_CHANNELS[channel]
        channel_data: dict[str, object] = {"enabled": True}

        for field in fields:
            is_secret = field in _SENSITIVE_FIELDS
            value = Prompt.ask(
                f"    {field}",
                console=console,
                password=is_secret,
            )
            channel_data[field] = value

        channels_config[channel] = channel_data

    # Summary
    _show_summary(console, channels_config)

    return {"channels": channels_config}


def _parse_selection(raw: str, channel_names: list[str]) -> list[str]:
    """Parse a comma-separated selection into a list of valid channel names.

    Accepts numbers (1-based index) or channel names.  Silently skips
    invalid entries.

    Args:
        raw: Raw input string from the user.
        channel_names: Ordered list of available channel names.

    Returns:
        De-duplicated list of selected channel names in input order.
    """
    if not raw.strip():
        return []

    selected: list[str] = []
    seen: set[str] = set()

    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue

        # Try numeric index first
        try:
            idx = int(token) - 1
            if 0 <= idx < len(channel_names):
                name = channel_names[idx]
                if name not in seen:
                    selected.append(name)
                    seen.add(name)
                continue
        except ValueError:
            pass  # Not numeric — fall through to name match below

        # Try name match
        if token in channel_names and token not in seen:
            selected.append(token)
            seen.add(token)

    return selected


def _show_summary(console: Console, channels_config: dict[str, dict[str, object]]) -> None:
    """Display a summary table of configured channels."""
    console.print()
    table = Table(title="Configured Channels", border_style="green")
    table.add_column("Channel", style="bold")
    table.add_column("Status", style="green")
    table.add_column("Fields", style="dim")

    for name, data in channels_config.items():
        field_count = len([k for k in data if k != "enabled"])
        fields_str = f"{field_count} field(s)" if field_count else "no extra config"
        table.add_row(name, "Enabled", fields_str)

    console.print(table)
