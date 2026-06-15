"""Kernel channel integrations (Discord, Slack, etc.)."""

from __future__ import annotations

try:
    from animus_kernel.channels.discord_bot import DiscordBot
except Exception:  # pragma: no cover — optional dep (discord.py)
    DiscordBot = None  # type: ignore[misc, assignment]

__all__ = ["DiscordBot"]
