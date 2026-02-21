"""Discord channel adapter via discord.py."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from animus_bootstrap.gateway.models import (
    ChannelHealth,
    GatewayMessage,
    GatewayResponse,
    create_message,
)

try:
    import discord

    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False

logger = logging.getLogger(__name__)

MessageCallback = Callable[[GatewayMessage], Coroutine[Any, Any, None]]


class DiscordAdapter:
    """Channel adapter for Discord via discord.py."""

    name = "discord"

    def __init__(
        self,
        bot_token: str,
        allowed_guilds: list[str] | None = None,
    ) -> None:
        if not HAS_DISCORD:
            raise ImportError("Install discord.py: pip install animus-bootstrap[discord]")
        self.is_connected = False
        self._token = bot_token
        self._allowed_guilds: set[str] = set(allowed_guilds or [])
        self._client: discord.Client | None = None
        self._callback: MessageCallback | None = None
        self._run_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        """Start the Discord client and begin listening for messages."""
        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)

        adapter = self  # capture for closure

        @self._client.event
        async def on_ready() -> None:
            logger.info("Discord adapter connected as %s", self._client.user)  # type: ignore[union-attr]
            adapter.is_connected = True

        @self._client.event
        async def on_message(message: discord.Message) -> None:
            # Ignore own messages
            if message.author == adapter._client.user:  # type: ignore[union-attr]
                return

            # Guild filtering
            if adapter._allowed_guilds and message.guild:
                if str(message.guild.id) not in adapter._allowed_guilds:
                    return

            gw_msg = create_message(
                channel="discord",
                sender_id=str(message.author.id),
                sender_name=str(message.author),
                text=message.content or "",
                channel_message_id=str(message.id),
                metadata={
                    "channel_id": str(message.channel.id),
                    "guild_id": str(message.guild.id) if message.guild else "",
                },
            )

            if adapter._callback:
                try:
                    await adapter._callback(gw_msg)
                except Exception:
                    logger.exception("Error in Discord message callback")

        # Start the client in a background task
        loop = asyncio.get_running_loop()
        self._run_task = loop.create_task(self._client.start(self._token))
        self.is_connected = True

    async def disconnect(self) -> None:
        """Close the Discord client."""
        if self._client:
            try:
                await self._client.close()
            except Exception:
                logger.exception("Error closing Discord client")

        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass

        self.is_connected = False
        logger.info("Discord adapter disconnected")

    async def send_message(self, response: GatewayResponse) -> str:
        """Send a message to a Discord channel.

        The ``channel_id`` must be present in ``response.metadata``.
        Returns the sent message ID as a string.
        """
        if not self._client:
            raise RuntimeError("Discord adapter is not connected")

        channel_id = response.metadata.get("channel_id")
        if not channel_id:
            raise ValueError("response.metadata must contain 'channel_id'")

        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(channel_id))

        if not hasattr(channel, "send"):
            raise ValueError(f"Channel {channel_id} is not a text channel")

        sent = await channel.send(response.text)  # type: ignore[union-attr]
        return str(sent.id)

    async def on_message(self, callback: MessageCallback) -> None:
        """Register a callback to receive incoming messages."""
        self._callback = callback

    async def health_check(self) -> ChannelHealth:
        """Check if the Discord client is connected and ready."""
        connected = self._client is not None and self._client.is_ready()
        return ChannelHealth(
            channel="discord",
            connected=connected,
            error=None if connected else "Not connected or not ready",
        )
