"""Telegram Bot API channel adapter."""

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
    from telegram import Update
    from telegram.ext import Application, MessageHandler, filters

    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False

logger = logging.getLogger(__name__)

MessageCallback = Callable[[GatewayMessage], Coroutine[Any, Any, None]]


class TelegramAdapter:
    """Channel adapter for Telegram Bot API via python-telegram-bot."""

    name = "telegram"

    def __init__(self, bot_token: str) -> None:
        if not HAS_TELEGRAM:
            raise ImportError("Install python-telegram-bot: pip install animus-bootstrap[telegram]")
        self.is_connected = False
        self._token = bot_token
        self._app: Application | None = None  # type: ignore[type-arg]
        self._callback: MessageCallback | None = None
        self._poll_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        """Start the Telegram bot and begin polling for messages."""
        self._app = Application.builder().token(self._token).build()

        # Register a catch-all message handler
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_update))

        await self._app.initialize()
        await self._app.start()

        # Start polling in background
        loop = asyncio.get_running_loop()
        self._poll_task = loop.create_task(self._poll())
        self.is_connected = True
        logger.info("Telegram adapter connected")

    async def _poll(self) -> None:
        """Run the updater polling loop."""
        if self._app is None:
            return
        try:
            await self._app.updater.start_polling()  # type: ignore[union-attr]
        except asyncio.CancelledError:
            logger.debug("Telegram polling cancelled")

    async def _handle_update(self, update: Update, _context: Any) -> None:  # type: ignore[name-defined]
        """Convert a Telegram Update into a GatewayMessage and dispatch."""
        if not update.message or not update.message.text:
            return

        msg = update.message
        gw_msg = create_message(
            channel="telegram",
            sender_id=str(msg.from_user.id) if msg.from_user else "",
            sender_name=(msg.from_user.full_name if msg.from_user else "Unknown"),
            text=msg.text,
            channel_message_id=str(msg.message_id),
            metadata={
                "chat_id": str(msg.chat_id),
                "chat_type": msg.chat.type if msg.chat else "private",
            },
        )

        if self._callback:
            try:
                await self._callback(gw_msg)
            except Exception:
                logger.exception("Error in Telegram message callback")

    async def disconnect(self) -> None:
        """Stop the bot and clean up resources."""
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                logger.debug("Telegram poll task cancelled")

        if self._app:
            try:
                if self._app.updater and self._app.updater.running:
                    await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception:
                logger.exception("Error shutting down Telegram app")

        self.is_connected = False
        logger.info("Telegram adapter disconnected")

    async def send_message(self, response: GatewayResponse) -> str:
        """Send a message to a Telegram chat.

        The ``chat_id`` must be present in ``response.metadata``.
        Returns the sent message ID as a string.
        """
        if not self._app or not self._app.bot:
            raise RuntimeError("Telegram adapter is not connected")

        chat_id = response.metadata.get("chat_id")
        if not chat_id:
            raise ValueError("response.metadata must contain 'chat_id'")

        sent = await self._app.bot.send_message(
            chat_id=chat_id,
            text=response.text,
        )
        return str(sent.message_id)

    async def on_message(self, callback: MessageCallback) -> None:
        """Register a callback to receive incoming messages."""
        self._callback = callback

    async def health_check(self) -> ChannelHealth:
        """Verify the bot token by calling ``get_me``."""
        if not self._app or not self._app.bot:
            return ChannelHealth(
                channel="telegram",
                connected=False,
                error="Not connected",
            )
        try:
            await self._app.bot.get_me()
            return ChannelHealth(
                channel="telegram",
                connected=True,
                error=None,
            )
        except Exception as exc:
            return ChannelHealth(
                channel="telegram",
                connected=False,
                error=str(exc),
            )
