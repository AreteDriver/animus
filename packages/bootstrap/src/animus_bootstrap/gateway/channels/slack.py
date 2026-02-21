"""Slack channel adapter via slack-bolt SDK."""

from __future__ import annotations

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
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    from slack_bolt.async_app import AsyncApp

    HAS_SLACK = True
except ImportError:
    HAS_SLACK = False

logger = logging.getLogger(__name__)

MessageCallback = Callable[[GatewayMessage], Coroutine[Any, Any, None]]


class SlackAdapter:
    """Channel adapter for Slack via the Bolt SDK (socket mode)."""

    name = "slack"

    def __init__(self, bot_token: str, app_token: str = "") -> None:
        if not HAS_SLACK:
            raise ImportError("Install slack-bolt: pip install animus-bootstrap[slack]")
        self.is_connected = False
        self._bot_token = bot_token
        self._app_token = app_token
        self._app: AsyncApp | None = None  # type: ignore[type-arg]
        self._handler: AsyncSocketModeHandler | None = None  # type: ignore[type-arg]
        self._callback: MessageCallback | None = None

    async def connect(self) -> None:
        """Initialize the Slack app and start socket-mode if an app token is set."""
        self._app = AsyncApp(token=self._bot_token)

        adapter = self  # capture for closure

        @self._app.message("")
        async def handle_message(message: dict[str, Any], say: Any) -> None:
            if adapter._callback:
                gw_msg = create_message(
                    channel="slack",
                    sender_id=message.get("user", ""),
                    sender_name=message.get("user", ""),
                    text=message.get("text", ""),
                    channel_message_id=message.get("ts", ""),
                    metadata={
                        "channel_id": message.get("channel", ""),
                        "ts": message.get("ts", ""),
                        "thread_ts": message.get("thread_ts", ""),
                    },
                )
                try:
                    await adapter._callback(gw_msg)
                except Exception:
                    logger.exception("Error in Slack message callback")

        if self._app_token:
            self._handler = AsyncSocketModeHandler(self._app, self._app_token)
            await self._handler.start_async()

        self.is_connected = True
        logger.info("Slack adapter connected")

    async def disconnect(self) -> None:
        """Close the socket-mode handler."""
        if self._handler:
            try:
                await self._handler.close_async()
            except Exception:
                logger.exception("Error closing Slack socket handler")
        self.is_connected = False
        logger.info("Slack adapter disconnected")

    async def send_message(self, response: GatewayResponse) -> str:
        """Send a message to a Slack channel.

        The ``channel_id`` must be present in ``response.metadata``.
        Optionally ``thread_ts`` can be set to reply in a thread.
        Returns the message timestamp (``ts``) as the message ID.
        """
        if not self._app:
            raise RuntimeError("Slack adapter is not connected")

        channel_id = response.metadata.get("channel_id")
        if not channel_id:
            raise ValueError("response.metadata must contain 'channel_id'")

        kwargs: dict[str, Any] = {
            "channel": channel_id,
            "text": response.text,
        }
        thread_ts = response.metadata.get("thread_ts")
        if thread_ts:
            kwargs["thread_ts"] = thread_ts

        result = await self._app.client.chat_postMessage(**kwargs)
        return result.get("ts", "")

    async def on_message(self, callback: MessageCallback) -> None:
        """Register a callback to receive incoming messages."""
        self._callback = callback

    async def health_check(self) -> ChannelHealth:
        """Test auth via ``auth.test`` API call."""
        if not self._app:
            return ChannelHealth(
                channel="slack",
                connected=False,
                error="Not connected",
            )
        try:
            result = await self._app.client.auth_test()
            ok = result.get("ok", False)
            return ChannelHealth(
                channel="slack",
                connected=ok,
                error=None if ok else "auth.test returned not ok",
            )
        except Exception as exc:
            return ChannelHealth(
                channel="slack",
                connected=False,
                error=str(exc),
            )
