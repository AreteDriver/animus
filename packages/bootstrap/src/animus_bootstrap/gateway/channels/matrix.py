"""Matrix channel adapter via matrix-nio."""

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
    from nio import AsyncClient, MatrixRoom, RoomMessageText

    HAS_MATRIX = True
except ImportError:
    HAS_MATRIX = False

logger = logging.getLogger(__name__)

MessageCallback = Callable[[GatewayMessage], Coroutine[Any, Any, None]]


class MatrixAdapter:
    """Channel adapter for Matrix via matrix-nio."""

    name = "matrix"

    def __init__(
        self,
        homeserver: str,
        access_token: str,
        room_ids: list[str] | None = None,
    ) -> None:
        if not HAS_MATRIX:
            raise ImportError("Install matrix-nio: pip install animus-bootstrap[matrix]")
        self.is_connected = False
        self._homeserver = homeserver
        self._access_token = access_token
        self._room_ids: set[str] = set(room_ids or [])
        self._client: AsyncClient | None = None  # type: ignore[type-arg]
        self._callback: MessageCallback | None = None
        self._sync_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        """Connect to the Matrix homeserver and start syncing."""
        self._client = AsyncClient(self._homeserver)
        self._client.access_token = self._access_token

        adapter = self  # capture for closure

        async def message_callback(room: MatrixRoom, event: RoomMessageText) -> None:  # type: ignore[name-defined]
            # Filter by room if configured
            if adapter._room_ids and room.room_id not in adapter._room_ids:
                return

            if adapter._callback:
                gw_msg = create_message(
                    channel="matrix",
                    sender_id=event.sender,
                    sender_name=event.sender,
                    text=event.body,
                    channel_message_id=event.event_id,
                    metadata={
                        "room_id": room.room_id,
                        "room_name": room.display_name,
                    },
                )
                try:
                    await adapter._callback(gw_msg)
                except Exception:
                    logger.exception("Error in Matrix message callback")

        self._client.add_event_callback(message_callback, RoomMessageText)

        # Start sync in background
        loop = asyncio.get_running_loop()
        self._sync_task = loop.create_task(self._sync_loop())
        self.is_connected = True
        logger.info("Matrix adapter connected to %s", self._homeserver)

    async def _sync_loop(self) -> None:
        """Run the client sync loop."""
        if not self._client:
            return
        try:
            await self._client.sync_forever(timeout=30000)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Matrix sync error")

    async def disconnect(self) -> None:
        """Stop syncing and close the Matrix client."""
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        if self._client:
            try:
                await self._client.close()
            except Exception:
                logger.exception("Error closing Matrix client")

        self.is_connected = False
        logger.info("Matrix adapter disconnected")

    async def send_message(self, response: GatewayResponse) -> str:
        """Send a message to a Matrix room.

        The ``room_id`` must be present in ``response.metadata``.
        Returns the event ID of the sent message.
        """
        if not self._client:
            raise RuntimeError("Matrix adapter is not connected")

        room_id = response.metadata.get("room_id")
        if not room_id:
            raise ValueError("response.metadata must contain 'room_id'")

        result = await self._client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                "body": response.text,
            },
        )

        # nio returns a RoomSendResponse with event_id
        return getattr(result, "event_id", "")

    async def on_message(self, callback: MessageCallback) -> None:
        """Register a callback to receive incoming messages."""
        self._callback = callback

    async def health_check(self) -> ChannelHealth:
        """Check connection by calling ``whoami``."""
        if not self._client:
            return ChannelHealth(
                channel="matrix",
                connected=False,
                error="Not connected",
            )
        try:
            resp = await self._client.whoami()
            # nio returns a WhoamiResponse; check for user_id attribute
            connected = hasattr(resp, "user_id") and resp.user_id is not None
            return ChannelHealth(
                channel="matrix",
                connected=connected,
                error=None if connected else "whoami failed",
            )
        except Exception as exc:
            return ChannelHealth(
                channel="matrix",
                connected=False,
                error=str(exc),
            )
