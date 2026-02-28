"""Channel adapter protocol — defines the interface all channels must implement."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from animus_bootstrap.gateway.models import ChannelHealth, GatewayMessage, GatewayResponse

__all__ = [
    "ChannelAdapter",
    "ChannelHealth",
    "GatewayMessage",
    "GatewayResponse",
    "MessageCallback",
]

MessageCallback = Callable[["GatewayMessage"], Awaitable[None]]


@runtime_checkable
class ChannelAdapter(Protocol):
    """Protocol that all gateway channel adapters must satisfy.

    Each adapter bridges a specific messaging platform (WebChat, Telegram,
    Discord, etc.) to the normalised :class:`GatewayMessage` /
    :class:`GatewayResponse` format.
    """

    name: str
    is_connected: bool

    async def connect(self) -> None:
        """Establish the channel connection."""

    async def disconnect(self) -> None:
        """Tear down the channel connection and clean up resources."""

    async def send_message(self, response: GatewayResponse) -> str:
        """Send a response message through the channel.

        Returns:
            A platform-specific message ID for tracking.
        """

    async def on_message(self, callback: MessageCallback) -> None:
        """Register a callback to be invoked when an inbound message arrives."""

    async def health_check(self) -> ChannelHealth:
        """Return the current health status of this channel."""
