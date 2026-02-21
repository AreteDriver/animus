"""Gateway data models — lightweight message objects."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Attachment:
    """File or media attachment on a message."""

    filename: str
    content_type: str
    data: bytes | None = None
    url: str | None = None


@dataclass
class GatewayMessage:
    """Normalised message flowing through the gateway."""

    id: str
    channel: str
    channel_message_id: str
    sender_id: str
    sender_name: str
    text: str
    timestamp: datetime
    attachments: list[Attachment] = field(default_factory=list)
    reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    role: str = "user"


@dataclass
class GatewayResponse:
    """Response object returned from the router to a channel adapter."""

    text: str
    channel: str
    attachments: list[Attachment] = field(default_factory=list)
    channel_message_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelHealth:
    """Health status for a connected channel."""

    channel: str
    connected: bool
    latency_ms: float | None = None
    last_message_at: datetime | None = None
    error: str | None = None


def create_message(
    channel: str,
    sender_id: str,
    sender_name: str,
    text: str,
    **kwargs: Any,
) -> GatewayMessage:
    """Create a :class:`GatewayMessage` with auto-generated UUID and timestamp."""
    return GatewayMessage(
        id=kwargs.pop("id", str(uuid.uuid4())),
        channel=channel,
        channel_message_id=kwargs.pop("channel_message_id", ""),
        sender_id=sender_id,
        sender_name=sender_name,
        text=text,
        timestamp=kwargs.pop("timestamp", datetime.now(UTC)),
        **kwargs,
    )
