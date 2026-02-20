"""WebSocket message models for real-time execution updates."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """WebSocket message types."""

    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"

    # Server -> Client
    CONNECTED = "connected"
    EXECUTION_STATUS = "execution_status"
    EXECUTION_LOG = "execution_log"
    EXECUTION_METRICS = "execution_metrics"
    PONG = "pong"
    ERROR = "error"


# =============================================================================
# Client -> Server Messages
# =============================================================================


class SubscribeMessage(BaseModel):
    """Request to subscribe to execution updates."""

    type: Literal["subscribe"] = "subscribe"
    execution_ids: list[str]


class UnsubscribeMessage(BaseModel):
    """Request to unsubscribe from execution updates."""

    type: Literal["unsubscribe"] = "unsubscribe"
    execution_ids: list[str]


class PingMessage(BaseModel):
    """Client ping for connection keepalive."""

    type: Literal["ping"] = "ping"
    timestamp: int


# =============================================================================
# Server -> Client Messages
# =============================================================================


class ConnectedMessage(BaseModel):
    """Sent on successful WebSocket connection."""

    type: Literal["connected"] = "connected"
    connection_id: str
    server_time: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExecutionStatusMessage(BaseModel):
    """Execution status update."""

    type: Literal["execution_status"] = "execution_status"
    execution_id: str
    status: str
    progress: int
    current_step: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class ExecutionLogMessage(BaseModel):
    """Execution log entry."""

    type: Literal["execution_log"] = "execution_log"
    execution_id: str
    log: dict[str, Any]


class ExecutionMetricsMessage(BaseModel):
    """Execution metrics update."""

    type: Literal["execution_metrics"] = "execution_metrics"
    execution_id: str
    metrics: dict[str, Any]


class PongMessage(BaseModel):
    """Server pong response."""

    type: Literal["pong"] = "pong"
    timestamp: int


class ErrorMessage(BaseModel):
    """Error message."""

    type: Literal["error"] = "error"
    code: str
    message: str
    details: dict[str, Any] | None = None


# Type alias for all outbound messages
OutboundMessage = (
    ConnectedMessage
    | ExecutionStatusMessage
    | ExecutionLogMessage
    | ExecutionMetricsMessage
    | PongMessage
    | ErrorMessage
)
