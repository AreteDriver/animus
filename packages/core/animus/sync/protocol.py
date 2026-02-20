"""
Sync Protocol

Defines the message format and types for cross-device synchronization.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from animus.logging import get_logger

logger = get_logger("sync.protocol")


class MessageType(Enum):
    """Types of sync messages."""

    # Connection
    AUTH = "auth"
    AUTH_OK = "auth_ok"
    AUTH_FAIL = "auth_fail"

    # Sync operations
    SNAPSHOT_REQUEST = "snapshot_request"
    SNAPSHOT_RESPONSE = "snapshot_response"
    DELTA_PUSH = "delta_push"
    DELTA_ACK = "delta_ack"

    # Handoff
    HANDOFF_REQUEST = "handoff_request"
    HANDOFF_ACCEPT = "handoff_accept"
    HANDOFF_REJECT = "handoff_reject"

    # Status
    PING = "ping"
    PONG = "pong"
    STATUS = "status"
    ERROR = "error"


@dataclass
class SyncMessage:
    """A sync protocol message."""

    type: MessageType
    device_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    payload: dict[str, Any] = field(default_factory=dict)
    message_id: str = ""

    def __post_init__(self):
        if not self.message_id:
            import uuid

            self.message_id = str(uuid.uuid4())[:8]

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "type": self.type.value,
                "device_id": self.device_id,
                "timestamp": self.timestamp.isoformat(),
                "payload": self.payload,
                "message_id": self.message_id,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> "SyncMessage":
        """Deserialize from JSON string."""
        parsed = json.loads(data)
        return cls(
            type=MessageType(parsed["type"]),
            device_id=parsed["device_id"],
            timestamp=datetime.fromisoformat(parsed["timestamp"]),
            payload=parsed.get("payload", {}),
            message_id=parsed.get("message_id", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "message_id": self.message_id,
        }


# Message factory functions for common operations


def create_auth_message(device_id: str, shared_secret: str) -> SyncMessage:
    """Create an authentication message."""
    import hashlib

    # Hash the secret for transmission
    auth_hash = hashlib.sha256(shared_secret.encode()).hexdigest()
    return SyncMessage(
        type=MessageType.AUTH,
        device_id=device_id,
        payload={"auth_hash": auth_hash},
    )


def create_auth_ok_message(device_id: str, device_name: str, version: int) -> SyncMessage:
    """Create authentication success response."""
    return SyncMessage(
        type=MessageType.AUTH_OK,
        device_id=device_id,
        payload={"device_name": device_name, "version": version},
    )


def create_auth_fail_message(device_id: str, reason: str) -> SyncMessage:
    """Create authentication failure response."""
    return SyncMessage(
        type=MessageType.AUTH_FAIL,
        device_id=device_id,
        payload={"reason": reason},
    )


def create_snapshot_request(device_id: str, since_version: int = 0) -> SyncMessage:
    """Create a snapshot request message."""
    return SyncMessage(
        type=MessageType.SNAPSHOT_REQUEST,
        device_id=device_id,
        payload={"since_version": since_version},
    )


def create_snapshot_response(
    device_id: str,
    snapshot_data: dict[str, Any],
    version: int,
) -> SyncMessage:
    """Create a snapshot response message."""
    return SyncMessage(
        type=MessageType.SNAPSHOT_RESPONSE,
        device_id=device_id,
        payload={"snapshot": snapshot_data, "version": version},
    )


def create_delta_push(device_id: str, delta_data: dict[str, Any]) -> SyncMessage:
    """Create a delta push message."""
    return SyncMessage(
        type=MessageType.DELTA_PUSH,
        device_id=device_id,
        payload={"delta": delta_data},
    )


def create_delta_ack(device_id: str, delta_id: str, success: bool) -> SyncMessage:
    """Create a delta acknowledgment message."""
    return SyncMessage(
        type=MessageType.DELTA_ACK,
        device_id=device_id,
        payload={"delta_id": delta_id, "success": success},
    )


def create_handoff_request(
    device_id: str,
    context: dict[str, Any],
) -> SyncMessage:
    """Create a handoff request message."""
    return SyncMessage(
        type=MessageType.HANDOFF_REQUEST,
        device_id=device_id,
        payload={"context": context},
    )


def create_handoff_accept(device_id: str) -> SyncMessage:
    """Create a handoff acceptance message."""
    return SyncMessage(
        type=MessageType.HANDOFF_ACCEPT,
        device_id=device_id,
    )


def create_handoff_reject(device_id: str, reason: str = "") -> SyncMessage:
    """Create a handoff rejection message."""
    return SyncMessage(
        type=MessageType.HANDOFF_REJECT,
        device_id=device_id,
        payload={"reason": reason},
    )


def create_ping(device_id: str) -> SyncMessage:
    """Create a ping message."""
    return SyncMessage(
        type=MessageType.PING,
        device_id=device_id,
    )


def create_pong(device_id: str) -> SyncMessage:
    """Create a pong message."""
    return SyncMessage(
        type=MessageType.PONG,
        device_id=device_id,
    )


def create_status_message(
    device_id: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> SyncMessage:
    """Create a status message."""
    return SyncMessage(
        type=MessageType.STATUS,
        device_id=device_id,
        payload={"status": status, "details": details or {}},
    )


def create_error_message(device_id: str, error: str, code: str = "") -> SyncMessage:
    """Create an error message."""
    return SyncMessage(
        type=MessageType.ERROR,
        device_id=device_id,
        payload={"error": error, "code": code},
    )
