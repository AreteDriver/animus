"""
Base Integration Framework

Abstract base class and common types for all integrations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from animus.tools import Tool


class IntegrationStatus(Enum):
    """Status of an integration connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    EXPIRED = "expired"


class AuthType(Enum):
    """Authentication type required by integration."""

    NONE = "none"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"


@dataclass
class IntegrationInfo:
    """Information about an integration's current state."""

    name: str
    display_name: str
    status: IntegrationStatus
    auth_type: AuthType
    connected_at: datetime | None = None
    expires_at: datetime | None = None
    error_message: str | None = None
    capabilities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "status": self.status.value,
            "auth_type": self.auth_type.value,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "error_message": self.error_message,
            "capabilities": self.capabilities,
        }


class BaseIntegration(ABC):
    """
    Abstract base class for external integrations.

    All integrations must implement:
    - connect(): Establish connection with credentials
    - disconnect(): Clean up and disconnect
    - verify(): Check if connection is still valid
    - get_tools(): Return list of tools this integration provides
    """

    name: str = "base"
    display_name: str = "Base Integration"
    auth_type: AuthType = AuthType.NONE

    def __init__(self):
        self._status = IntegrationStatus.DISCONNECTED
        self._connected_at: datetime | None = None
        self._expires_at: datetime | None = None
        self._error_message: str | None = None
        self._credentials: dict[str, Any] = {}

    @property
    def status(self) -> IntegrationStatus:
        """Current connection status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Check if integration is connected."""
        return self._status == IntegrationStatus.CONNECTED

    @abstractmethod
    async def connect(self, credentials: dict[str, Any]) -> bool:
        """
        Establish connection to the service.

        Args:
            credentials: Authentication credentials (API key, OAuth tokens, etc.)

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the service and clean up.

        Returns:
            True if disconnection successful
        """
        pass

    @abstractmethod
    async def verify(self) -> bool:
        """
        Verify the connection is still valid.

        Returns:
            True if connection is valid
        """
        pass

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """
        Get the tools provided by this integration.

        Returns:
            List of Tool instances
        """
        pass

    def get_info(self) -> IntegrationInfo:
        """Get current integration information."""
        return IntegrationInfo(
            name=self.name,
            display_name=self.display_name,
            status=self._status,
            auth_type=self.auth_type,
            connected_at=self._connected_at,
            expires_at=self._expires_at,
            error_message=self._error_message,
            capabilities=[t.name for t in self.get_tools()] if self.is_connected else [],
        )

    def _set_connected(self, expires_at: datetime | None = None) -> None:
        """Mark integration as connected."""
        self._status = IntegrationStatus.CONNECTED
        self._connected_at = datetime.now()
        self._expires_at = expires_at
        self._error_message = None

    def _set_disconnected(self) -> None:
        """Mark integration as disconnected."""
        self._status = IntegrationStatus.DISCONNECTED
        self._connected_at = None
        self._expires_at = None
        self._credentials = {}

    def _set_error(self, message: str) -> None:
        """Mark integration as having an error."""
        self._status = IntegrationStatus.ERROR
        self._error_message = message

    def _set_expired(self) -> None:
        """Mark integration as expired."""
        self._status = IntegrationStatus.EXPIRED
        self._error_message = "Authentication expired"
