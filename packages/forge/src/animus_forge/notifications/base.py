"""Base notification channel interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import NotificationEvent


class NotificationChannel(ABC):
    """Base class for notification channels."""

    @abstractmethod
    def send(self, event: NotificationEvent) -> bool:
        """Send a notification.

        Args:
            event: The notification event

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Get channel name."""
        pass
