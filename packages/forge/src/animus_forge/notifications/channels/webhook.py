"""Generic webhook notification channel."""

from __future__ import annotations

import json
import logging
from urllib.request import Request

import animus_forge.notifications.notifier as _notifier_mod

from ..base import NotificationChannel
from ..models import NotificationEvent

logger = logging.getLogger(__name__)


class WebhookChannel(NotificationChannel):
    """Send notifications to a generic webhook endpoint."""

    def __init__(
        self,
        url: str,
        headers: dict | None = None,
        method: str = "POST",
    ):
        """Initialize generic webhook channel.

        Args:
            url: Webhook URL
            headers: Optional custom headers
            method: HTTP method (default POST)
        """
        self.url = url
        self.headers = headers or {}
        self.method = method

    def name(self) -> str:
        return "webhook"

    def send(self, event: NotificationEvent) -> bool:
        """Send notification to webhook."""
        try:
            data = json.dumps(event.to_dict()).encode("utf-8")
            headers = {"Content-Type": "application/json", **self.headers}
            req = Request(self.url, data=data, headers=headers, method=self.method)
            with _notifier_mod.urlopen(req, timeout=10) as response:
                return 200 <= response.status < 300
        except _notifier_mod.URLError as e:
            logger.error(f"Webhook notification failed: {e}")
            return False
