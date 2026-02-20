"""Slack notification channel."""

from __future__ import annotations

import json
import logging
from urllib.request import Request

import animus_forge.notifications.notifier as _notifier_mod

from ..base import NotificationChannel
from ..models import EventType, NotificationEvent

logger = logging.getLogger(__name__)


class SlackChannel(NotificationChannel):
    """Send notifications to Slack via webhook."""

    def __init__(
        self,
        webhook_url: str,
        channel: str | None = None,
        username: str = "Gorgon",
        icon_emoji: str = ":robot_face:",
    ):
        """Initialize Slack channel.

        Args:
            webhook_url: Slack incoming webhook URL
            channel: Optional channel override
            username: Bot username
            icon_emoji: Bot icon emoji
        """
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji

    def name(self) -> str:
        return "slack"

    def send(self, event: NotificationEvent) -> bool:
        """Send notification to Slack."""
        color = self._severity_to_color(event.severity)

        # Build Slack message
        payload = {
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "attachments": [
                {
                    "color": color,
                    "title": f"{self._event_emoji(event.event_type)} {event.workflow_name}",
                    "text": event.message,
                    "fields": self._build_fields(event),
                    "footer": "Gorgon Workflow Engine",
                    "ts": int(event.timestamp.timestamp()),
                }
            ],
        }

        if self.channel:
            payload["channel"] = self.channel

        return self._post(payload)

    def _severity_to_color(self, severity: str) -> str:
        colors = {
            "info": "#3498db",
            "success": "#2ecc71",
            "warning": "#f39c12",
            "error": "#e74c3c",
        }
        return colors.get(severity, "#95a5a6")

    def _event_emoji(self, event_type: EventType) -> str:
        emojis = {
            EventType.WORKFLOW_STARTED: ":arrow_forward:",
            EventType.WORKFLOW_COMPLETED: ":white_check_mark:",
            EventType.WORKFLOW_FAILED: ":x:",
            EventType.STEP_COMPLETED: ":heavy_check_mark:",
            EventType.STEP_FAILED: ":warning:",
            EventType.BUDGET_WARNING: ":moneybag:",
            EventType.BUDGET_EXCEEDED: ":no_entry:",
            EventType.SCHEDULE_TRIGGERED: ":alarm_clock:",
        }
        return emojis.get(event_type, ":bell:")

    def _build_fields(self, event: NotificationEvent) -> list:
        fields = [
            {"title": "Event", "value": event.event_type.value, "short": True},
            {"title": "Severity", "value": event.severity.upper(), "short": True},
        ]

        # Add details as fields
        for key, value in event.details.items():
            if isinstance(value, (str, int, float, bool)):
                fields.append(
                    {
                        "title": key.replace("_", " ").title(),
                        "value": str(value),
                        "short": True,
                    }
                )

        return fields

    def _post(self, payload: dict) -> bool:
        try:
            data = json.dumps(payload).encode("utf-8")
            req = Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with _notifier_mod.urlopen(req, timeout=10) as response:
                return response.status == 200
        except _notifier_mod.URLError as e:
            logger.error(f"Slack notification failed: {e}")
            return False
