"""Discord notification channel."""

from __future__ import annotations

import json
import logging
from urllib.request import Request

import animus_forge.notifications.notifier as _notifier_mod

from ..base import NotificationChannel
from ..models import EventType, NotificationEvent

logger = logging.getLogger(__name__)


class DiscordChannel(NotificationChannel):
    """Send notifications to Discord via webhook."""

    def __init__(
        self,
        webhook_url: str,
        username: str = "Gorgon",
        avatar_url: str | None = None,
    ):
        """Initialize Discord channel.

        Args:
            webhook_url: Discord webhook URL
            username: Bot username
            avatar_url: Optional avatar URL
        """
        self.webhook_url = webhook_url
        self.username = username
        self.avatar_url = avatar_url

    def name(self) -> str:
        return "discord"

    def send(self, event: NotificationEvent) -> bool:
        """Send notification to Discord."""
        color = self._severity_to_color(event.severity)

        # Build Discord embed
        embed = {
            "title": f"{self._event_emoji(event.event_type)} {event.workflow_name}",
            "description": event.message,
            "color": color,
            "fields": self._build_fields(event),
            "footer": {"text": "Gorgon Workflow Engine"},
            "timestamp": event.timestamp.isoformat(),
        }

        payload = {
            "username": self.username,
            "embeds": [embed],
        }

        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url

        return self._post(payload)

    def _severity_to_color(self, severity: str) -> int:
        # Discord uses decimal colors
        colors = {
            "info": 3447003,  # Blue
            "success": 3066993,  # Green
            "warning": 15844367,  # Orange
            "error": 15158332,  # Red
        }
        return colors.get(severity, 9807270)  # Gray

    def _event_emoji(self, event_type: EventType) -> str:
        emojis = {
            EventType.WORKFLOW_STARTED: "\u25b6\ufe0f",
            EventType.WORKFLOW_COMPLETED: "\u2705",
            EventType.WORKFLOW_FAILED: "\u274c",
            EventType.STEP_COMPLETED: "\u2611\ufe0f",
            EventType.STEP_FAILED: "\u26a0\ufe0f",
            EventType.BUDGET_WARNING: "\U0001f4b0",
            EventType.BUDGET_EXCEEDED: "\U0001f6ab",
            EventType.SCHEDULE_TRIGGERED: "\u23f0",
        }
        return emojis.get(event_type, "\U0001f514")

    def _build_fields(self, event: NotificationEvent) -> list:
        fields = [
            {"name": "Event", "value": event.event_type.value, "inline": True},
            {"name": "Severity", "value": event.severity.upper(), "inline": True},
        ]

        for key, value in event.details.items():
            if isinstance(value, (str, int, float, bool)):
                fields.append(
                    {
                        "name": key.replace("_", " ").title(),
                        "value": str(value),
                        "inline": True,
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
                return response.status in (200, 204)
        except _notifier_mod.URLError as e:
            logger.error(f"Discord notification failed: {e}")
            return False
