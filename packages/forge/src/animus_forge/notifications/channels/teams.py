"""Microsoft Teams notification channel."""

from __future__ import annotations

import json
import logging
from urllib.request import Request

import animus_forge.notifications.notifier as _notifier_mod

from ..base import NotificationChannel
from ..models import EventType, NotificationEvent

logger = logging.getLogger(__name__)


class TeamsChannel(NotificationChannel):
    """Send notifications to Microsoft Teams via webhook."""

    def __init__(self, webhook_url: str):
        """Initialize Teams channel.

        Args:
            webhook_url: Microsoft Teams incoming webhook URL
        """
        self.webhook_url = webhook_url

    def name(self) -> str:
        return "teams"

    def send(self, event: NotificationEvent) -> bool:
        """Send notification to Teams."""
        color = self._severity_to_color(event.severity)

        # Build Teams Adaptive Card
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": f"{event.workflow_name}: {event.message}",
            "sections": [
                {
                    "activityTitle": f"{self._event_emoji(event.event_type)} {event.workflow_name}",
                    "activitySubtitle": event.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "text": event.message,
                    "facts": self._build_facts(event),
                }
            ],
        }

        return self._post(payload)

    def _severity_to_color(self, severity: str) -> str:
        colors = {
            "info": "0078D7",
            "success": "2DC72D",
            "warning": "FFA500",
            "error": "FF0000",
        }
        return colors.get(severity, "808080")

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

    def _build_facts(self, event: NotificationEvent) -> list:
        facts = [
            {"name": "Event", "value": event.event_type.value},
            {"name": "Severity", "value": event.severity.upper()},
        ]
        for key, value in event.details.items():
            if isinstance(value, (str, int, float, bool)):
                facts.append({"name": key.replace("_", " ").title(), "value": str(value)})
        return facts

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
            logger.error(f"Teams notification failed: {e}")
            return False
