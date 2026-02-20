"""PagerDuty notification channel."""

from __future__ import annotations

import json
import logging
from urllib.request import Request

import animus_forge.notifications.notifier as _notifier_mod

from ..base import NotificationChannel
from ..models import EventType, NotificationEvent

logger = logging.getLogger(__name__)


class PagerDutyChannel(NotificationChannel):
    """Send notifications to PagerDuty via Events API v2."""

    def __init__(
        self,
        routing_key: str,
        source: str = "gorgon",
        component: str = "workflow-engine",
    ):
        """Initialize PagerDuty channel.

        Args:
            routing_key: PagerDuty Events API v2 integration key
            source: Source identifier for events
            component: Component name
        """
        self.routing_key = routing_key
        self.source = source
        self.component = component
        self.api_url = "https://events.pagerduty.com/v2/enqueue"

    def name(self) -> str:
        return "pagerduty"

    def send(self, event: NotificationEvent) -> bool:
        """Send notification to PagerDuty."""
        # Only send critical events to PagerDuty
        if event.severity not in ("error", "warning") and event.event_type not in (
            EventType.WORKFLOW_FAILED,
            EventType.BUDGET_EXCEEDED,
        ):
            logger.debug(f"Skipping PagerDuty for non-critical event: {event.event_type}")
            return True

        severity = self._map_severity(event.severity)
        action = "trigger" if event.severity == "error" else "trigger"

        payload = {
            "routing_key": self.routing_key,
            "event_action": action,
            "dedup_key": f"{event.workflow_name}-{event.event_type.value}",
            "payload": {
                "summary": f"{event.workflow_name}: {event.message}",
                "source": self.source,
                "severity": severity,
                "timestamp": event.timestamp.isoformat(),
                "component": self.component,
                "custom_details": {
                    "workflow_name": event.workflow_name,
                    "event_type": event.event_type.value,
                    **event.details,
                },
            },
        }

        return self._post(payload)

    def _map_severity(self, severity: str) -> str:
        """Map internal severity to PagerDuty severity."""
        mapping = {
            "error": "critical",
            "warning": "warning",
            "info": "info",
            "success": "info",
        }
        return mapping.get(severity, "info")

    def _post(self, payload: dict) -> bool:
        try:
            data = json.dumps(payload).encode("utf-8")
            req = Request(
                self.api_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with _notifier_mod.urlopen(req, timeout=10) as response:
                return response.status in (200, 202)
        except _notifier_mod.URLError as e:
            logger.error(f"PagerDuty notification failed: {e}")
            return False

    def resolve(self, workflow_name: str, event_type: EventType) -> bool:
        """Resolve a PagerDuty incident.

        Args:
            workflow_name: Name of the workflow
            event_type: Original event type that triggered the incident

        Returns:
            True if resolved successfully
        """
        payload = {
            "routing_key": self.routing_key,
            "event_action": "resolve",
            "dedup_key": f"{workflow_name}-{event_type.value}",
        }
        return self._post(payload)
