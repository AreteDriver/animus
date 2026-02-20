"""Central notification manager."""

from __future__ import annotations

import logging

from .base import NotificationChannel
from .models import EventType, NotificationEvent

logger = logging.getLogger(__name__)


class Notifier:
    """Central notification manager.

    Usage:
        notifier = Notifier()
        notifier.add_channel(SlackChannel(webhook_url="..."))
        notifier.add_channel(DiscordChannel(webhook_url="..."))

        # Send notification
        notifier.notify(NotificationEvent(
            event_type=EventType.WORKFLOW_COMPLETED,
            workflow_name="feature-build",
            message="Workflow completed successfully",
            severity="success",
            details={"tokens_used": 5000, "duration_ms": 12000}
        ))

        # Or use convenience methods
        notifier.workflow_completed("feature-build", tokens=5000)
        notifier.workflow_failed("feature-build", error="Step 3 failed")
    """

    def __init__(self):
        self._channels: list[NotificationChannel] = []
        self._event_filters: dict[EventType, bool] = {e: True for e in EventType}

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self._channels.append(channel)
        logger.info(f"Added notification channel: {channel.name()}")

    def remove_channel(self, channel_name: str) -> bool:
        """Remove a channel by name."""
        for i, ch in enumerate(self._channels):
            if ch.name() == channel_name:
                self._channels.pop(i)
                return True
        return False

    def set_filter(self, event_type: EventType, enabled: bool) -> None:
        """Enable/disable notifications for an event type."""
        self._event_filters[event_type] = enabled

    def notify(self, event: NotificationEvent) -> dict[str, bool]:
        """Send notification to all channels.

        Args:
            event: The notification event

        Returns:
            Dict of channel_name -> success status
        """
        # Check filter
        if not self._event_filters.get(event.event_type, True):
            return {}

        results = {}
        for channel in self._channels:
            try:
                results[channel.name()] = channel.send(event)
            except Exception as e:
                logger.error(f"Channel {channel.name()} failed: {e}")
                results[channel.name()] = False

        return results

    # Convenience methods

    def workflow_started(self, workflow_name: str, **details) -> dict[str, bool]:
        """Notify that a workflow has started."""
        return self.notify(
            NotificationEvent(
                event_type=EventType.WORKFLOW_STARTED,
                workflow_name=workflow_name,
                message=f"Workflow '{workflow_name}' started",
                severity="info",
                details=details,
            )
        )

    def workflow_completed(
        self,
        workflow_name: str,
        tokens: int = 0,
        duration_ms: int = 0,
        **details,
    ) -> dict[str, bool]:
        """Notify that a workflow completed successfully."""
        details.update({"tokens_used": tokens, "duration_ms": duration_ms})
        return self.notify(
            NotificationEvent(
                event_type=EventType.WORKFLOW_COMPLETED,
                workflow_name=workflow_name,
                message=f"Workflow '{workflow_name}' completed successfully",
                severity="success",
                details=details,
            )
        )

    def workflow_failed(
        self,
        workflow_name: str,
        error: str,
        step: str | None = None,
        **details,
    ) -> dict[str, bool]:
        """Notify that a workflow failed."""
        if step:
            details["failed_step"] = step
        details["error"] = error
        return self.notify(
            NotificationEvent(
                event_type=EventType.WORKFLOW_FAILED,
                workflow_name=workflow_name,
                message=f"Workflow '{workflow_name}' failed: {error}",
                severity="error",
                details=details,
            )
        )

    def step_failed(
        self,
        workflow_name: str,
        step_name: str,
        error: str,
        **details,
    ) -> dict[str, bool]:
        """Notify that a step failed."""
        details.update({"step": step_name, "error": error})
        return self.notify(
            NotificationEvent(
                event_type=EventType.STEP_FAILED,
                workflow_name=workflow_name,
                message=f"Step '{step_name}' failed in '{workflow_name}': {error}",
                severity="warning",
                details=details,
            )
        )

    def budget_warning(
        self,
        workflow_name: str,
        used: int,
        budget: int,
        percent: float,
        **details,
    ) -> dict[str, bool]:
        """Notify that budget threshold was crossed."""
        details.update({"used": used, "budget": budget, "percent": f"{percent:.1f}%"})
        return self.notify(
            NotificationEvent(
                event_type=EventType.BUDGET_WARNING,
                workflow_name=workflow_name,
                message=f"Budget warning: {percent:.1f}% used ({used}/{budget} tokens)",
                severity="warning",
                details=details,
            )
        )

    def budget_exceeded(
        self,
        workflow_name: str,
        used: int,
        budget: int,
        **details,
    ) -> dict[str, bool]:
        """Notify that budget was exceeded."""
        details.update({"used": used, "budget": budget})
        return self.notify(
            NotificationEvent(
                event_type=EventType.BUDGET_EXCEEDED,
                workflow_name=workflow_name,
                message=f"Budget exceeded: {used}/{budget} tokens",
                severity="error",
                details=details,
            )
        )
