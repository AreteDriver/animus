"""Outbound Notification System for Workflow Events.

This module is a backward-compatibility shim. All classes have been refactored
into focused submodules:

- models.py: EventType, NotificationEvent
- base.py: NotificationChannel (ABC)
- channels/: SlackChannel, DiscordChannel, WebhookChannel, EmailChannel, TeamsChannel, PagerDutyChannel
- manager.py: Notifier

This shim also re-exports urlopen/Request/URLError so that existing test patches
targeting 'animus_forge.notifications.notifier.urlopen' continue to work.
"""

# These must come BEFORE channel imports so channels can reference them
# via the notifier module during circular import resolution.
from urllib.error import URLError  # noqa: F401
from urllib.request import Request, urlopen  # noqa: F401

from .base import NotificationChannel
from .channels import (
    DiscordChannel,
    EmailChannel,
    PagerDutyChannel,
    SlackChannel,
    TeamsChannel,
    WebhookChannel,
)
from .manager import Notifier
from .models import EventType, NotificationEvent

__all__ = [
    "EventType",
    "NotificationEvent",
    "NotificationChannel",
    "SlackChannel",
    "DiscordChannel",
    "WebhookChannel",
    "EmailChannel",
    "TeamsChannel",
    "PagerDutyChannel",
    "Notifier",
    "urlopen",
    "Request",
    "URLError",
]
