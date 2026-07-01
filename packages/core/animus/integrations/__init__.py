"""
Animus Integrations Package

External service integrations: Calendar, Email, Tasks, Filesystem, Webhooks.
"""

from animus.integrations.base import (
    AuthType,
    BaseIntegration,
    IntegrationInfo,
    IntegrationStatus,
)
from animus.integrations.filesystem import FilesystemIntegration
from animus.integrations.manager import IntegrationManager
from animus.integrations.todoist import TodoistIntegration
from animus.integrations.webhooks import WebhookIntegration

__all__ = [
    "AuthType",
    "BaseIntegration",
    "FilesystemIntegration",
    "IntegrationInfo",
    "IntegrationManager",
    "IntegrationStatus",
    "TodoistIntegration",
    "WebhookIntegration",
]

# Optional Google integrations (require google-api-python-client)
try:
    from animus.integrations.google import GoogleCalendarIntegration

    __all__.append("GoogleCalendarIntegration")
except ImportError:
    GoogleCalendarIntegration = None  # type: ignore[misc, assignment]

try:
    from animus.integrations.google.gmail import GmailIntegration

    __all__.append("GmailIntegration")
except ImportError:
    GmailIntegration = None  # type: ignore[misc, assignment]

# Optional Gorgon integration (requires httpx)
try:
    from animus.integrations.gorgon import GorgonIntegration

    __all__.append("GorgonIntegration")
except ImportError:
    GorgonIntegration = None  # type: ignore[misc, assignment]
