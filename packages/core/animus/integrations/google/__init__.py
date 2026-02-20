"""
Google Integrations Package

Google Calendar and Gmail integrations.
"""

from animus.integrations.google.calendar import GoogleCalendarIntegration

__all__ = [
    "GoogleCalendarIntegration",
]

# Gmail added after implementation
try:
    from animus.integrations.google import gmail as _gmail_module

    GmailIntegration = _gmail_module.GmailIntegration
    __all__.append("GmailIntegration")
except ImportError:
    GmailIntegration = None  # type: ignore[misc, assignment]
