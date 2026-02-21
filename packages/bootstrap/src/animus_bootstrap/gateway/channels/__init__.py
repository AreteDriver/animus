"""Gateway channel adapters for external messaging platforms.

Each adapter wraps a specific platform SDK/API and provides a uniform
interface for the message gateway. All platform libraries are optional
dependencies — adapters guard their imports and raise helpful messages
when the required library is not installed.
"""

from animus_bootstrap.gateway.channels.discord_channel import DiscordAdapter
from animus_bootstrap.gateway.channels.email_channel import EmailAdapter
from animus_bootstrap.gateway.channels.matrix import MatrixAdapter
from animus_bootstrap.gateway.channels.signal_channel import SignalAdapter
from animus_bootstrap.gateway.channels.slack import SlackAdapter
from animus_bootstrap.gateway.channels.telegram import TelegramAdapter
from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

__all__ = [
    "DiscordAdapter",
    "EmailAdapter",
    "MatrixAdapter",
    "SignalAdapter",
    "SlackAdapter",
    "TelegramAdapter",
    "WhatsAppAdapter",
]
