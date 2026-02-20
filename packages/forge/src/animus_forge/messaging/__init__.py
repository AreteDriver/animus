"""Messaging module for bot communication.

Provides integrations with messaging platforms like Telegram and Discord
for sending notifications and alerts.

Supported Platforms:
    - Telegram: Full support via python-telegram-bot
    - Discord: Full support via discord.py
"""

from .base import BotMessage, BotUser, MessagePlatform, MessagingBot
from .discord_bot import DiscordBot, create_discord_bot
from .telegram_bot import TelegramBot, create_telegram_bot

__all__ = [
    # Base classes
    "MessagingBot",
    "BotMessage",
    "BotUser",
    "MessagePlatform",
    # Telegram
    "TelegramBot",
    "create_telegram_bot",
    # Discord
    "DiscordBot",
    "create_discord_bot",
]
