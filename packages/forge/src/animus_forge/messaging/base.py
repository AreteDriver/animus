"""Base classes for messaging bot integrations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MessagePlatform(str, Enum):
    """Supported messaging platforms."""

    TELEGRAM = "telegram"
    DISCORD = "discord"
    WHATSAPP = "whatsapp"
    SLACK = "slack"
    SIGNAL = "signal"


@dataclass
class BotUser:
    """Represents a user on a messaging platform."""

    id: str
    platform: MessagePlatform
    username: str | None = None
    display_name: str | None = None
    is_admin: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def identifier(self) -> str:
        """Get unique identifier across platforms."""
        return f"{self.platform.value}:{self.id}"


@dataclass
class BotMessage:
    """Represents an incoming or outgoing message."""

    id: str
    platform: MessagePlatform
    user: BotUser
    content: str
    chat_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to_id: str | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_attachments(self) -> bool:
        """Check if message has attachments."""
        return len(self.attachments) > 0


# Type alias for message handler callback
MessageCallback = Callable[[BotMessage], Coroutine[Any, Any, str | None]]


class MessagingBot(ABC):
    """Abstract base class for messaging bot implementations.

    Subclasses should implement:
    - start(): Initialize and start the bot
    - stop(): Gracefully stop the bot
    - send_message(): Send a message to a chat
    - _setup_handlers(): Set up platform-specific message handlers
    """

    def __init__(
        self,
        name: str = "Gorgon",
        allowed_users: list[str] | None = None,
        admin_users: list[str] | None = None,
    ):
        """Initialize the messaging bot.

        Args:
            name: Bot display name.
            allowed_users: List of user IDs allowed to interact (None = all allowed).
            admin_users: List of user IDs with admin privileges.
        """
        self.name = name
        self.allowed_users = set(allowed_users) if allowed_users else None
        self.admin_users = set(admin_users) if admin_users else set()
        self._message_callback: MessageCallback | None = None
        self._running = False

    @property
    @abstractmethod
    def platform(self) -> MessagePlatform:
        """Get the messaging platform."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the bot and begin listening for messages."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the bot gracefully."""
        pass

    @abstractmethod
    async def send_message(
        self,
        chat_id: str,
        content: str,
        reply_to_id: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Send a message to a chat.

        Args:
            chat_id: Target chat/channel ID.
            content: Message content.
            reply_to_id: Optional message ID to reply to.
            **kwargs: Platform-specific options.

        Returns:
            Message ID if sent successfully, None otherwise.
        """
        pass

    @abstractmethod
    async def send_typing(self, chat_id: str) -> None:
        """Send typing indicator to a chat.

        Args:
            chat_id: Target chat/channel ID.
        """
        pass

    def set_message_callback(self, callback: MessageCallback) -> None:
        """Set the callback function for handling incoming messages.

        Args:
            callback: Async function that receives BotMessage and returns response.
        """
        self._message_callback = callback

    def is_user_allowed(self, user: BotUser) -> bool:
        """Check if a user is allowed to interact with the bot.

        Args:
            user: The user to check.

        Returns:
            True if allowed, False otherwise.
        """
        if self.allowed_users is None:
            return True
        return user.id in self.allowed_users or user.id in self.admin_users

    def is_user_admin(self, user: BotUser) -> bool:
        """Check if a user has admin privileges.

        Args:
            user: The user to check.

        Returns:
            True if admin, False otherwise.
        """
        return user.id in self.admin_users

    async def handle_message(self, message: BotMessage) -> None:
        """Process an incoming message.

        Args:
            message: The incoming message.
        """
        # Check authorization
        if not self.is_user_allowed(message.user):
            logger.warning(
                f"Unauthorized message from {message.user.identifier}: {message.content[:50]}"
            )
            await self.send_message(
                message.chat_id,
                "Sorry, you are not authorized to use this bot.",
                reply_to_id=message.id,
            )
            return

        # Check if callback is set
        if self._message_callback is None:
            logger.error("No message callback set")
            return

        try:
            # Send typing indicator
            await self.send_typing(message.chat_id)

            # Process message through callback
            response = await self._message_callback(message)

            # Send response if any
            if response:
                await self.send_message(
                    message.chat_id,
                    response,
                    reply_to_id=message.id,
                )
        except Exception as e:
            logger.exception(f"Error handling message: {e}")
            await self.send_message(
                message.chat_id,
                f"An error occurred while processing your message: {str(e)}",
                reply_to_id=message.id,
            )

    @property
    def is_running(self) -> bool:
        """Check if the bot is currently running."""
        return self._running
