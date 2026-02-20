"""Telegram bot integration for Gorgon.

Provides two-way messaging with Telegram using the python-telegram-bot library.
Enables Clawdbot-style operation where Gorgon acts as a personal AI assistant.

Usage:
    from animus_forge.messaging import TelegramBot, MessageHandler

    bot = TelegramBot(token="YOUR_BOT_TOKEN")
    handler = MessageHandler(session_manager, supervisor)
    bot.set_message_callback(handler.handle_message)

    await bot.start()
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .base import BotMessage, BotUser, MessagePlatform, MessagingBot

logger = logging.getLogger(__name__)

# Try to import telegram library
try:
    from telegram import Update
    from telegram.constants import ChatAction
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        filters,
    )
    from telegram.ext import (
        MessageHandler as TGMessageHandler,
    )

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None
    Application = None
    CommandHandler = None
    ContextTypes = None
    TGMessageHandler = None
    filters = None
    ChatAction = None


class TelegramBot(MessagingBot):
    """Telegram bot implementation for two-way messaging.

    This bot:
    - Receives messages from Telegram users
    - Routes them through Gorgon's agent system
    - Sends responses back to users
    - Supports commands (/start, /help, /new, etc.)
    - Handles attachments (images, documents, etc.)

    Requires:
        pip install python-telegram-bot

    Environment:
        TELEGRAM_BOT_TOKEN: Your Telegram bot token from @BotFather
    """

    def __init__(
        self,
        token: str,
        name: str = "Gorgon",
        allowed_users: list[str] | None = None,
        admin_users: list[str] | None = None,
        parse_mode: str = "Markdown",
    ):
        """Initialize the Telegram bot.

        Args:
            token: Telegram bot token from @BotFather.
            name: Bot display name.
            allowed_users: List of Telegram user IDs allowed to interact.
            admin_users: List of Telegram user IDs with admin privileges.
            parse_mode: Message parse mode (Markdown, HTML, or None).
        """
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot is not installed. "
                "Install it with: pip install python-telegram-bot"
            )

        super().__init__(name, allowed_users, admin_users)
        self.token = token
        self.parse_mode = parse_mode
        self._app: Application | None = None
        self._command_handler: Any = None  # Will be set by MessageHandler

    @property
    def platform(self) -> MessagePlatform:
        """Get the messaging platform."""
        return MessagePlatform.TELEGRAM

    def set_command_handler(self, handler: Any) -> None:
        """Set the command handler for processing bot commands.

        Args:
            handler: MessageHandler instance with handle_command method.
        """
        self._command_handler = handler

    async def start(self) -> None:
        """Start the Telegram bot and begin polling for updates."""
        if self._running:
            logger.warning("Telegram bot is already running")
            return

        logger.info("Starting Telegram bot...")

        # Build application
        self._app = Application.builder().token(self.token).build()

        # Register handlers
        self._setup_handlers()

        # Start polling
        self._running = True
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        logger.info("Telegram bot started successfully")

    async def stop(self) -> None:
        """Stop the Telegram bot gracefully."""
        if not self._running or not self._app:
            return

        logger.info("Stopping Telegram bot...")

        self._running = False
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

        logger.info("Telegram bot stopped")

    def _setup_handlers(self) -> None:
        """Set up Telegram message and command handlers."""
        if not self._app:
            return

        # Command handlers
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("help", self._handle_help))
        self._app.add_handler(CommandHandler("new", self._handle_new))
        self._app.add_handler(CommandHandler("status", self._handle_status))
        self._app.add_handler(CommandHandler("history", self._handle_history))

        # Message handler (text messages)
        self._app.add_handler(
            TGMessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        # Photo handler
        self._app.add_handler(TGMessageHandler(filters.PHOTO, self._handle_photo))

        # Document handler
        self._app.add_handler(TGMessageHandler(filters.Document.ALL, self._handle_document))

        # Voice message handler
        self._app.add_handler(TGMessageHandler(filters.VOICE, self._handle_voice))

    def _create_bot_user(self, tg_user: Any) -> BotUser:
        """Create BotUser from Telegram user object.

        Args:
            tg_user: Telegram User object.

        Returns:
            BotUser instance.
        """
        return BotUser(
            id=str(tg_user.id),
            platform=MessagePlatform.TELEGRAM,
            username=tg_user.username,
            display_name=tg_user.full_name,
            is_admin=str(tg_user.id) in self.admin_users,
            metadata={
                "language_code": tg_user.language_code,
                "is_bot": tg_user.is_bot,
            },
        )

    def _create_bot_message(
        self,
        update: Update,
        content: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> BotMessage:
        """Create BotMessage from Telegram update.

        Args:
            update: Telegram Update object.
            content: Message content.
            attachments: Optional list of attachments.

        Returns:
            BotMessage instance.
        """
        msg = update.message or update.edited_message
        user = self._create_bot_user(msg.from_user)

        return BotMessage(
            id=str(msg.message_id),
            platform=MessagePlatform.TELEGRAM,
            user=user,
            content=content,
            chat_id=str(msg.chat_id),
            timestamp=msg.date or datetime.now(),
            reply_to_id=str(msg.reply_to_message.message_id) if msg.reply_to_message else None,
            attachments=attachments or [],
            metadata={
                "chat_type": msg.chat.type,
                "chat_title": msg.chat.title,
            },
        )

    async def _handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle incoming text message.

        Args:
            update: Telegram update.
            context: Handler context.
        """
        if not update.message or not update.message.text:
            return

        message = self._create_bot_message(update, update.message.text)
        await self.handle_message(message)

    async def _handle_photo(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle incoming photo.

        Args:
            update: Telegram update.
            context: Handler context.
        """
        if not update.message or not update.message.photo:
            return

        # Get the largest photo
        photo = update.message.photo[-1]
        file = await photo.get_file()

        attachments = [
            {
                "type": "photo",
                "file_id": photo.file_id,
                "file_unique_id": photo.file_unique_id,
                "file_path": file.file_path,
                "width": photo.width,
                "height": photo.height,
                "file_size": photo.file_size,
            }
        ]

        content = update.message.caption or "[Photo]"
        message = self._create_bot_message(update, content, attachments)
        await self.handle_message(message)

    async def _handle_document(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle incoming document.

        Args:
            update: Telegram update.
            context: Handler context.
        """
        if not update.message or not update.message.document:
            return

        doc = update.message.document
        file = await doc.get_file()

        attachments = [
            {
                "type": "document",
                "file_id": doc.file_id,
                "file_unique_id": doc.file_unique_id,
                "file_path": file.file_path,
                "file_name": doc.file_name,
                "mime_type": doc.mime_type,
                "file_size": doc.file_size,
            }
        ]

        content = update.message.caption or f"[Document: {doc.file_name}]"
        message = self._create_bot_message(update, content, attachments)
        await self.handle_message(message)

    async def _handle_voice(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle incoming voice message.

        Args:
            update: Telegram update.
            context: Handler context.
        """
        if not update.message or not update.message.voice:
            return

        voice = update.message.voice
        file = await voice.get_file()

        attachments = [
            {
                "type": "voice",
                "file_id": voice.file_id,
                "file_unique_id": voice.file_unique_id,
                "file_path": file.file_path,
                "duration": voice.duration,
                "mime_type": voice.mime_type,
                "file_size": voice.file_size,
            }
        ]

        content = "[Voice message]"
        message = self._create_bot_message(update, content, attachments)
        await self.handle_message(message)

    async def _handle_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        command: str,
    ) -> None:
        """Handle a command through the message handler.

        Args:
            update: Telegram update.
            context: Handler context.
            command: Command name.
        """
        if not update.message:
            return

        message = self._create_bot_message(update, update.message.text or "")

        if self._command_handler:
            args = context.args or []
            response = await self._command_handler.handle_command(message, command, args)
            if response:
                await self.send_message(
                    str(update.message.chat_id),
                    response,
                    reply_to_id=str(update.message.message_id),
                )
        else:
            # Fallback command handling
            await self.handle_message(message)

    async def _handle_start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /start command."""
        await self._handle_command(update, context, "start")

    async def _handle_help(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /help command."""
        await self._handle_command(update, context, "help")

    async def _handle_new(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /new command."""
        await self._handle_command(update, context, "new")

    async def _handle_status(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /status command."""
        await self._handle_command(update, context, "status")

    async def _handle_history(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /history command."""
        await self._handle_command(update, context, "history")

    async def send_message(
        self,
        chat_id: str,
        content: str,
        reply_to_id: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Send a message to a Telegram chat.

        Args:
            chat_id: Telegram chat ID.
            content: Message text.
            reply_to_id: Optional message ID to reply to.
            **kwargs: Additional options (disable_notification, etc.).

        Returns:
            Message ID if sent successfully, None otherwise.
        """
        if not self._app:
            logger.error("Telegram bot not initialized")
            return None

        try:
            # Split long messages (Telegram limit is 4096 characters)
            max_length = 4096
            messages_sent = []

            for i in range(0, len(content), max_length):
                chunk = content[i : i + max_length]

                msg = await self._app.bot.send_message(
                    chat_id=int(chat_id),
                    text=chunk,
                    parse_mode=self.parse_mode if kwargs.get("parse_mode", True) else None,
                    reply_to_message_id=int(reply_to_id)
                    if reply_to_id and not messages_sent
                    else None,
                    disable_notification=kwargs.get("disable_notification", False),
                )
                messages_sent.append(str(msg.message_id))

            return messages_sent[0] if messages_sent else None

        except Exception as e:
            logger.exception(f"Failed to send Telegram message: {e}")
            return None

    async def send_typing(self, chat_id: str) -> None:
        """Send typing indicator to a Telegram chat.

        Args:
            chat_id: Telegram chat ID.
        """
        if not self._app:
            return

        try:
            await self._app.bot.send_chat_action(
                chat_id=int(chat_id),
                action=ChatAction.TYPING,
            )
        except Exception as e:
            logger.debug(f"Failed to send typing indicator: {e}")

    async def send_photo(
        self,
        chat_id: str,
        photo: str | bytes,
        caption: str | None = None,
        reply_to_id: str | None = None,
    ) -> str | None:
        """Send a photo to a Telegram chat.

        Args:
            chat_id: Telegram chat ID.
            photo: Photo file path, URL, or bytes.
            caption: Optional photo caption.
            reply_to_id: Optional message ID to reply to.

        Returns:
            Message ID if sent successfully, None otherwise.
        """
        if not self._app:
            return None

        try:
            msg = await self._app.bot.send_photo(
                chat_id=int(chat_id),
                photo=photo,
                caption=caption,
                parse_mode=self.parse_mode,
                reply_to_message_id=int(reply_to_id) if reply_to_id else None,
            )
            return str(msg.message_id)
        except Exception as e:
            logger.exception(f"Failed to send photo: {e}")
            return None

    async def send_document(
        self,
        chat_id: str,
        document: str | bytes,
        filename: str | None = None,
        caption: str | None = None,
        reply_to_id: str | None = None,
    ) -> str | None:
        """Send a document to a Telegram chat.

        Args:
            chat_id: Telegram chat ID.
            document: Document file path or bytes.
            filename: Optional filename.
            caption: Optional document caption.
            reply_to_id: Optional message ID to reply to.

        Returns:
            Message ID if sent successfully, None otherwise.
        """
        if not self._app:
            return None

        try:
            msg = await self._app.bot.send_document(
                chat_id=int(chat_id),
                document=document,
                filename=filename,
                caption=caption,
                parse_mode=self.parse_mode,
                reply_to_message_id=int(reply_to_id) if reply_to_id else None,
            )
            return str(msg.message_id)
        except Exception as e:
            logger.exception(f"Failed to send document: {e}")
            return None


def create_telegram_bot(
    token: str | None = None,
    allowed_users: list[str] | None = None,
    admin_users: list[str] | None = None,
) -> TelegramBot:
    """Create a Telegram bot from environment or provided config.

    Args:
        token: Bot token (or uses TELEGRAM_BOT_TOKEN env var).
        allowed_users: List of allowed user IDs.
        admin_users: List of admin user IDs.

    Returns:
        Configured TelegramBot instance.
    """
    from animus_forge.config.settings import get_settings

    settings = get_settings()

    bot_token = token or settings.telegram_bot_token
    if not bot_token:
        raise ValueError(
            "Telegram bot token not provided. "
            "Set TELEGRAM_BOT_TOKEN environment variable or pass token parameter."
        )

    # Load allowed/admin users from settings if not provided
    if allowed_users is None:
        allowed_env = settings.telegram_allowed_users or ""
        allowed_users = [u.strip() for u in allowed_env.split(",") if u.strip()] or None

    if admin_users is None:
        admin_env = settings.telegram_admin_users or ""
        admin_users = [u.strip() for u in admin_env.split(",") if u.strip()] or None

    return TelegramBot(
        token=bot_token,
        allowed_users=allowed_users,
        admin_users=admin_users,
    )
