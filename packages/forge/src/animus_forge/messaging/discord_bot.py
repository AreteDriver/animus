"""Discord bot integration for Gorgon.

Provides two-way messaging with Discord using discord.py library.
Enables Clawdbot-style operation where Gorgon acts as a personal AI assistant.

Usage:
    from animus_forge.messaging import DiscordBot, MessageHandler

    bot = DiscordBot(token="YOUR_BOT_TOKEN")
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

# Try to import discord library
try:
    import discord
    from discord.ext import commands

    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None
    commands = None


class DiscordBot(MessagingBot):
    """Discord bot implementation for two-way messaging.

    This bot:
    - Receives messages from Discord channels and DMs
    - Routes them through Gorgon's agent system
    - Sends responses back to users
    - Supports slash commands
    - Handles attachments (images, files)

    Requires:
        pip install discord.py

    Environment:
        DISCORD_BOT_TOKEN: Your Discord bot token
        DISCORD_ALLOWED_GUILDS: Comma-separated guild IDs (optional)
    """

    def __init__(
        self,
        token: str,
        name: str = "Gorgon",
        allowed_users: list[str] | None = None,
        admin_users: list[str] | None = None,
        allowed_guilds: list[str] | None = None,
        respond_to_mentions: bool = True,
        respond_to_dms: bool = True,
        command_prefix: str = "!",
    ):
        """Initialize the Discord bot.

        Args:
            token: Discord bot token.
            name: Bot display name.
            allowed_users: List of Discord user IDs allowed to interact.
            admin_users: List of Discord user IDs with admin privileges.
            allowed_guilds: List of guild IDs the bot can operate in.
            respond_to_mentions: Whether to respond when mentioned.
            respond_to_dms: Whether to respond to direct messages.
            command_prefix: Prefix for text commands (e.g., "!help").
        """
        if not DISCORD_AVAILABLE:
            raise ImportError(
                "discord.py is not installed. Install it with: pip install discord.py"
            )

        super().__init__(name, allowed_users, admin_users)
        self.token = token
        self.allowed_guilds = set(allowed_guilds) if allowed_guilds else None
        self.respond_to_mentions = respond_to_mentions
        self.respond_to_dms = respond_to_dms
        self.command_prefix = command_prefix

        self._client: discord.Client | None = None
        self._command_handler: Any = None

        # Create intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        intents.guild_messages = True

        # Create client
        self._client = commands.Bot(
            command_prefix=command_prefix,
            intents=intents,
            help_command=None,  # We'll provide our own help
        )

        # Set up event handlers
        self._setup_handlers()

    @property
    def platform(self) -> MessagePlatform:
        """Get the messaging platform."""
        return MessagePlatform.DISCORD

    def set_command_handler(self, handler: Any) -> None:
        """Set the command handler for processing bot commands.

        Args:
            handler: MessageHandler instance with handle_command method.
        """
        self._command_handler = handler

    async def start(self) -> None:
        """Start the Discord bot."""
        if self._running:
            logger.warning("Discord bot is already running")
            return

        logger.info("Starting Discord bot...")
        self._running = True

        # Start the bot (this runs until disconnected)
        try:
            await self._client.start(self.token)
        except Exception as e:
            self._running = False
            raise e

    async def stop(self) -> None:
        """Stop the Discord bot gracefully."""
        if not self._running or not self._client:
            return

        logger.info("Stopping Discord bot...")
        self._running = False
        await self._client.close()
        logger.info("Discord bot stopped")

    def _setup_handlers(self) -> None:
        """Set up Discord event handlers."""
        if not self._client:
            return

        @self._client.event
        async def on_ready():
            logger.info(f"Discord bot logged in as {self._client.user}")
            # Set custom status
            await self._client.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.listening,
                    name="your messages",
                )
            )

        @self._client.event
        async def on_message(message: discord.Message):
            # Ignore own messages
            if message.author == self._client.user:
                return

            # Ignore bot messages
            if message.author.bot:
                return

            # Check guild restrictions
            if message.guild and self.allowed_guilds:
                if str(message.guild.id) not in self.allowed_guilds:
                    return

            # Determine if we should respond
            should_respond = False

            # Check for DM
            if isinstance(message.channel, discord.DMChannel):
                if self.respond_to_dms:
                    should_respond = True
            # Check for mention
            elif self._client.user in message.mentions:
                if self.respond_to_mentions:
                    should_respond = True
            # Check for command prefix
            elif message.content.startswith(self.command_prefix):
                should_respond = True

            if not should_respond:
                return

            # Process the message
            await self._handle_message(message)

        # Add text commands
        @self._client.command(name="help")
        async def help_command(ctx):
            await self._handle_text_command(ctx, "help", [])

        @self._client.command(name="start")
        async def start_command(ctx):
            await self._handle_text_command(ctx, "start", [])

        @self._client.command(name="new")
        async def new_command(ctx):
            await self._handle_text_command(ctx, "new", [])

        @self._client.command(name="status")
        async def status_command(ctx):
            await self._handle_text_command(ctx, "status", [])

        @self._client.command(name="history")
        async def history_command(ctx, limit: int = 5):
            await self._handle_text_command(ctx, "history", [str(limit)])

    def _create_bot_user(self, discord_user: discord.User) -> BotUser:
        """Create BotUser from Discord user object.

        Args:
            discord_user: Discord User object.

        Returns:
            BotUser instance.
        """
        return BotUser(
            id=str(discord_user.id),
            platform=MessagePlatform.DISCORD,
            username=discord_user.name,
            display_name=discord_user.display_name,
            is_admin=str(discord_user.id) in self.admin_users,
            metadata={
                "discriminator": discord_user.discriminator,
                "bot": discord_user.bot,
                "avatar_url": str(discord_user.avatar.url) if discord_user.avatar else None,
            },
        )

    def _create_bot_message(
        self,
        message: discord.Message,
        content: str | None = None,
    ) -> BotMessage:
        """Create BotMessage from Discord message.

        Args:
            message: Discord Message object.
            content: Optional content override.

        Returns:
            BotMessage instance.
        """
        user = self._create_bot_user(message.author)

        # Remove bot mention from content
        msg_content = content or message.content
        if self._client and self._client.user:
            msg_content = msg_content.replace(f"<@{self._client.user.id}>", "").strip()
            msg_content = msg_content.replace(f"<@!{self._client.user.id}>", "").strip()

        # Process attachments
        attachments = []
        for att in message.attachments:
            attachments.append(
                {
                    "type": "file",
                    "filename": att.filename,
                    "url": att.url,
                    "size": att.size,
                    "content_type": att.content_type,
                }
            )

        return BotMessage(
            id=str(message.id),
            platform=MessagePlatform.DISCORD,
            user=user,
            content=msg_content,
            chat_id=str(message.channel.id),
            timestamp=message.created_at or datetime.now(),
            reply_to_id=str(message.reference.message_id) if message.reference else None,
            attachments=attachments,
            metadata={
                "guild_id": str(message.guild.id) if message.guild else None,
                "guild_name": message.guild.name if message.guild else None,
                "channel_name": getattr(message.channel, "name", "DM"),
            },
        )

    async def _handle_message(self, message: discord.Message) -> None:
        """Handle incoming Discord message.

        Args:
            message: Discord message object.
        """
        bot_message = self._create_bot_message(message)
        await self.handle_message(bot_message)

    async def _handle_text_command(
        self,
        ctx: commands.Context,
        command: str,
        args: list[str],
    ) -> None:
        """Handle a text command.

        Args:
            ctx: Command context.
            command: Command name.
            args: Command arguments.
        """
        message = self._create_bot_message(ctx.message, ctx.message.content)

        if self._command_handler:
            response = await self._command_handler.handle_command(message, command, args)
            if response:
                await self.send_message(
                    str(ctx.channel.id),
                    response,
                    reply_to_id=str(ctx.message.id),
                )
        else:
            # Fallback
            await self.handle_message(message)

    async def send_message(
        self,
        chat_id: str,
        content: str,
        reply_to_id: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Send a message to a Discord channel.

        Args:
            chat_id: Discord channel ID.
            content: Message text.
            reply_to_id: Optional message ID to reply to.
            **kwargs: Additional options (embed, etc.).

        Returns:
            Message ID if sent successfully, None otherwise.
        """
        if not self._client:
            logger.error("Discord bot not initialized")
            return None

        try:
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))

            # Split long messages (Discord limit is 2000 characters)
            max_length = 2000
            messages_sent = []

            for i in range(0, len(content), max_length):
                chunk = content[i : i + max_length]

                # Handle reply reference
                reference = None
                if reply_to_id and not messages_sent:
                    try:
                        reference = discord.MessageReference(
                            message_id=int(reply_to_id),
                            channel_id=int(chat_id),
                        )
                    except Exception as exc:
                        logger.debug(
                            "Failed to build Discord reply reference "
                            "for chat_id=%s, reply_to_id=%s: %r",
                            chat_id,
                            reply_to_id,
                            exc,
                        )

                msg = await channel.send(
                    content=chunk,
                    reference=reference,
                    mention_author=False,
                )
                messages_sent.append(str(msg.id))

            return messages_sent[0] if messages_sent else None

        except Exception as e:
            logger.exception(f"Failed to send Discord message: {e}")
            return None

    async def send_typing(self, chat_id: str) -> None:
        """Send typing indicator to a Discord channel.

        Args:
            chat_id: Discord channel ID.
        """
        if not self._client:
            return

        try:
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            await channel.typing()
        except Exception as e:
            logger.debug(f"Failed to send typing indicator: {e}")

    async def send_embed(
        self,
        chat_id: str,
        title: str,
        description: str,
        color: int = 0x3498DB,
        fields: list[dict[str, Any]] | None = None,
        reply_to_id: str | None = None,
    ) -> str | None:
        """Send an embed message to a Discord channel.

        Args:
            chat_id: Discord channel ID.
            title: Embed title.
            description: Embed description.
            color: Embed color (hex).
            fields: Optional list of field dicts with name, value, inline.
            reply_to_id: Optional message ID to reply to.

        Returns:
            Message ID if sent successfully, None otherwise.
        """
        if not self._client:
            return None

        try:
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))

            embed = discord.Embed(
                title=title,
                description=description,
                color=color,
            )

            if fields:
                for field in fields:
                    embed.add_field(
                        name=field.get("name", ""),
                        value=field.get("value", ""),
                        inline=field.get("inline", False),
                    )

            embed.set_footer(text="Gorgon AI Assistant")

            reference = None
            if reply_to_id:
                try:
                    reference = discord.MessageReference(
                        message_id=int(reply_to_id),
                        channel_id=int(chat_id),
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to create Discord MessageReference for reply_to_id %r "
                        "in channel %r: %s",
                        reply_to_id,
                        chat_id,
                        e,
                    )
                    reference = None

            msg = await channel.send(embed=embed, reference=reference)
            return str(msg.id)

        except Exception as e:
            logger.exception(f"Failed to send Discord embed: {e}")
            return None


def create_discord_bot(
    token: str | None = None,
    allowed_users: list[str] | None = None,
    admin_users: list[str] | None = None,
    allowed_guilds: list[str] | None = None,
) -> DiscordBot:
    """Create a Discord bot from environment or provided config.

    Args:
        token: Bot token (or uses DISCORD_BOT_TOKEN env var).
        allowed_users: List of allowed user IDs.
        admin_users: List of admin user IDs.
        allowed_guilds: List of allowed guild IDs.

    Returns:
        Configured DiscordBot instance.
    """
    from animus_forge.config.settings import get_settings

    settings = get_settings()

    bot_token = token or settings.discord_bot_token
    if not bot_token:
        raise ValueError(
            "Discord bot token not provided. "
            "Set DISCORD_BOT_TOKEN environment variable or pass token parameter."
        )

    # Load from settings if not provided
    if allowed_users is None:
        allowed_env = settings.discord_allowed_users or ""
        allowed_users = [u.strip() for u in allowed_env.split(",") if u.strip()] or None

    if admin_users is None:
        admin_env = settings.discord_admin_users or ""
        admin_users = [u.strip() for u in admin_env.split(",") if u.strip()] or None

    if allowed_guilds is None:
        guilds_env = settings.discord_allowed_guilds or ""
        allowed_guilds = [g.strip() for g in guilds_env.split(",") if g.strip()] or None

    return DiscordBot(
        token=bot_token,
        allowed_users=allowed_users,
        admin_users=admin_users,
        allowed_guilds=allowed_guilds,
    )
