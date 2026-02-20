"""Tests for messaging module.

Comprehensively tests base classes, message handler, Discord bot, and Telegram bot
with all external dependencies mocked.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_forge.messaging.base import (
    BotMessage,
    BotUser,
    MessagePlatform,
    MessagingBot,
)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_user(**overrides) -> BotUser:
    """Create a BotUser with sensible defaults."""
    defaults = dict(
        id="user-123",
        platform=MessagePlatform.TELEGRAM,
        username="testuser",
        display_name="Test User",
        is_admin=False,
        metadata={},
    )
    defaults.update(overrides)
    return BotUser(**defaults)


def _make_message(**overrides) -> BotMessage:
    """Create a BotMessage with sensible defaults."""
    user = overrides.pop("user", None) or _make_user()
    defaults = dict(
        id="msg-456",
        platform=MessagePlatform.TELEGRAM,
        user=user,
        content="Hello, bot!",
        chat_id="chat-789",
        timestamp=datetime(2026, 2, 12, 10, 0),
        reply_to_id=None,
        attachments=[],
        metadata={},
    )
    defaults.update(overrides)
    return BotMessage(**defaults)


# ---------------------------------------------------------------------------
# Concrete bot subclass for testing abstract MessagingBot
# ---------------------------------------------------------------------------


class _StubBot(MessagingBot):
    """Concrete stub to test the abstract base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sent_messages: list[tuple] = []
        self.typing_sent: list[str] = []

    @property
    def platform(self) -> MessagePlatform:
        return MessagePlatform.TELEGRAM

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def send_message(self, chat_id, content, reply_to_id=None, **kwargs) -> str | None:
        self.sent_messages.append((chat_id, content, reply_to_id))
        return "sent-id"

    async def send_typing(self, chat_id) -> None:
        self.typing_sent.append(chat_id)


# ===========================================================================
# 1. base.py tests
# ===========================================================================


class TestMessagePlatform:
    """Tests for MessagePlatform enum."""

    def test_all_platforms_exist(self):
        assert MessagePlatform.TELEGRAM == "telegram"
        assert MessagePlatform.DISCORD == "discord"
        assert MessagePlatform.WHATSAPP == "whatsapp"
        assert MessagePlatform.SLACK == "slack"
        assert MessagePlatform.SIGNAL == "signal"

    def test_is_str_enum(self):
        assert isinstance(MessagePlatform.TELEGRAM, str)

    def test_platform_count(self):
        assert len(MessagePlatform) == 5


class TestBotUser:
    """Tests for BotUser dataclass."""

    def test_defaults(self):
        user = BotUser(id="1", platform=MessagePlatform.DISCORD)
        assert user.username is None
        assert user.display_name is None
        assert user.is_admin is False
        assert user.metadata == {}

    def test_full_construction(self):
        user = _make_user(is_admin=True, metadata={"lang": "en"})
        assert user.id == "user-123"
        assert user.platform == MessagePlatform.TELEGRAM
        assert user.username == "testuser"
        assert user.display_name == "Test User"
        assert user.is_admin is True
        assert user.metadata == {"lang": "en"}

    def test_identifier_property(self):
        user = _make_user(id="42", platform=MessagePlatform.DISCORD)
        assert user.identifier == "discord:42"

    def test_identifier_telegram(self):
        user = _make_user(id="99", platform=MessagePlatform.TELEGRAM)
        assert user.identifier == "telegram:99"

    def test_identifier_each_platform(self):
        for p in list(MessagePlatform):
            user = _make_user(id="x", platform=p)
            assert user.identifier == f"{p.value}:x"


class TestBotMessage:
    """Tests for BotMessage dataclass."""

    def test_defaults(self):
        user = _make_user()
        msg = BotMessage(
            id="1",
            platform=MessagePlatform.TELEGRAM,
            user=user,
            content="hi",
            chat_id="c1",
        )
        assert msg.reply_to_id is None
        assert msg.attachments == []
        assert msg.metadata == {}
        assert isinstance(msg.timestamp, datetime)

    def test_full_construction(self):
        msg = _make_message(
            attachments=[{"type": "photo", "url": "http://example.com"}],
            reply_to_id="orig-1",
            metadata={"key": "val"},
        )
        assert msg.content == "Hello, bot!"
        assert msg.reply_to_id == "orig-1"
        assert len(msg.attachments) == 1

    def test_has_attachments_true(self):
        msg = _make_message(attachments=[{"type": "file"}])
        assert msg.has_attachments is True

    def test_has_attachments_false(self):
        msg = _make_message(attachments=[])
        assert msg.has_attachments is False

    def test_has_attachments_multiple(self):
        msg = _make_message(attachments=[{"type": "a"}, {"type": "b"}])
        assert msg.has_attachments is True


class TestMessagingBot:
    """Tests for the abstract MessagingBot via _StubBot."""

    def test_init_defaults(self):
        bot = _StubBot()
        assert bot.name == "Gorgon"
        assert bot.allowed_users is None
        assert bot.admin_users == set()
        assert bot._message_callback is None
        assert bot._running is False

    def test_init_custom(self):
        bot = _StubBot(
            name="Custom",
            allowed_users=["u1", "u2"],
            admin_users=["a1"],
        )
        assert bot.name == "Custom"
        assert bot.allowed_users == {"u1", "u2"}
        assert bot.admin_users == {"a1"}

    def test_is_running(self):
        bot = _StubBot()
        assert bot.is_running is False

        async def _test():
            await bot.start()
            assert bot.is_running is True
            await bot.stop()
            assert bot.is_running is False

        asyncio.run(_test())

    def test_platform_property(self):
        bot = _StubBot()
        assert bot.platform == MessagePlatform.TELEGRAM

    def test_set_message_callback(self):
        bot = _StubBot()
        cb = AsyncMock()
        bot.set_message_callback(cb)
        assert bot._message_callback is cb

    # -- is_user_allowed --

    def test_is_user_allowed_no_restriction(self):
        bot = _StubBot()
        assert bot.is_user_allowed(_make_user(id="anyone")) is True

    def test_is_user_allowed_in_allowed_set(self):
        bot = _StubBot(allowed_users=["u1", "u2"])
        assert bot.is_user_allowed(_make_user(id="u1")) is True

    def test_is_user_not_allowed(self):
        bot = _StubBot(allowed_users=["u1"])
        assert bot.is_user_allowed(_make_user(id="u99")) is False

    def test_is_user_allowed_admin_bypass(self):
        bot = _StubBot(allowed_users=["u1"], admin_users=["a1"])
        assert bot.is_user_allowed(_make_user(id="a1")) is True

    # -- is_user_admin --

    def test_is_user_admin_true(self):
        bot = _StubBot(admin_users=["a1"])
        assert bot.is_user_admin(_make_user(id="a1")) is True

    def test_is_user_admin_false(self):
        bot = _StubBot(admin_users=["a1"])
        assert bot.is_user_admin(_make_user(id="u1")) is False

    def test_is_user_admin_empty(self):
        bot = _StubBot()
        assert bot.is_user_admin(_make_user(id="u1")) is False

    # -- handle_message --

    def test_handle_message_unauthorized(self):
        async def _test():
            bot = _StubBot(allowed_users=["u1"])
            msg = _make_message(user=_make_user(id="bad"))
            await bot.handle_message(msg)
            assert len(bot.sent_messages) == 1
            _, content, _ = bot.sent_messages[0]
            assert "not authorized" in content

        asyncio.run(_test())

    def test_handle_message_no_callback(self):
        async def _test():
            bot = _StubBot()
            msg = _make_message()
            await bot.handle_message(msg)
            # No callback set -> no message sent, no crash
            assert bot.sent_messages == []

        asyncio.run(_test())

    def test_handle_message_success(self):
        async def _test():
            bot = _StubBot()
            cb = AsyncMock(return_value="response text")
            bot.set_message_callback(cb)
            msg = _make_message(chat_id="c1")
            await bot.handle_message(msg)
            cb.assert_awaited_once_with(msg)
            assert ("c1", "response text", "msg-456") in bot.sent_messages

        asyncio.run(_test())

    def test_handle_message_callback_returns_none(self):
        async def _test():
            bot = _StubBot()
            cb = AsyncMock(return_value=None)
            bot.set_message_callback(cb)
            msg = _make_message()
            await bot.handle_message(msg)
            # Typing sent but no response message
            assert len(bot.typing_sent) == 1
            assert len(bot.sent_messages) == 0

        asyncio.run(_test())

    def test_handle_message_callback_returns_empty(self):
        async def _test():
            bot = _StubBot()
            cb = AsyncMock(return_value="")
            bot.set_message_callback(cb)
            msg = _make_message()
            await bot.handle_message(msg)
            # Empty string is falsy -> no response sent
            assert len(bot.sent_messages) == 0

        asyncio.run(_test())

    def test_handle_message_callback_error(self):
        async def _test():
            bot = _StubBot()
            cb = AsyncMock(side_effect=ValueError("boom"))
            bot.set_message_callback(cb)
            msg = _make_message(chat_id="c1")
            await bot.handle_message(msg)
            # Should send error message
            assert len(bot.sent_messages) == 1
            _, content, _ = bot.sent_messages[0]
            assert "error occurred" in content.lower()
            assert "boom" in content

        asyncio.run(_test())

    def test_handle_message_sends_typing(self):
        async def _test():
            bot = _StubBot()
            cb = AsyncMock(return_value="ok")
            bot.set_message_callback(cb)
            msg = _make_message(chat_id="c1")
            await bot.handle_message(msg)
            assert "c1" in bot.typing_sent

        asyncio.run(_test())


# ===========================================================================
# 2. handler.py — removed (chat moved to Animus)
# ===========================================================================

# MessageHandler tests removed with chat module (moved to Animus)


# ===========================================================================
# 3. discord_bot.py tests
# ===========================================================================


class TestDiscordBotImportGuard:
    """Test the import availability guard."""

    def test_discord_not_available_raises(self):
        with patch.dict(
            "sys.modules",
            {"discord": None, "discord.ext": None, "discord.ext.commands": None},
        ):
            with patch("animus_forge.messaging.discord_bot.DISCORD_AVAILABLE", False):
                from animus_forge.messaging.discord_bot import DiscordBot

                with pytest.raises(ImportError, match="discord.py is not installed"):
                    DiscordBot(token="fake")


class TestDiscordBot:
    """Tests for DiscordBot with discord.py fully mocked."""

    @pytest.fixture(autouse=True)
    def mock_discord(self):
        """Mock the discord library at the module level."""
        mock_intents = MagicMock()
        mock_intents_cls = MagicMock(return_value=mock_intents)
        mock_intents_cls.default.return_value = mock_intents

        mock_bot_instance = MagicMock()
        mock_bot_instance.user = MagicMock()
        mock_bot_instance.user.id = 12345
        mock_bot_instance.event = MagicMock(side_effect=lambda f: f)
        mock_bot_instance.command = MagicMock(side_effect=lambda **kw: lambda f: f)
        mock_bot_instance.start = AsyncMock()
        mock_bot_instance.close = AsyncMock()
        mock_bot_instance.get_channel = MagicMock(return_value=None)
        mock_bot_instance.fetch_channel = AsyncMock()
        mock_bot_instance.change_presence = AsyncMock()

        mock_commands = MagicMock()
        mock_commands.Bot.return_value = mock_bot_instance

        mock_discord_mod = MagicMock()
        mock_discord_mod.Intents = mock_intents_cls
        mock_discord_mod.DMChannel = type("DMChannel", (), {})
        mock_discord_mod.Activity.return_value = MagicMock()
        mock_discord_mod.ActivityType.listening = "listening"
        mock_discord_mod.MessageReference.return_value = MagicMock()
        mock_discord_mod.Embed.return_value = MagicMock()

        with (
            patch("animus_forge.messaging.discord_bot.DISCORD_AVAILABLE", True),
            patch("animus_forge.messaging.discord_bot.discord", mock_discord_mod),
            patch("animus_forge.messaging.discord_bot.commands", mock_commands),
        ):
            self.mock_discord_mod = mock_discord_mod
            self.mock_commands = mock_commands
            self.mock_bot_instance = mock_bot_instance
            yield

    def _make_bot(self, **kwargs):
        from animus_forge.messaging.discord_bot import DiscordBot

        defaults = dict(token="test-token")
        defaults.update(kwargs)
        return DiscordBot(**defaults)

    def test_init_defaults(self):
        bot = self._make_bot()
        assert bot.token == "test-token"
        assert bot.name == "Gorgon"
        assert bot.allowed_guilds is None
        assert bot.respond_to_mentions is True
        assert bot.respond_to_dms is True
        assert bot.command_prefix == "!"

    def test_init_custom(self):
        bot = self._make_bot(
            name="Custom",
            allowed_guilds=["g1", "g2"],
            respond_to_mentions=False,
            respond_to_dms=False,
            command_prefix="/",
        )
        assert bot.name == "Custom"
        assert bot.allowed_guilds == {"g1", "g2"}
        assert bot.respond_to_mentions is False
        assert bot.respond_to_dms is False
        assert bot.command_prefix == "/"

    def test_platform_is_discord(self):
        bot = self._make_bot()
        assert bot.platform == MessagePlatform.DISCORD

    def test_set_command_handler(self):
        bot = self._make_bot()
        ch = MagicMock()
        bot.set_command_handler(ch)
        assert bot._command_handler is ch

    def test_start_sets_running(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            assert bot._running is True

        asyncio.run(_test())

    def test_start_already_running(self):
        async def _test():
            bot = self._make_bot()
            bot._running = True
            await bot.start()
            # Should not call client.start again
            self.mock_bot_instance.start.assert_not_awaited()

        asyncio.run(_test())

    def test_start_failure(self):
        async def _test():
            self.mock_bot_instance.start.side_effect = ConnectionError("fail")
            bot = self._make_bot()
            with pytest.raises(ConnectionError):
                await bot.start()
            assert bot._running is False

        asyncio.run(_test())

    def test_stop(self):
        async def _test():
            bot = self._make_bot()
            bot._running = True
            await bot.stop()
            assert bot._running is False
            self.mock_bot_instance.close.assert_awaited_once()

        asyncio.run(_test())

    def test_stop_not_running(self):
        async def _test():
            bot = self._make_bot()
            bot._running = False
            await bot.stop()
            self.mock_bot_instance.close.assert_not_awaited()

        asyncio.run(_test())

    # -- _create_bot_user --

    def test_create_bot_user(self):
        bot = self._make_bot(admin_users=["999"])
        discord_user = MagicMock()
        discord_user.id = 999
        discord_user.name = "testuser"
        discord_user.display_name = "Test User"
        discord_user.discriminator = "1234"
        discord_user.bot = False
        discord_user.avatar = MagicMock()
        discord_user.avatar.url = "https://cdn.discord.com/avatar.png"
        user = bot._create_bot_user(discord_user)
        assert user.id == "999"
        assert user.platform == MessagePlatform.DISCORD
        assert user.username == "testuser"
        assert user.display_name == "Test User"
        assert user.is_admin is True
        assert user.metadata["discriminator"] == "1234"

    def test_create_bot_user_no_avatar(self):
        bot = self._make_bot()
        discord_user = MagicMock()
        discord_user.id = 1
        discord_user.name = "u"
        discord_user.display_name = "U"
        discord_user.discriminator = "0"
        discord_user.bot = False
        discord_user.avatar = None
        user = bot._create_bot_user(discord_user)
        assert user.metadata["avatar_url"] is None

    def test_create_bot_user_not_admin(self):
        bot = self._make_bot(admin_users=["999"])
        discord_user = MagicMock()
        discord_user.id = 1
        discord_user.name = "u"
        discord_user.display_name = "U"
        discord_user.discriminator = "0"
        discord_user.bot = False
        discord_user.avatar = None
        user = bot._create_bot_user(discord_user)
        assert user.is_admin is False

    # -- _create_bot_message --

    def test_create_bot_message(self):
        bot = self._make_bot()
        msg = MagicMock()
        msg.id = 111
        msg.author = MagicMock()
        msg.author.id = 42
        msg.author.name = "u"
        msg.author.display_name = "U"
        msg.author.discriminator = "0"
        msg.author.bot = False
        msg.author.avatar = None
        msg.content = "Hello <@12345> world"
        msg.channel = MagicMock()
        msg.channel.id = 99
        msg.channel.name = "general"
        msg.created_at = datetime(2026, 1, 1)
        msg.reference = None
        msg.attachments = []
        msg.guild = MagicMock()
        msg.guild.id = 555
        msg.guild.name = "Test Guild"

        bot_msg = bot._create_bot_message(msg)
        assert bot_msg.id == "111"
        assert bot_msg.platform == MessagePlatform.DISCORD
        # Bot mention removed
        assert "<@12345>" not in bot_msg.content
        assert "Hello" in bot_msg.content
        assert "world" in bot_msg.content
        assert bot_msg.chat_id == "99"
        assert bot_msg.metadata["guild_id"] == "555"

    def test_create_bot_message_with_content_override(self):
        bot = self._make_bot()
        msg = MagicMock()
        msg.id = 1
        msg.author = MagicMock()
        msg.author.id = 2
        msg.author.name = "u"
        msg.author.display_name = "U"
        msg.author.discriminator = "0"
        msg.author.bot = False
        msg.author.avatar = None
        msg.channel = MagicMock()
        msg.channel.id = 3
        msg.channel.name = "ch"
        msg.created_at = None
        msg.reference = None
        msg.attachments = []
        msg.guild = None
        msg.content = "original"

        bot_msg = bot._create_bot_message(msg, content="override")
        assert "override" in bot_msg.content
        assert bot_msg.metadata["guild_id"] is None

    def test_create_bot_message_with_reply(self):
        bot = self._make_bot()
        msg = MagicMock()
        msg.id = 1
        msg.author = MagicMock()
        msg.author.id = 2
        msg.author.name = "u"
        msg.author.display_name = "U"
        msg.author.discriminator = "0"
        msg.author.bot = False
        msg.author.avatar = None
        msg.content = "reply msg"
        msg.channel = MagicMock()
        msg.channel.id = 3
        msg.channel.name = "ch"
        msg.created_at = datetime(2026, 1, 1)
        msg.reference = MagicMock()
        msg.reference.message_id = 777
        msg.attachments = []
        msg.guild = None

        bot_msg = bot._create_bot_message(msg)
        assert bot_msg.reply_to_id == "777"

    def test_create_bot_message_with_attachments(self):
        bot = self._make_bot()
        att = MagicMock()
        att.filename = "test.png"
        att.url = "https://cdn.discord.com/test.png"
        att.size = 1024
        att.content_type = "image/png"

        msg = MagicMock()
        msg.id = 1
        msg.author = MagicMock()
        msg.author.id = 2
        msg.author.name = "u"
        msg.author.display_name = "U"
        msg.author.discriminator = "0"
        msg.author.bot = False
        msg.author.avatar = None
        msg.content = "with file"
        msg.channel = MagicMock()
        msg.channel.id = 3
        msg.channel.name = "ch"
        msg.created_at = datetime(2026, 1, 1)
        msg.reference = None
        msg.attachments = [att]
        msg.guild = None

        bot_msg = bot._create_bot_message(msg)
        assert len(bot_msg.attachments) == 1
        assert bot_msg.attachments[0]["filename"] == "test.png"
        assert bot_msg.has_attachments is True

    def test_create_bot_message_removes_nick_mention(self):
        bot = self._make_bot()
        msg = MagicMock()
        msg.id = 1
        msg.author = MagicMock()
        msg.author.id = 2
        msg.author.name = "u"
        msg.author.display_name = "U"
        msg.author.discriminator = "0"
        msg.author.bot = False
        msg.author.avatar = None
        msg.content = "Hey <@!12345> help me"
        msg.channel = MagicMock()
        msg.channel.id = 3
        msg.channel.name = "ch"
        msg.created_at = datetime(2026, 1, 1)
        msg.reference = None
        msg.attachments = []
        msg.guild = None

        bot_msg = bot._create_bot_message(msg)
        assert "<@!12345>" not in bot_msg.content

    # -- send_message --

    def test_send_message_success(self):
        async def _test():
            bot = self._make_bot()
            channel = AsyncMock()
            sent_msg = MagicMock()
            sent_msg.id = 999
            channel.send = AsyncMock(return_value=sent_msg)
            self.mock_bot_instance.get_channel.return_value = channel
            result = await bot.send_message("100", "hello")
            assert result == "999"
            channel.send.assert_awaited_once()

        asyncio.run(_test())

    def test_send_message_fetch_channel(self):
        async def _test():
            bot = self._make_bot()
            self.mock_bot_instance.get_channel.return_value = None
            channel = AsyncMock()
            sent_msg = MagicMock()
            sent_msg.id = 888
            channel.send = AsyncMock(return_value=sent_msg)
            self.mock_bot_instance.fetch_channel = AsyncMock(return_value=channel)
            result = await bot.send_message("100", "hello")
            assert result == "888"

        asyncio.run(_test())

    def test_send_message_long_content_splits(self):
        async def _test():
            bot = self._make_bot()
            channel = AsyncMock()
            sent = MagicMock()
            sent.id = 1
            channel.send = AsyncMock(return_value=sent)
            self.mock_bot_instance.get_channel.return_value = channel
            long_content = "x" * 4500  # > 2000 Discord limit
            await bot.send_message("100", long_content)
            assert channel.send.await_count == 3  # ceil(4500/2000) = 3

        asyncio.run(_test())

    def test_send_message_with_reply(self):
        async def _test():
            bot = self._make_bot()
            channel = AsyncMock()
            sent = MagicMock()
            sent.id = 1
            channel.send = AsyncMock(return_value=sent)
            self.mock_bot_instance.get_channel.return_value = channel
            await bot.send_message("100", "hello", reply_to_id="50")
            # MessageReference should be created
            self.mock_discord_mod.MessageReference.assert_called_once()

        asyncio.run(_test())

    def test_send_message_no_client(self):
        async def _test():
            bot = self._make_bot()
            bot._client = None
            result = await bot.send_message("100", "hello")
            assert result is None

        asyncio.run(_test())

    def test_send_message_exception(self):
        async def _test():
            bot = self._make_bot()
            self.mock_bot_instance.get_channel.side_effect = Exception("fail")
            result = await bot.send_message("100", "hello")
            assert result is None

        asyncio.run(_test())

    # -- send_typing --

    def test_send_typing(self):
        async def _test():
            bot = self._make_bot()
            channel = AsyncMock()
            self.mock_bot_instance.get_channel.return_value = channel
            await bot.send_typing("100")
            channel.typing.assert_awaited_once()

        asyncio.run(_test())

    def test_send_typing_fetch_channel(self):
        async def _test():
            bot = self._make_bot()
            self.mock_bot_instance.get_channel.return_value = None
            channel = AsyncMock()
            self.mock_bot_instance.fetch_channel = AsyncMock(return_value=channel)
            await bot.send_typing("100")
            channel.typing.assert_awaited_once()

        asyncio.run(_test())

    def test_send_typing_no_client(self):
        async def _test():
            bot = self._make_bot()
            bot._client = None
            await bot.send_typing("100")  # Should not raise

        asyncio.run(_test())

    def test_send_typing_exception(self):
        async def _test():
            bot = self._make_bot()
            self.mock_bot_instance.get_channel.side_effect = Exception("fail")
            await bot.send_typing("100")  # Should not raise

        asyncio.run(_test())

    # -- send_embed --

    def test_send_embed_success(self):
        async def _test():
            bot = self._make_bot()
            channel = AsyncMock()
            sent_msg = MagicMock()
            sent_msg.id = 555
            channel.send = AsyncMock(return_value=sent_msg)
            self.mock_bot_instance.get_channel.return_value = channel
            result = await bot.send_embed(
                "100",
                "Title",
                "Description",
                fields=[{"name": "F1", "value": "V1", "inline": True}],
            )
            assert result == "555"
            self.mock_discord_mod.Embed.assert_called_once()

        asyncio.run(_test())

    def test_send_embed_with_reply(self):
        async def _test():
            bot = self._make_bot()
            channel = AsyncMock()
            sent_msg = MagicMock()
            sent_msg.id = 1
            channel.send = AsyncMock(return_value=sent_msg)
            self.mock_bot_instance.get_channel.return_value = channel
            await bot.send_embed("100", "T", "D", reply_to_id="50")
            self.mock_discord_mod.MessageReference.assert_called()

        asyncio.run(_test())

    def test_send_embed_no_client(self):
        async def _test():
            bot = self._make_bot()
            bot._client = None
            result = await bot.send_embed("100", "T", "D")
            assert result is None

        asyncio.run(_test())

    def test_send_embed_exception(self):
        async def _test():
            bot = self._make_bot()
            self.mock_bot_instance.get_channel.side_effect = Exception("fail")
            result = await bot.send_embed("100", "T", "D")
            assert result is None

        asyncio.run(_test())

    def test_send_embed_no_fields(self):
        async def _test():
            bot = self._make_bot()
            channel = AsyncMock()
            sent_msg = MagicMock()
            sent_msg.id = 1
            channel.send = AsyncMock(return_value=sent_msg)
            self.mock_bot_instance.get_channel.return_value = channel
            await bot.send_embed("100", "T", "D", fields=None)
            # Should not crash without fields

        asyncio.run(_test())

    def test_send_embed_reply_reference_failure(self):
        async def _test():
            bot = self._make_bot()
            channel = AsyncMock()
            sent_msg = MagicMock()
            sent_msg.id = 1
            channel.send = AsyncMock(return_value=sent_msg)
            self.mock_bot_instance.get_channel.return_value = channel
            self.mock_discord_mod.MessageReference.side_effect = ValueError("bad ref")
            result = await bot.send_embed("100", "T", "D", reply_to_id="50")
            # Should still send embed without reference
            assert result == "1"

        asyncio.run(_test())

    # -- _handle_text_command --

    def test_handle_text_command_with_handler(self):
        async def _test():
            bot = self._make_bot()
            cmd_handler = AsyncMock()
            cmd_handler.handle_command = AsyncMock(return_value="cmd response")
            bot.set_command_handler(cmd_handler)

            ctx = MagicMock()
            ctx.message = MagicMock()
            ctx.message.id = 1
            ctx.message.author = MagicMock()
            ctx.message.author.id = 2
            ctx.message.author.name = "u"
            ctx.message.author.display_name = "U"
            ctx.message.author.discriminator = "0"
            ctx.message.author.bot = False
            ctx.message.author.avatar = None
            ctx.message.content = "!help"
            ctx.message.channel = MagicMock()
            ctx.message.channel.id = 3
            ctx.message.channel.name = "ch"
            ctx.message.created_at = datetime(2026, 1, 1)
            ctx.message.reference = None
            ctx.message.attachments = []
            ctx.message.guild = None
            ctx.channel = MagicMock()
            ctx.channel.id = 3

            await bot._handle_text_command(ctx, "help", [])
            cmd_handler.handle_command.assert_awaited_once()

        asyncio.run(_test())

    def test_handle_text_command_no_handler_falls_back(self):
        async def _test():
            bot = self._make_bot()
            bot._command_handler = None
            cb = AsyncMock(return_value=None)
            bot.set_message_callback(cb)

            ctx = MagicMock()
            ctx.message = MagicMock()
            ctx.message.id = 1
            ctx.message.author = MagicMock()
            ctx.message.author.id = 2
            ctx.message.author.name = "u"
            ctx.message.author.display_name = "U"
            ctx.message.author.discriminator = "0"
            ctx.message.author.bot = False
            ctx.message.author.avatar = None
            ctx.message.content = "!help"
            ctx.message.channel = MagicMock()
            ctx.message.channel.id = 3
            ctx.message.channel.name = "ch"
            ctx.message.created_at = datetime(2026, 1, 1)
            ctx.message.reference = None
            ctx.message.attachments = []
            ctx.message.guild = None

            await bot._handle_text_command(ctx, "help", [])
            # Falls back to handle_message which calls the callback
            cb.assert_awaited_once()

        asyncio.run(_test())

    # -- _handle_message --

    def test_handle_discord_message(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock(return_value="ok")
            bot.set_message_callback(cb)

            discord_msg = MagicMock()
            discord_msg.id = 1
            discord_msg.author = MagicMock()
            discord_msg.author.id = 2
            discord_msg.author.name = "u"
            discord_msg.author.display_name = "U"
            discord_msg.author.discriminator = "0"
            discord_msg.author.bot = False
            discord_msg.author.avatar = None
            discord_msg.content = "test"
            discord_msg.channel = MagicMock()
            discord_msg.channel.id = 3
            discord_msg.channel.name = "ch"
            discord_msg.created_at = datetime(2026, 1, 1)
            discord_msg.reference = None
            discord_msg.attachments = []
            discord_msg.guild = None

            await bot._handle_message(discord_msg)
            cb.assert_awaited_once()

        asyncio.run(_test())


class TestCreateDiscordBot:
    """Tests for create_discord_bot factory function."""

    @pytest.fixture(autouse=True)
    def mock_discord_available(self):
        mock_intents = MagicMock()
        mock_intents_cls = MagicMock()
        mock_intents_cls.default.return_value = mock_intents
        mock_bot = MagicMock()
        mock_bot.event = MagicMock(side_effect=lambda f: f)
        mock_bot.command = MagicMock(side_effect=lambda **kw: lambda f: f)
        mock_commands = MagicMock()
        mock_commands.Bot.return_value = mock_bot
        mock_discord_mod = MagicMock()
        mock_discord_mod.Intents = mock_intents_cls

        with (
            patch("animus_forge.messaging.discord_bot.DISCORD_AVAILABLE", True),
            patch("animus_forge.messaging.discord_bot.discord", mock_discord_mod),
            patch("animus_forge.messaging.discord_bot.commands", mock_commands),
        ):
            yield

    def test_create_with_token(self):
        from animus_forge.messaging.discord_bot import create_discord_bot

        bot = create_discord_bot(token="test-token")
        assert bot.token == "test-token"

    def test_create_from_env(self):
        from animus_forge.messaging.discord_bot import create_discord_bot

        mock_settings = MagicMock()
        mock_settings.discord_bot_token = "env-token"
        mock_settings.discord_allowed_users = None
        mock_settings.discord_admin_users = None
        mock_settings.discord_allowed_guilds = None
        with patch("animus_forge.config.settings.get_settings", return_value=mock_settings):
            bot = create_discord_bot()
        assert bot.token == "env-token"

    def test_create_no_token_raises(self):
        from animus_forge.messaging.discord_bot import create_discord_bot

        mock_settings = MagicMock()
        mock_settings.discord_bot_token = None
        with patch("animus_forge.config.settings.get_settings", return_value=mock_settings):
            with pytest.raises(ValueError, match="bot token not provided"):
                create_discord_bot()

    def test_create_with_allowed_users(self):
        from animus_forge.messaging.discord_bot import create_discord_bot

        bot = create_discord_bot(
            token="t",
            allowed_users=["u1", "u2"],
            admin_users=["a1"],
            allowed_guilds=["g1"],
        )
        assert bot.allowed_users == {"u1", "u2"}
        assert bot.admin_users == {"a1"}
        assert bot.allowed_guilds == {"g1"}

    def test_create_loads_users_from_env(self):
        from animus_forge.messaging.discord_bot import create_discord_bot

        mock_settings = MagicMock()
        mock_settings.discord_bot_token = "t"
        mock_settings.discord_allowed_users = "u1, u2"
        mock_settings.discord_admin_users = "a1"
        mock_settings.discord_allowed_guilds = "g1, g2"
        with patch("animus_forge.config.settings.get_settings", return_value=mock_settings):
            bot = create_discord_bot()
        assert bot.allowed_users == {"u1", "u2"}
        assert bot.admin_users == {"a1"}
        assert bot.allowed_guilds == {"g1", "g2"}

    def test_create_empty_env_vars(self):
        from animus_forge.messaging.discord_bot import create_discord_bot

        mock_settings = MagicMock()
        mock_settings.discord_bot_token = "t"
        mock_settings.discord_allowed_users = ""
        mock_settings.discord_admin_users = ""
        mock_settings.discord_allowed_guilds = ""
        with patch("animus_forge.config.settings.get_settings", return_value=mock_settings):
            bot = create_discord_bot()
        assert bot.allowed_users is None
        assert bot.admin_users == set()
        assert bot.allowed_guilds is None


# ===========================================================================
# 4. telegram_bot.py tests
# ===========================================================================


class TestTelegramBotImportGuard:
    """Test the import availability guard."""

    def test_telegram_not_available_raises(self):
        with patch("animus_forge.messaging.telegram_bot.TELEGRAM_AVAILABLE", False):
            from animus_forge.messaging.telegram_bot import TelegramBot

            with pytest.raises(ImportError, match="python-telegram-bot is not installed"):
                TelegramBot(token="fake")


class TestTelegramBot:
    """Tests for TelegramBot with python-telegram-bot fully mocked."""

    @pytest.fixture(autouse=True)
    def mock_telegram(self):
        """Mock telegram library at module level."""
        mock_app_instance = MagicMock()
        mock_app_instance.initialize = AsyncMock()
        mock_app_instance.start = AsyncMock()
        mock_app_instance.stop = AsyncMock()
        mock_app_instance.shutdown = AsyncMock()
        mock_updater = MagicMock()
        mock_updater.start_polling = AsyncMock()
        mock_updater.stop = AsyncMock()
        mock_app_instance.updater = mock_updater

        mock_bot_obj = AsyncMock()
        mock_app_instance.bot = mock_bot_obj

        mock_builder = MagicMock()
        mock_builder.token.return_value = mock_builder
        mock_builder.build.return_value = mock_app_instance

        mock_application = MagicMock()
        mock_application.builder.return_value = mock_builder

        mock_command_handler = MagicMock()
        mock_tg_message_handler = MagicMock()
        mock_filters = MagicMock()
        mock_chat_action = MagicMock()
        mock_chat_action.TYPING = "typing"

        with (
            patch("animus_forge.messaging.telegram_bot.TELEGRAM_AVAILABLE", True),
            patch("animus_forge.messaging.telegram_bot.Application", mock_application),
            patch("animus_forge.messaging.telegram_bot.CommandHandler", mock_command_handler),
            patch(
                "animus_forge.messaging.telegram_bot.TGMessageHandler",
                mock_tg_message_handler,
            ),
            patch("animus_forge.messaging.telegram_bot.filters", mock_filters),
            patch("animus_forge.messaging.telegram_bot.ChatAction", mock_chat_action),
            patch("animus_forge.messaging.telegram_bot.Update", MagicMock()),
        ):
            self.mock_app_instance = mock_app_instance
            self.mock_application = mock_application
            self.mock_bot_obj = mock_bot_obj
            self.mock_command_handler = mock_command_handler
            self.mock_tg_message_handler = mock_tg_message_handler
            self.mock_filters = mock_filters
            self.mock_chat_action = mock_chat_action
            yield

    def _make_bot(self, **kwargs):
        from animus_forge.messaging.telegram_bot import TelegramBot

        defaults = dict(token="tg-test-token")
        defaults.update(kwargs)
        return TelegramBot(**defaults)

    def test_init_defaults(self):
        bot = self._make_bot()
        assert bot.token == "tg-test-token"
        assert bot.name == "Gorgon"
        assert bot.parse_mode == "Markdown"
        assert bot._app is None
        assert bot._command_handler is None

    def test_init_custom(self):
        bot = self._make_bot(
            name="Custom",
            allowed_users=["u1"],
            admin_users=["a1"],
            parse_mode="HTML",
        )
        assert bot.name == "Custom"
        assert bot.allowed_users == {"u1"}
        assert bot.admin_users == {"a1"}
        assert bot.parse_mode == "HTML"

    def test_platform_is_telegram(self):
        bot = self._make_bot()
        assert bot.platform == MessagePlatform.TELEGRAM

    def test_set_command_handler(self):
        bot = self._make_bot()
        ch = MagicMock()
        bot.set_command_handler(ch)
        assert bot._command_handler is ch

    # -- start / stop --

    def test_start(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            assert bot._running is True
            assert bot._app is not None
            self.mock_app_instance.initialize.assert_awaited_once()
            self.mock_app_instance.start.assert_awaited_once()
            self.mock_app_instance.updater.start_polling.assert_awaited_once()

        asyncio.run(_test())

    def test_start_already_running(self):
        async def _test():
            bot = self._make_bot()
            bot._running = True
            await bot.start()
            # Should not build a new app
            self.mock_application.builder.assert_not_called()

        asyncio.run(_test())

    def test_stop(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            await bot.stop()
            assert bot._running is False
            self.mock_app_instance.updater.stop.assert_awaited_once()
            self.mock_app_instance.stop.assert_awaited_once()
            self.mock_app_instance.shutdown.assert_awaited_once()

        asyncio.run(_test())

    def test_stop_not_running(self):
        async def _test():
            bot = self._make_bot()
            bot._running = False
            await bot.stop()
            # Should do nothing
            self.mock_app_instance.updater.stop.assert_not_awaited()

        asyncio.run(_test())

    # -- _setup_handlers --

    def test_setup_handlers_no_app(self):
        bot = self._make_bot()
        bot._app = None
        bot._setup_handlers()  # Should not raise

    def test_setup_handlers_registers_all(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            # 5 command handlers + 3 message handlers (TEXT, PHOTO, DOCUMENT) + 1 VOICE
            assert self.mock_app_instance.add_handler.call_count >= 8

        asyncio.run(_test())

    # -- _create_bot_user --

    def test_create_bot_user(self):
        bot = self._make_bot(admin_users=["42"])
        tg_user = MagicMock()
        tg_user.id = 42
        tg_user.username = "alice"
        tg_user.full_name = "Alice Smith"
        tg_user.language_code = "en"
        tg_user.is_bot = False
        user = bot._create_bot_user(tg_user)
        assert user.id == "42"
        assert user.platform == MessagePlatform.TELEGRAM
        assert user.username == "alice"
        assert user.display_name == "Alice Smith"
        assert user.is_admin is True
        assert user.metadata["language_code"] == "en"

    def test_create_bot_user_not_admin(self):
        bot = self._make_bot(admin_users=["999"])
        tg_user = MagicMock()
        tg_user.id = 1
        tg_user.username = "u"
        tg_user.full_name = "U"
        tg_user.language_code = None
        tg_user.is_bot = False
        user = bot._create_bot_user(tg_user)
        assert user.is_admin is False

    # -- _create_bot_message --

    def test_create_bot_message(self):
        bot = self._make_bot()
        update = MagicMock()
        msg = MagicMock()
        msg.message_id = 100
        msg.from_user = MagicMock()
        msg.from_user.id = 42
        msg.from_user.username = "alice"
        msg.from_user.full_name = "Alice"
        msg.from_user.language_code = "en"
        msg.from_user.is_bot = False
        msg.chat_id = 999
        msg.date = datetime(2026, 1, 1)
        msg.reply_to_message = None
        msg.chat = MagicMock()
        msg.chat.type = "private"
        msg.chat.title = None
        update.message = msg
        update.edited_message = None

        bot_msg = bot._create_bot_message(update, "Hello!")
        assert bot_msg.id == "100"
        assert bot_msg.platform == MessagePlatform.TELEGRAM
        assert bot_msg.content == "Hello!"
        assert bot_msg.chat_id == "999"
        assert bot_msg.metadata["chat_type"] == "private"

    def test_create_bot_message_with_reply(self):
        bot = self._make_bot()
        update = MagicMock()
        msg = MagicMock()
        msg.message_id = 100
        msg.from_user = MagicMock()
        msg.from_user.id = 1
        msg.from_user.username = "u"
        msg.from_user.full_name = "U"
        msg.from_user.language_code = None
        msg.from_user.is_bot = False
        msg.chat_id = 999
        msg.date = datetime(2026, 1, 1)
        msg.reply_to_message = MagicMock()
        msg.reply_to_message.message_id = 50
        msg.chat = MagicMock()
        msg.chat.type = "group"
        msg.chat.title = "Group"
        update.message = msg
        update.edited_message = None

        bot_msg = bot._create_bot_message(update, "text")
        assert bot_msg.reply_to_id == "50"
        assert bot_msg.metadata["chat_title"] == "Group"

    def test_create_bot_message_with_attachments(self):
        bot = self._make_bot()
        update = MagicMock()
        msg = MagicMock()
        msg.message_id = 1
        msg.from_user = MagicMock()
        msg.from_user.id = 1
        msg.from_user.username = "u"
        msg.from_user.full_name = "U"
        msg.from_user.language_code = None
        msg.from_user.is_bot = False
        msg.chat_id = 1
        msg.date = None
        msg.reply_to_message = None
        msg.chat = MagicMock()
        msg.chat.type = "private"
        msg.chat.title = None
        update.message = msg
        update.edited_message = None

        att = [{"type": "photo", "file_id": "abc"}]
        bot_msg = bot._create_bot_message(update, "[Photo]", attachments=att)
        assert len(bot_msg.attachments) == 1

    def test_create_bot_message_uses_edited_message(self):
        bot = self._make_bot()
        update = MagicMock()
        update.message = None
        msg = MagicMock()
        msg.message_id = 200
        msg.from_user = MagicMock()
        msg.from_user.id = 1
        msg.from_user.username = "u"
        msg.from_user.full_name = "U"
        msg.from_user.language_code = None
        msg.from_user.is_bot = False
        msg.chat_id = 5
        msg.date = datetime(2026, 1, 1)
        msg.reply_to_message = None
        msg.chat = MagicMock()
        msg.chat.type = "private"
        msg.chat.title = None
        update.edited_message = msg

        bot_msg = bot._create_bot_message(update, "edited text")
        assert bot_msg.id == "200"

    # -- _handle_message --

    def test_handle_message_text(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock(return_value="reply")
            bot.set_message_callback(cb)

            update = MagicMock()
            msg = MagicMock()
            msg.message_id = 1
            msg.text = "Hello"
            msg.from_user = MagicMock()
            msg.from_user.id = 1
            msg.from_user.username = "u"
            msg.from_user.full_name = "U"
            msg.from_user.language_code = None
            msg.from_user.is_bot = False
            msg.chat_id = 1
            msg.date = datetime(2026, 1, 1)
            msg.reply_to_message = None
            msg.chat = MagicMock()
            msg.chat.type = "private"
            msg.chat.title = None
            update.message = msg
            update.edited_message = None

            context = MagicMock()
            await bot._handle_message(update, context)
            cb.assert_awaited_once()

        asyncio.run(_test())

    def test_handle_message_no_message(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock()
            bot.set_message_callback(cb)
            update = MagicMock()
            update.message = None
            context = MagicMock()
            await bot._handle_message(update, context)
            cb.assert_not_awaited()

        asyncio.run(_test())

    def test_handle_message_no_text(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock()
            bot.set_message_callback(cb)
            update = MagicMock()
            update.message = MagicMock()
            update.message.text = None
            context = MagicMock()
            await bot._handle_message(update, context)
            cb.assert_not_awaited()

        asyncio.run(_test())

    # -- _handle_photo --

    def test_handle_photo(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock(return_value=None)
            bot.set_message_callback(cb)

            photo = MagicMock()
            photo.file_id = "fid"
            photo.file_unique_id = "fuid"
            photo.width = 800
            photo.height = 600
            photo.file_size = 5000
            file_obj = MagicMock()
            file_obj.file_path = "/tmp/photo.jpg"
            photo.get_file = AsyncMock(return_value=file_obj)

            update = MagicMock()
            msg = MagicMock()
            msg.message_id = 1
            msg.photo = [MagicMock(), photo]  # Last is largest
            msg.caption = "Nice photo"
            msg.from_user = MagicMock()
            msg.from_user.id = 1
            msg.from_user.username = "u"
            msg.from_user.full_name = "U"
            msg.from_user.language_code = None
            msg.from_user.is_bot = False
            msg.chat_id = 1
            msg.date = datetime(2026, 1, 1)
            msg.reply_to_message = None
            msg.chat = MagicMock()
            msg.chat.type = "private"
            msg.chat.title = None
            update.message = msg
            update.edited_message = None

            context = MagicMock()
            await bot._handle_photo(update, context)
            cb.assert_awaited_once()
            call_msg = cb.call_args[0][0]
            assert call_msg.content == "Nice photo"
            assert len(call_msg.attachments) == 1
            assert call_msg.attachments[0]["type"] == "photo"

        asyncio.run(_test())

    def test_handle_photo_no_caption(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock(return_value=None)
            bot.set_message_callback(cb)

            photo = MagicMock()
            photo.file_id = "fid"
            photo.file_unique_id = "fuid"
            photo.width = 800
            photo.height = 600
            photo.file_size = 5000
            photo.get_file = AsyncMock(return_value=MagicMock(file_path="/tmp/p.jpg"))

            update = MagicMock()
            msg = MagicMock()
            msg.message_id = 1
            msg.photo = [photo]
            msg.caption = None
            msg.from_user = MagicMock()
            msg.from_user.id = 1
            msg.from_user.username = "u"
            msg.from_user.full_name = "U"
            msg.from_user.language_code = None
            msg.from_user.is_bot = False
            msg.chat_id = 1
            msg.date = datetime(2026, 1, 1)
            msg.reply_to_message = None
            msg.chat = MagicMock()
            msg.chat.type = "private"
            msg.chat.title = None
            update.message = msg
            update.edited_message = None

            context = MagicMock()
            await bot._handle_photo(update, context)
            call_msg = cb.call_args[0][0]
            assert call_msg.content == "[Photo]"

        asyncio.run(_test())

    def test_handle_photo_no_message(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock()
            bot.set_message_callback(cb)
            update = MagicMock()
            update.message = None
            context = MagicMock()
            await bot._handle_photo(update, context)
            cb.assert_not_awaited()

        asyncio.run(_test())

    def test_handle_photo_no_photo(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock()
            bot.set_message_callback(cb)
            update = MagicMock()
            update.message = MagicMock()
            update.message.photo = None
            context = MagicMock()
            await bot._handle_photo(update, context)
            cb.assert_not_awaited()

        asyncio.run(_test())

    # -- _handle_document --

    def test_handle_document(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock(return_value=None)
            bot.set_message_callback(cb)

            doc = MagicMock()
            doc.file_id = "fid"
            doc.file_unique_id = "fuid"
            doc.file_name = "report.pdf"
            doc.mime_type = "application/pdf"
            doc.file_size = 10000
            doc.get_file = AsyncMock(return_value=MagicMock(file_path="/tmp/report.pdf"))

            update = MagicMock()
            msg = MagicMock()
            msg.message_id = 1
            msg.document = doc
            msg.caption = "Here is the report"
            msg.from_user = MagicMock()
            msg.from_user.id = 1
            msg.from_user.username = "u"
            msg.from_user.full_name = "U"
            msg.from_user.language_code = None
            msg.from_user.is_bot = False
            msg.chat_id = 1
            msg.date = datetime(2026, 1, 1)
            msg.reply_to_message = None
            msg.chat = MagicMock()
            msg.chat.type = "private"
            msg.chat.title = None
            update.message = msg
            update.edited_message = None

            context = MagicMock()
            await bot._handle_document(update, context)
            cb.assert_awaited_once()
            call_msg = cb.call_args[0][0]
            assert call_msg.content == "Here is the report"
            assert call_msg.attachments[0]["type"] == "document"
            assert call_msg.attachments[0]["file_name"] == "report.pdf"

        asyncio.run(_test())

    def test_handle_document_no_caption(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock(return_value=None)
            bot.set_message_callback(cb)

            doc = MagicMock()
            doc.file_id = "fid"
            doc.file_unique_id = "fuid"
            doc.file_name = "data.csv"
            doc.mime_type = "text/csv"
            doc.file_size = 500
            doc.get_file = AsyncMock(return_value=MagicMock(file_path="/tmp/d.csv"))

            update = MagicMock()
            msg = MagicMock()
            msg.message_id = 1
            msg.document = doc
            msg.caption = None
            msg.from_user = MagicMock()
            msg.from_user.id = 1
            msg.from_user.username = "u"
            msg.from_user.full_name = "U"
            msg.from_user.language_code = None
            msg.from_user.is_bot = False
            msg.chat_id = 1
            msg.date = None
            msg.reply_to_message = None
            msg.chat = MagicMock()
            msg.chat.type = "private"
            msg.chat.title = None
            update.message = msg
            update.edited_message = None

            context = MagicMock()
            await bot._handle_document(update, context)
            call_msg = cb.call_args[0][0]
            assert "[Document: data.csv]" in call_msg.content

        asyncio.run(_test())

    def test_handle_document_no_message(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock()
            bot.set_message_callback(cb)
            update = MagicMock()
            update.message = None
            context = MagicMock()
            await bot._handle_document(update, context)
            cb.assert_not_awaited()

        asyncio.run(_test())

    def test_handle_document_no_document(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock()
            bot.set_message_callback(cb)
            update = MagicMock()
            update.message = MagicMock()
            update.message.document = None
            context = MagicMock()
            await bot._handle_document(update, context)
            cb.assert_not_awaited()

        asyncio.run(_test())

    # -- _handle_voice --

    def test_handle_voice(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock(return_value=None)
            bot.set_message_callback(cb)

            voice = MagicMock()
            voice.file_id = "fid"
            voice.file_unique_id = "fuid"
            voice.duration = 10
            voice.mime_type = "audio/ogg"
            voice.file_size = 15000
            voice.get_file = AsyncMock(return_value=MagicMock(file_path="/tmp/voice.ogg"))

            update = MagicMock()
            msg = MagicMock()
            msg.message_id = 1
            msg.voice = voice
            msg.from_user = MagicMock()
            msg.from_user.id = 1
            msg.from_user.username = "u"
            msg.from_user.full_name = "U"
            msg.from_user.language_code = None
            msg.from_user.is_bot = False
            msg.chat_id = 1
            msg.date = datetime(2026, 1, 1)
            msg.reply_to_message = None
            msg.chat = MagicMock()
            msg.chat.type = "private"
            msg.chat.title = None
            update.message = msg
            update.edited_message = None

            context = MagicMock()
            await bot._handle_voice(update, context)
            cb.assert_awaited_once()
            call_msg = cb.call_args[0][0]
            assert call_msg.content == "[Voice message]"
            assert call_msg.attachments[0]["type"] == "voice"
            assert call_msg.attachments[0]["duration"] == 10

        asyncio.run(_test())

    def test_handle_voice_no_message(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock()
            bot.set_message_callback(cb)
            update = MagicMock()
            update.message = None
            context = MagicMock()
            await bot._handle_voice(update, context)
            cb.assert_not_awaited()

        asyncio.run(_test())

    def test_handle_voice_no_voice(self):
        async def _test():
            bot = self._make_bot()
            cb = AsyncMock()
            bot.set_message_callback(cb)
            update = MagicMock()
            update.message = MagicMock()
            update.message.voice = None
            context = MagicMock()
            await bot._handle_voice(update, context)
            cb.assert_not_awaited()

        asyncio.run(_test())

    # -- _handle_command --

    def test_handle_command_with_handler(self):
        async def _test():
            bot = self._make_bot()
            cmd_handler = AsyncMock()
            cmd_handler.handle_command = AsyncMock(return_value="response")
            bot.set_command_handler(cmd_handler)

            update = MagicMock()
            msg = MagicMock()
            msg.message_id = 1
            msg.text = "/start"
            msg.from_user = MagicMock()
            msg.from_user.id = 1
            msg.from_user.username = "u"
            msg.from_user.full_name = "U"
            msg.from_user.language_code = None
            msg.from_user.is_bot = False
            msg.chat_id = 999
            msg.date = datetime(2026, 1, 1)
            msg.reply_to_message = None
            msg.chat = MagicMock()
            msg.chat.type = "private"
            msg.chat.title = None
            update.message = msg
            update.edited_message = None

            context = MagicMock()
            context.args = ["arg1"]
            await bot._handle_command(update, context, "start")
            cmd_handler.handle_command.assert_awaited_once()

        asyncio.run(_test())

    def test_handle_command_no_message(self):
        async def _test():
            bot = self._make_bot()
            cmd_handler = AsyncMock()
            bot.set_command_handler(cmd_handler)
            update = MagicMock()
            update.message = None
            context = MagicMock()
            await bot._handle_command(update, context, "start")
            # Should return early
            cmd_handler.handle_command.assert_not_awaited()

        asyncio.run(_test())

    def test_handle_command_no_handler_falls_back(self):
        async def _test():
            bot = self._make_bot()
            bot._command_handler = None
            cb = AsyncMock(return_value=None)
            bot.set_message_callback(cb)

            update = MagicMock()
            msg = MagicMock()
            msg.message_id = 1
            msg.text = "/start"
            msg.from_user = MagicMock()
            msg.from_user.id = 1
            msg.from_user.username = "u"
            msg.from_user.full_name = "U"
            msg.from_user.language_code = None
            msg.from_user.is_bot = False
            msg.chat_id = 999
            msg.date = datetime(2026, 1, 1)
            msg.reply_to_message = None
            msg.chat = MagicMock()
            msg.chat.type = "private"
            msg.chat.title = None
            update.message = msg
            update.edited_message = None

            context = MagicMock()
            context.args = []
            await bot._handle_command(update, context, "start")
            # Falls back to handle_message
            cb.assert_awaited_once()

        asyncio.run(_test())

    def test_handle_command_with_none_args(self):
        async def _test():
            bot = self._make_bot()
            cmd_handler = AsyncMock()
            cmd_handler.handle_command = AsyncMock(return_value="ok")
            bot.set_command_handler(cmd_handler)

            update = MagicMock()
            msg = MagicMock()
            msg.message_id = 1
            msg.text = "/help"
            msg.from_user = MagicMock()
            msg.from_user.id = 1
            msg.from_user.username = "u"
            msg.from_user.full_name = "U"
            msg.from_user.language_code = None
            msg.from_user.is_bot = False
            msg.chat_id = 1
            msg.date = datetime(2026, 1, 1)
            msg.reply_to_message = None
            msg.chat = MagicMock()
            msg.chat.type = "private"
            msg.chat.title = None
            update.message = msg
            update.edited_message = None

            context = MagicMock()
            context.args = None
            await bot._handle_command(update, context, "help")
            _, kwargs = cmd_handler.handle_command.call_args
            # args should default to []
            assert kwargs.get("args", cmd_handler.handle_command.call_args[0][2]) == []

        asyncio.run(_test())

    # -- Command shortcut methods --

    def test_handle_start(self):
        async def _test():
            bot = self._make_bot()
            bot._handle_command = AsyncMock()
            update = MagicMock()
            context = MagicMock()
            await bot._handle_start(update, context)
            bot._handle_command.assert_awaited_once_with(update, context, "start")

        asyncio.run(_test())

    def test_handle_help(self):
        async def _test():
            bot = self._make_bot()
            bot._handle_command = AsyncMock()
            update = MagicMock()
            context = MagicMock()
            await bot._handle_help(update, context)
            bot._handle_command.assert_awaited_once_with(update, context, "help")

        asyncio.run(_test())

    def test_handle_new(self):
        async def _test():
            bot = self._make_bot()
            bot._handle_command = AsyncMock()
            update = MagicMock()
            context = MagicMock()
            await bot._handle_new(update, context)
            bot._handle_command.assert_awaited_once_with(update, context, "new")

        asyncio.run(_test())

    def test_handle_status(self):
        async def _test():
            bot = self._make_bot()
            bot._handle_command = AsyncMock()
            update = MagicMock()
            context = MagicMock()
            await bot._handle_status(update, context)
            bot._handle_command.assert_awaited_once_with(update, context, "status")

        asyncio.run(_test())

    def test_handle_history(self):
        async def _test():
            bot = self._make_bot()
            bot._handle_command = AsyncMock()
            update = MagicMock()
            context = MagicMock()
            await bot._handle_history(update, context)
            bot._handle_command.assert_awaited_once_with(update, context, "history")

        asyncio.run(_test())

    # -- send_message --

    def test_send_message_success(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            sent_msg = MagicMock()
            sent_msg.message_id = 777
            self.mock_bot_obj.send_message = AsyncMock(return_value=sent_msg)
            result = await bot.send_message("123", "hello")
            assert result == "777"

        asyncio.run(_test())

    def test_send_message_with_reply(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            sent_msg = MagicMock()
            sent_msg.message_id = 1
            self.mock_bot_obj.send_message = AsyncMock(return_value=sent_msg)
            await bot.send_message("123", "hello", reply_to_id="50")
            call_kwargs = self.mock_bot_obj.send_message.call_args[1]
            assert call_kwargs["reply_to_message_id"] == 50

        asyncio.run(_test())

    def test_send_message_long_splits(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            sent_msg = MagicMock()
            sent_msg.message_id = 1
            self.mock_bot_obj.send_message = AsyncMock(return_value=sent_msg)
            long = "x" * 10000  # > 4096
            await bot.send_message("123", long)
            assert self.mock_bot_obj.send_message.await_count == 3  # ceil(10000/4096) = 3

        asyncio.run(_test())

    def test_send_message_no_app(self):
        async def _test():
            bot = self._make_bot()
            bot._app = None
            result = await bot.send_message("123", "hello")
            assert result is None

        asyncio.run(_test())

    def test_send_message_exception(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            self.mock_bot_obj.send_message = AsyncMock(side_effect=Exception("fail"))
            result = await bot.send_message("123", "hello")
            assert result is None

        asyncio.run(_test())

    def test_send_message_disable_parse_mode(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            sent_msg = MagicMock()
            sent_msg.message_id = 1
            self.mock_bot_obj.send_message = AsyncMock(return_value=sent_msg)
            await bot.send_message("123", "hello", parse_mode=False)
            call_kwargs = self.mock_bot_obj.send_message.call_args[1]
            assert call_kwargs["parse_mode"] is None

        asyncio.run(_test())

    # -- send_typing --

    def test_send_typing(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            self.mock_bot_obj.send_chat_action = AsyncMock()
            await bot.send_typing("123")
            self.mock_bot_obj.send_chat_action.assert_awaited_once()

        asyncio.run(_test())

    def test_send_typing_no_app(self):
        async def _test():
            bot = self._make_bot()
            bot._app = None
            await bot.send_typing("123")  # Should not raise

        asyncio.run(_test())

    def test_send_typing_exception(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            self.mock_bot_obj.send_chat_action = AsyncMock(side_effect=Exception("fail"))
            await bot.send_typing("123")  # Should not raise

        asyncio.run(_test())

    # -- send_photo --

    def test_send_photo_success(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            sent_msg = MagicMock()
            sent_msg.message_id = 555
            self.mock_bot_obj.send_photo = AsyncMock(return_value=sent_msg)
            result = await bot.send_photo("123", b"imagebytes", caption="Photo")
            assert result == "555"

        asyncio.run(_test())

    def test_send_photo_with_reply(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            sent_msg = MagicMock()
            sent_msg.message_id = 1
            self.mock_bot_obj.send_photo = AsyncMock(return_value=sent_msg)
            await bot.send_photo("123", "http://example.com/photo.jpg", reply_to_id="50")
            call_kwargs = self.mock_bot_obj.send_photo.call_args[1]
            assert call_kwargs["reply_to_message_id"] == 50

        asyncio.run(_test())

    def test_send_photo_no_app(self):
        async def _test():
            bot = self._make_bot()
            bot._app = None
            result = await bot.send_photo("123", b"img")
            assert result is None

        asyncio.run(_test())

    def test_send_photo_exception(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            self.mock_bot_obj.send_photo = AsyncMock(side_effect=Exception("fail"))
            result = await bot.send_photo("123", b"img")
            assert result is None

        asyncio.run(_test())

    # -- send_document --

    def test_send_document_success(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            sent_msg = MagicMock()
            sent_msg.message_id = 333
            self.mock_bot_obj.send_document = AsyncMock(return_value=sent_msg)
            result = await bot.send_document(
                "123", b"docbytes", filename="doc.pdf", caption="A doc"
            )
            assert result == "333"

        asyncio.run(_test())

    def test_send_document_with_reply(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            sent_msg = MagicMock()
            sent_msg.message_id = 1
            self.mock_bot_obj.send_document = AsyncMock(return_value=sent_msg)
            await bot.send_document("123", b"d", reply_to_id="50")
            call_kwargs = self.mock_bot_obj.send_document.call_args[1]
            assert call_kwargs["reply_to_message_id"] == 50

        asyncio.run(_test())

    def test_send_document_no_app(self):
        async def _test():
            bot = self._make_bot()
            bot._app = None
            result = await bot.send_document("123", b"d")
            assert result is None

        asyncio.run(_test())

    def test_send_document_exception(self):
        async def _test():
            bot = self._make_bot()
            await bot.start()
            self.mock_bot_obj.send_document = AsyncMock(side_effect=Exception("fail"))
            result = await bot.send_document("123", b"d")
            assert result is None

        asyncio.run(_test())


class TestCreateTelegramBot:
    """Tests for create_telegram_bot factory function."""

    @pytest.fixture(autouse=True)
    def mock_telegram_available(self):
        with (
            patch("animus_forge.messaging.telegram_bot.TELEGRAM_AVAILABLE", True),
            patch("animus_forge.messaging.telegram_bot.Application", MagicMock()),
            patch("animus_forge.messaging.telegram_bot.CommandHandler", MagicMock()),
            patch("animus_forge.messaging.telegram_bot.TGMessageHandler", MagicMock()),
            patch("animus_forge.messaging.telegram_bot.filters", MagicMock()),
            patch("animus_forge.messaging.telegram_bot.ChatAction", MagicMock()),
            patch("animus_forge.messaging.telegram_bot.Update", MagicMock()),
        ):
            yield

    def test_create_with_token(self):
        from animus_forge.messaging.telegram_bot import create_telegram_bot

        bot = create_telegram_bot(token="tg-token")
        assert bot.token == "tg-token"

    def test_create_from_env(self):
        from animus_forge.messaging.telegram_bot import create_telegram_bot

        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = "env-tg-token"
        mock_settings.telegram_allowed_users = None
        mock_settings.telegram_admin_users = None
        with patch("animus_forge.config.settings.get_settings", return_value=mock_settings):
            bot = create_telegram_bot()
        assert bot.token == "env-tg-token"

    def test_create_no_token_raises(self):
        from animus_forge.messaging.telegram_bot import create_telegram_bot

        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = None
        with patch("animus_forge.config.settings.get_settings", return_value=mock_settings):
            with pytest.raises(ValueError, match="bot token not provided"):
                create_telegram_bot()

    def test_create_with_users(self):
        from animus_forge.messaging.telegram_bot import create_telegram_bot

        bot = create_telegram_bot(
            token="t",
            allowed_users=["u1", "u2"],
            admin_users=["a1"],
        )
        assert bot.allowed_users == {"u1", "u2"}
        assert bot.admin_users == {"a1"}

    def test_create_loads_users_from_env(self):
        from animus_forge.messaging.telegram_bot import create_telegram_bot

        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = "t"
        mock_settings.telegram_allowed_users = "u1, u2"
        mock_settings.telegram_admin_users = "a1"
        with patch("animus_forge.config.settings.get_settings", return_value=mock_settings):
            bot = create_telegram_bot()
        assert bot.allowed_users == {"u1", "u2"}
        assert bot.admin_users == {"a1"}

    def test_create_empty_env_vars(self):
        from animus_forge.messaging.telegram_bot import create_telegram_bot

        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = "t"
        mock_settings.telegram_allowed_users = ""
        mock_settings.telegram_admin_users = ""
        with patch("animus_forge.config.settings.get_settings", return_value=mock_settings):
            bot = create_telegram_bot()
        assert bot.allowed_users is None
        assert bot.admin_users == set()
