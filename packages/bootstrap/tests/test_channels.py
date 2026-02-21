"""Tests for all gateway channel adapters.

Since platform libraries are optional, every test mocks the HAS_* flag
and/or the library classes so we can exercise the adapter logic without
installing telegram, discord.py, slack-bolt, matrix-nio, etc.
"""

from __future__ import annotations

import imaplib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_bootstrap.gateway.models import ChannelHealth, GatewayResponse

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


class TestTelegramAdapter:
    """TelegramAdapter tests."""

    def test_import_error_when_library_missing(self) -> None:
        with patch("animus_bootstrap.gateway.channels.telegram.HAS_TELEGRAM", False):
            from animus_bootstrap.gateway.channels.telegram import TelegramAdapter

            with pytest.raises(ImportError, match="python-telegram-bot"):
                TelegramAdapter(bot_token="fake-token")

    def test_constructor_with_library(self) -> None:
        with patch("animus_bootstrap.gateway.channels.telegram.HAS_TELEGRAM", True):
            from animus_bootstrap.gateway.channels.telegram import TelegramAdapter

            adapter = TelegramAdapter(bot_token="fake-token")
            assert adapter.name == "telegram"
            assert adapter.is_connected is False
            assert adapter._token == "fake-token"

    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        with patch("animus_bootstrap.gateway.channels.telegram.HAS_TELEGRAM", True):
            from animus_bootstrap.gateway.channels.telegram import TelegramAdapter

            adapter = TelegramAdapter(bot_token="fake-token")

            # Mock the Application builder chain
            mock_app = MagicMock()
            mock_app.initialize = AsyncMock()
            mock_app.start = AsyncMock()
            mock_app.stop = AsyncMock()
            mock_app.shutdown = AsyncMock()
            mock_app.add_handler = MagicMock()
            mock_updater = MagicMock()
            mock_updater.start_polling = AsyncMock()
            mock_updater.running = True
            mock_updater.stop = AsyncMock()
            mock_app.updater = mock_updater

            mock_builder = MagicMock()
            mock_builder.token.return_value = mock_builder
            mock_builder.build.return_value = mock_app

            with (
                patch(
                    "animus_bootstrap.gateway.channels.telegram.Application",
                    create=True,
                ) as MockApplication,
                patch(
                    "animus_bootstrap.gateway.channels.telegram.MessageHandler",
                    create=True,
                ),
                patch(
                    "animus_bootstrap.gateway.channels.telegram.filters",
                    create=True,
                ),
            ):
                MockApplication.builder.return_value = mock_builder

                await adapter.connect()
                assert adapter.is_connected is True

                await adapter.disconnect()
                assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_send_message(self) -> None:
        with patch("animus_bootstrap.gateway.channels.telegram.HAS_TELEGRAM", True):
            from animus_bootstrap.gateway.channels.telegram import TelegramAdapter

            adapter = TelegramAdapter(bot_token="fake-token")

            mock_bot = AsyncMock()
            mock_sent = MagicMock()
            mock_sent.message_id = 12345
            mock_bot.send_message = AsyncMock(return_value=mock_sent)

            mock_app = MagicMock()
            mock_app.bot = mock_bot
            adapter._app = mock_app

            resp = GatewayResponse(
                text="Hello!",
                channel="telegram",
                metadata={"chat_id": "999"},
            )

            msg_id = await adapter.send_message(resp)
            assert msg_id == "12345"
            mock_bot.send_message.assert_awaited_once_with(chat_id="999", text="Hello!")

    @pytest.mark.asyncio
    async def test_send_message_missing_chat_id(self) -> None:
        with patch("animus_bootstrap.gateway.channels.telegram.HAS_TELEGRAM", True):
            from animus_bootstrap.gateway.channels.telegram import TelegramAdapter

            adapter = TelegramAdapter(bot_token="fake-token")
            mock_app = MagicMock()
            mock_app.bot = MagicMock()
            adapter._app = mock_app

            resp = GatewayResponse(text="Hello!", channel="telegram", metadata={})
            with pytest.raises(ValueError, match="chat_id"):
                await adapter.send_message(resp)

    @pytest.mark.asyncio
    async def test_send_message_not_connected(self) -> None:
        with patch("animus_bootstrap.gateway.channels.telegram.HAS_TELEGRAM", True):
            from animus_bootstrap.gateway.channels.telegram import TelegramAdapter

            adapter = TelegramAdapter(bot_token="fake-token")

            resp = GatewayResponse(text="Hello!", channel="telegram", metadata={"chat_id": "1"})
            with pytest.raises(RuntimeError, match="not connected"):
                await adapter.send_message(resp)

    @pytest.mark.asyncio
    async def test_on_message_stores_callback(self) -> None:
        with patch("animus_bootstrap.gateway.channels.telegram.HAS_TELEGRAM", True):
            from animus_bootstrap.gateway.channels.telegram import TelegramAdapter

            adapter = TelegramAdapter(bot_token="fake-token")
            cb = AsyncMock()
            await adapter.on_message(cb)
            assert adapter._callback is cb

    @pytest.mark.asyncio
    async def test_health_check_connected(self) -> None:
        with patch("animus_bootstrap.gateway.channels.telegram.HAS_TELEGRAM", True):
            from animus_bootstrap.gateway.channels.telegram import TelegramAdapter

            adapter = TelegramAdapter(bot_token="fake-token")

            mock_bot = AsyncMock()
            mock_bot.get_me = AsyncMock(return_value=MagicMock())
            mock_app = MagicMock()
            mock_app.bot = mock_bot
            adapter._app = mock_app

            health = await adapter.health_check()
            assert isinstance(health, ChannelHealth)
            assert health.channel == "telegram"
            assert health.connected is True

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self) -> None:
        with patch("animus_bootstrap.gateway.channels.telegram.HAS_TELEGRAM", True):
            from animus_bootstrap.gateway.channels.telegram import TelegramAdapter

            adapter = TelegramAdapter(bot_token="fake-token")
            health = await adapter.health_check()
            assert health.connected is False

    @pytest.mark.asyncio
    async def test_health_check_error(self) -> None:
        with patch("animus_bootstrap.gateway.channels.telegram.HAS_TELEGRAM", True):
            from animus_bootstrap.gateway.channels.telegram import TelegramAdapter

            adapter = TelegramAdapter(bot_token="fake-token")

            mock_bot = AsyncMock()
            mock_bot.get_me = AsyncMock(side_effect=RuntimeError("bad token"))
            mock_app = MagicMock()
            mock_app.bot = mock_bot
            adapter._app = mock_app

            health = await adapter.health_check()
            assert health.connected is False
            assert "bad token" in health.error


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------


class TestDiscordAdapter:
    """DiscordAdapter tests."""

    def test_import_error_when_library_missing(self) -> None:
        with patch(
            "animus_bootstrap.gateway.channels.discord_channel.HAS_DISCORD",
            False,
        ):
            from animus_bootstrap.gateway.channels.discord_channel import DiscordAdapter

            with pytest.raises(ImportError, match="discord.py"):
                DiscordAdapter(bot_token="fake-token")

    def test_constructor_with_library(self) -> None:
        with patch(
            "animus_bootstrap.gateway.channels.discord_channel.HAS_DISCORD",
            True,
        ):
            from animus_bootstrap.gateway.channels.discord_channel import DiscordAdapter

            adapter = DiscordAdapter(bot_token="fake-token", allowed_guilds=["123"])
            assert adapter.name == "discord"
            assert adapter.is_connected is False
            assert adapter._allowed_guilds == {"123"}

    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        with patch(
            "animus_bootstrap.gateway.channels.discord_channel.HAS_DISCORD",
            True,
        ):
            from animus_bootstrap.gateway.channels.discord_channel import DiscordAdapter

            adapter = DiscordAdapter(bot_token="fake-token")

            mock_client = MagicMock()
            mock_client.start = AsyncMock()
            mock_client.close = AsyncMock()
            mock_client.event = MagicMock(side_effect=lambda fn: fn)  # decorator passthrough

            with patch(
                "animus_bootstrap.gateway.channels.discord_channel.discord",
                create=True,
            ) as mock_discord:
                mock_discord.Client.return_value = mock_client
                mock_discord.Intents.default.return_value = MagicMock()

                await adapter.connect()
                assert adapter.is_connected is True

                await adapter.disconnect()
                assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_send_message(self) -> None:
        with patch(
            "animus_bootstrap.gateway.channels.discord_channel.HAS_DISCORD",
            True,
        ):
            from animus_bootstrap.gateway.channels.discord_channel import DiscordAdapter

            adapter = DiscordAdapter(bot_token="fake-token")

            mock_channel = AsyncMock()
            mock_sent = MagicMock()
            mock_sent.id = 67890
            mock_channel.send = AsyncMock(return_value=mock_sent)

            mock_client = MagicMock()
            mock_client.get_channel.return_value = mock_channel
            adapter._client = mock_client

            resp = GatewayResponse(
                text="Hello Discord!",
                channel="discord",
                metadata={"channel_id": "111"},
            )

            msg_id = await adapter.send_message(resp)
            assert msg_id == "67890"

    @pytest.mark.asyncio
    async def test_send_message_missing_channel_id(self) -> None:
        with patch(
            "animus_bootstrap.gateway.channels.discord_channel.HAS_DISCORD",
            True,
        ):
            from animus_bootstrap.gateway.channels.discord_channel import DiscordAdapter

            adapter = DiscordAdapter(bot_token="fake-token")
            adapter._client = MagicMock()

            resp = GatewayResponse(text="Hello!", channel="discord", metadata={})
            with pytest.raises(ValueError, match="channel_id"):
                await adapter.send_message(resp)

    @pytest.mark.asyncio
    async def test_send_message_not_connected(self) -> None:
        with patch(
            "animus_bootstrap.gateway.channels.discord_channel.HAS_DISCORD",
            True,
        ):
            from animus_bootstrap.gateway.channels.discord_channel import DiscordAdapter

            adapter = DiscordAdapter(bot_token="fake-token")

            resp = GatewayResponse(text="Hello!", channel="discord", metadata={"channel_id": "1"})
            with pytest.raises(RuntimeError, match="not connected"):
                await adapter.send_message(resp)

    @pytest.mark.asyncio
    async def test_on_message_stores_callback(self) -> None:
        with patch(
            "animus_bootstrap.gateway.channels.discord_channel.HAS_DISCORD",
            True,
        ):
            from animus_bootstrap.gateway.channels.discord_channel import DiscordAdapter

            adapter = DiscordAdapter(bot_token="fake-token")
            cb = AsyncMock()
            await adapter.on_message(cb)
            assert adapter._callback is cb

    @pytest.mark.asyncio
    async def test_health_check_connected(self) -> None:
        with patch(
            "animus_bootstrap.gateway.channels.discord_channel.HAS_DISCORD",
            True,
        ):
            from animus_bootstrap.gateway.channels.discord_channel import DiscordAdapter

            adapter = DiscordAdapter(bot_token="fake-token")
            mock_client = MagicMock()
            mock_client.is_ready.return_value = True
            adapter._client = mock_client

            health = await adapter.health_check()
            assert health.connected is True

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self) -> None:
        with patch(
            "animus_bootstrap.gateway.channels.discord_channel.HAS_DISCORD",
            True,
        ):
            from animus_bootstrap.gateway.channels.discord_channel import DiscordAdapter

            adapter = DiscordAdapter(bot_token="fake-token")
            health = await adapter.health_check()
            assert health.connected is False


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------


class TestSlackAdapter:
    """SlackAdapter tests."""

    def test_import_error_when_library_missing(self) -> None:
        with patch("animus_bootstrap.gateway.channels.slack.HAS_SLACK", False):
            from animus_bootstrap.gateway.channels.slack import SlackAdapter

            with pytest.raises(ImportError, match="slack-bolt"):
                SlackAdapter(bot_token="fake-token")

    def test_constructor_with_library(self) -> None:
        with patch("animus_bootstrap.gateway.channels.slack.HAS_SLACK", True):
            from animus_bootstrap.gateway.channels.slack import SlackAdapter

            adapter = SlackAdapter(bot_token="xoxb-fake", app_token="xapp-fake")
            assert adapter.name == "slack"
            assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_disconnect_no_socket_mode(self) -> None:
        with patch("animus_bootstrap.gateway.channels.slack.HAS_SLACK", True):
            from animus_bootstrap.gateway.channels.slack import SlackAdapter

            adapter = SlackAdapter(bot_token="xoxb-fake", app_token="")

            mock_app = MagicMock()
            mock_app.message = MagicMock(return_value=lambda fn: fn)

            with patch(
                "animus_bootstrap.gateway.channels.slack.AsyncApp",
                create=True,
                return_value=mock_app,
            ):
                await adapter.connect()
                assert adapter.is_connected is True

                await adapter.disconnect()
                assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_with_socket_mode(self) -> None:
        with patch("animus_bootstrap.gateway.channels.slack.HAS_SLACK", True):
            from animus_bootstrap.gateway.channels.slack import SlackAdapter

            adapter = SlackAdapter(bot_token="xoxb-fake", app_token="xapp-fake")

            mock_app = MagicMock()
            mock_app.message = MagicMock(return_value=lambda fn: fn)

            mock_handler = AsyncMock()
            mock_handler.start_async = AsyncMock()
            mock_handler.close_async = AsyncMock()

            with (
                patch(
                    "animus_bootstrap.gateway.channels.slack.AsyncApp",
                    create=True,
                    return_value=mock_app,
                ),
                patch(
                    "animus_bootstrap.gateway.channels.slack.AsyncSocketModeHandler",
                    create=True,
                    return_value=mock_handler,
                ),
            ):
                await adapter.connect()
                assert adapter.is_connected is True
                mock_handler.start_async.assert_awaited_once()

                await adapter.disconnect()
                assert adapter.is_connected is False
                mock_handler.close_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_message(self) -> None:
        with patch("animus_bootstrap.gateway.channels.slack.HAS_SLACK", True):
            from animus_bootstrap.gateway.channels.slack import SlackAdapter

            adapter = SlackAdapter(bot_token="xoxb-fake")

            mock_client = AsyncMock()
            mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1234.5678"})
            mock_app = MagicMock()
            mock_app.client = mock_client
            adapter._app = mock_app

            resp = GatewayResponse(
                text="Hello Slack!",
                channel="slack",
                metadata={"channel_id": "C123"},
            )

            msg_id = await adapter.send_message(resp)
            assert msg_id == "1234.5678"

    @pytest.mark.asyncio
    async def test_send_message_with_thread(self) -> None:
        with patch("animus_bootstrap.gateway.channels.slack.HAS_SLACK", True):
            from animus_bootstrap.gateway.channels.slack import SlackAdapter

            adapter = SlackAdapter(bot_token="xoxb-fake")

            mock_client = AsyncMock()
            mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1234.5678"})
            mock_app = MagicMock()
            mock_app.client = mock_client
            adapter._app = mock_app

            resp = GatewayResponse(
                text="Reply!",
                channel="slack",
                metadata={"channel_id": "C123", "thread_ts": "1111.0000"},
            )

            await adapter.send_message(resp)
            mock_client.chat_postMessage.assert_awaited_once_with(
                channel="C123",
                text="Reply!",
                thread_ts="1111.0000",
            )

    @pytest.mark.asyncio
    async def test_send_message_missing_channel_id(self) -> None:
        with patch("animus_bootstrap.gateway.channels.slack.HAS_SLACK", True):
            from animus_bootstrap.gateway.channels.slack import SlackAdapter

            adapter = SlackAdapter(bot_token="xoxb-fake")
            adapter._app = MagicMock()

            resp = GatewayResponse(text="Hello!", channel="slack", metadata={})
            with pytest.raises(ValueError, match="channel_id"):
                await adapter.send_message(resp)

    @pytest.mark.asyncio
    async def test_send_message_not_connected(self) -> None:
        with patch("animus_bootstrap.gateway.channels.slack.HAS_SLACK", True):
            from animus_bootstrap.gateway.channels.slack import SlackAdapter

            adapter = SlackAdapter(bot_token="xoxb-fake")

            resp = GatewayResponse(text="Hello!", channel="slack", metadata={"channel_id": "C1"})
            with pytest.raises(RuntimeError, match="not connected"):
                await adapter.send_message(resp)

    @pytest.mark.asyncio
    async def test_on_message_stores_callback(self) -> None:
        with patch("animus_bootstrap.gateway.channels.slack.HAS_SLACK", True):
            from animus_bootstrap.gateway.channels.slack import SlackAdapter

            adapter = SlackAdapter(bot_token="xoxb-fake")
            cb = AsyncMock()
            await adapter.on_message(cb)
            assert adapter._callback is cb

    @pytest.mark.asyncio
    async def test_health_check_ok(self) -> None:
        with patch("animus_bootstrap.gateway.channels.slack.HAS_SLACK", True):
            from animus_bootstrap.gateway.channels.slack import SlackAdapter

            adapter = SlackAdapter(bot_token="xoxb-fake")

            mock_client = AsyncMock()
            mock_client.auth_test = AsyncMock(return_value={"ok": True})
            mock_app = MagicMock()
            mock_app.client = mock_client
            adapter._app = mock_app

            health = await adapter.health_check()
            assert health.connected is True

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self) -> None:
        with patch("animus_bootstrap.gateway.channels.slack.HAS_SLACK", True):
            from animus_bootstrap.gateway.channels.slack import SlackAdapter

            adapter = SlackAdapter(bot_token="xoxb-fake")
            health = await adapter.health_check()
            assert health.connected is False

    @pytest.mark.asyncio
    async def test_health_check_error(self) -> None:
        with patch("animus_bootstrap.gateway.channels.slack.HAS_SLACK", True):
            from animus_bootstrap.gateway.channels.slack import SlackAdapter

            adapter = SlackAdapter(bot_token="xoxb-fake")

            mock_client = AsyncMock()
            mock_client.auth_test = AsyncMock(side_effect=RuntimeError("auth failed"))
            mock_app = MagicMock()
            mock_app.client = mock_client
            adapter._app = mock_app

            health = await adapter.health_check()
            assert health.connected is False
            assert "auth failed" in health.error


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


class TestMatrixAdapter:
    """MatrixAdapter tests."""

    def test_import_error_when_library_missing(self) -> None:
        with patch("animus_bootstrap.gateway.channels.matrix.HAS_MATRIX", False):
            from animus_bootstrap.gateway.channels.matrix import MatrixAdapter

            with pytest.raises(ImportError, match="matrix-nio"):
                MatrixAdapter(homeserver="https://matrix.org", access_token="tok")

    def test_constructor_with_library(self) -> None:
        with patch("animus_bootstrap.gateway.channels.matrix.HAS_MATRIX", True):
            from animus_bootstrap.gateway.channels.matrix import MatrixAdapter

            adapter = MatrixAdapter(
                homeserver="https://matrix.org",
                access_token="tok",
                room_ids=["!room:matrix.org"],
            )
            assert adapter.name == "matrix"
            assert adapter.is_connected is False
            assert adapter._room_ids == {"!room:matrix.org"}

    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        with patch("animus_bootstrap.gateway.channels.matrix.HAS_MATRIX", True):
            from animus_bootstrap.gateway.channels.matrix import MatrixAdapter

            adapter = MatrixAdapter(homeserver="https://matrix.org", access_token="tok")

            mock_client = MagicMock()
            mock_client.add_event_callback = MagicMock()
            mock_client.sync_forever = AsyncMock()
            mock_client.close = AsyncMock()

            with (
                patch(
                    "animus_bootstrap.gateway.channels.matrix.AsyncClient",
                    create=True,
                    return_value=mock_client,
                ),
                patch(
                    "animus_bootstrap.gateway.channels.matrix.RoomMessageText",
                    create=True,
                ),
            ):
                await adapter.connect()
                assert adapter.is_connected is True
                mock_client.add_event_callback.assert_called_once()

                await adapter.disconnect()
                assert adapter.is_connected is False
                mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_message(self) -> None:
        with patch("animus_bootstrap.gateway.channels.matrix.HAS_MATRIX", True):
            from animus_bootstrap.gateway.channels.matrix import MatrixAdapter

            adapter = MatrixAdapter(homeserver="https://matrix.org", access_token="tok")

            mock_result = MagicMock()
            mock_result.event_id = "$event123"
            mock_client = AsyncMock()
            mock_client.room_send = AsyncMock(return_value=mock_result)
            adapter._client = mock_client

            resp = GatewayResponse(
                text="Hello Matrix!",
                channel="matrix",
                metadata={"room_id": "!room:matrix.org"},
            )

            msg_id = await adapter.send_message(resp)
            assert msg_id == "$event123"

    @pytest.mark.asyncio
    async def test_send_message_missing_room_id(self) -> None:
        with patch("animus_bootstrap.gateway.channels.matrix.HAS_MATRIX", True):
            from animus_bootstrap.gateway.channels.matrix import MatrixAdapter

            adapter = MatrixAdapter(homeserver="https://matrix.org", access_token="tok")
            adapter._client = AsyncMock()

            resp = GatewayResponse(text="Hello!", channel="matrix", metadata={})
            with pytest.raises(ValueError, match="room_id"):
                await adapter.send_message(resp)

    @pytest.mark.asyncio
    async def test_send_message_not_connected(self) -> None:
        with patch("animus_bootstrap.gateway.channels.matrix.HAS_MATRIX", True):
            from animus_bootstrap.gateway.channels.matrix import MatrixAdapter

            adapter = MatrixAdapter(homeserver="https://matrix.org", access_token="tok")

            resp = GatewayResponse(text="Hello!", channel="matrix", metadata={"room_id": "!r:x"})
            with pytest.raises(RuntimeError, match="not connected"):
                await adapter.send_message(resp)

    @pytest.mark.asyncio
    async def test_on_message_stores_callback(self) -> None:
        with patch("animus_bootstrap.gateway.channels.matrix.HAS_MATRIX", True):
            from animus_bootstrap.gateway.channels.matrix import MatrixAdapter

            adapter = MatrixAdapter(homeserver="https://matrix.org", access_token="tok")
            cb = AsyncMock()
            await adapter.on_message(cb)
            assert adapter._callback is cb

    @pytest.mark.asyncio
    async def test_health_check_connected(self) -> None:
        with patch("animus_bootstrap.gateway.channels.matrix.HAS_MATRIX", True):
            from animus_bootstrap.gateway.channels.matrix import MatrixAdapter

            adapter = MatrixAdapter(homeserver="https://matrix.org", access_token="tok")

            mock_resp = MagicMock()
            mock_resp.user_id = "@bot:matrix.org"
            mock_client = AsyncMock()
            mock_client.whoami = AsyncMock(return_value=mock_resp)
            adapter._client = mock_client

            health = await adapter.health_check()
            assert health.connected is True

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self) -> None:
        with patch("animus_bootstrap.gateway.channels.matrix.HAS_MATRIX", True):
            from animus_bootstrap.gateway.channels.matrix import MatrixAdapter

            adapter = MatrixAdapter(homeserver="https://matrix.org", access_token="tok")
            health = await adapter.health_check()
            assert health.connected is False

    @pytest.mark.asyncio
    async def test_health_check_error(self) -> None:
        with patch("animus_bootstrap.gateway.channels.matrix.HAS_MATRIX", True):
            from animus_bootstrap.gateway.channels.matrix import MatrixAdapter

            adapter = MatrixAdapter(homeserver="https://matrix.org", access_token="tok")

            mock_client = AsyncMock()
            mock_client.whoami = AsyncMock(side_effect=RuntimeError("timeout"))
            adapter._client = mock_client

            health = await adapter.health_check()
            assert health.connected is False
            assert "timeout" in health.error


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


class TestSignalAdapter:
    """SignalAdapter tests."""

    def test_constructor(self) -> None:
        from animus_bootstrap.gateway.channels.signal_channel import SignalAdapter

        adapter = SignalAdapter(phone_number="+1234567890")
        assert adapter.name == "signal"
        assert adapter.is_connected is False
        assert adapter._phone == "+1234567890"

    @pytest.mark.asyncio
    async def test_connect_signal_cli_not_found(self) -> None:
        from animus_bootstrap.gateway.channels.signal_channel import SignalAdapter

        adapter = SignalAdapter(phone_number="+1", signal_cli_path="/nonexistent/signal-cli")

        with pytest.raises(FileNotFoundError, match="signal-cli not found"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        from animus_bootstrap.gateway.channels.signal_channel import SignalAdapter

        adapter = SignalAdapter(phone_number="+1234567890")

        with (
            patch(
                "animus_bootstrap.gateway.channels.signal_channel.shutil.which",
                return_value="/usr/bin/signal-cli",
            ),
            patch(
                "animus_bootstrap.gateway.channels.signal_channel.asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
            ) as mock_exec,
        ):
            # Mock the receive loop subprocess
            mock_proc = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(return_value=b"")
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            await adapter.connect()
            assert adapter.is_connected is True

            await adapter.disconnect()
            assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_send_message(self) -> None:
        from animus_bootstrap.gateway.channels.signal_channel import SignalAdapter

        adapter = SignalAdapter(phone_number="+1234567890")

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch(
            "animus_bootstrap.gateway.channels.signal_channel.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ):
            resp = GatewayResponse(
                text="Hello Signal!",
                channel="signal",
                metadata={"recipient": "+0987654321"},
            )

            msg_id = await adapter.send_message(resp)
            assert msg_id == ""

    @pytest.mark.asyncio
    async def test_send_message_failure(self) -> None:
        from animus_bootstrap.gateway.channels.signal_channel import SignalAdapter

        adapter = SignalAdapter(phone_number="+1234567890")

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error sending"))
        mock_proc.returncode = 1

        with patch(
            "animus_bootstrap.gateway.channels.signal_channel.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ):
            resp = GatewayResponse(
                text="Hello!",
                channel="signal",
                metadata={"recipient": "+0987654321"},
            )
            with pytest.raises(RuntimeError, match="signal-cli send failed"):
                await adapter.send_message(resp)

    @pytest.mark.asyncio
    async def test_send_message_missing_recipient(self) -> None:
        from animus_bootstrap.gateway.channels.signal_channel import SignalAdapter

        adapter = SignalAdapter(phone_number="+1234567890")

        resp = GatewayResponse(text="Hello!", channel="signal", metadata={})
        with pytest.raises(ValueError, match="recipient"):
            await adapter.send_message(resp)

    @pytest.mark.asyncio
    async def test_on_message_stores_callback(self) -> None:
        from animus_bootstrap.gateway.channels.signal_channel import SignalAdapter

        adapter = SignalAdapter(phone_number="+1234567890")
        cb = AsyncMock()
        await adapter.on_message(cb)
        assert adapter._callback is cb

    @pytest.mark.asyncio
    async def test_health_check_ok(self) -> None:
        from animus_bootstrap.gateway.channels.signal_channel import SignalAdapter

        adapter = SignalAdapter(phone_number="+1234567890")
        adapter.is_connected = True

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"signal-cli 0.13.0", b""))
        mock_proc.returncode = 0

        with patch(
            "animus_bootstrap.gateway.channels.signal_channel.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ):
            health = await adapter.health_check()
            assert health.connected is True
            assert health.error is None

    @pytest.mark.asyncio
    async def test_health_check_cli_not_found(self) -> None:
        from animus_bootstrap.gateway.channels.signal_channel import SignalAdapter

        adapter = SignalAdapter(phone_number="+1", signal_cli_path="/bad/path")

        with patch(
            "animus_bootstrap.gateway.channels.signal_channel.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            side_effect=FileNotFoundError("No such file"),
        ):
            health = await adapter.health_check()
            assert health.connected is False
            assert "not found" in health.error

    @pytest.mark.asyncio
    async def test_health_check_nonzero_exit(self) -> None:
        from animus_bootstrap.gateway.channels.signal_channel import SignalAdapter

        adapter = SignalAdapter(phone_number="+1234567890")

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))
        mock_proc.returncode = 1

        with patch(
            "animus_bootstrap.gateway.channels.signal_channel.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ):
            health = await adapter.health_check()
            assert health.connected is False

    @pytest.mark.asyncio
    async def test_process_signal_message(self) -> None:
        from animus_bootstrap.gateway.channels.signal_channel import SignalAdapter

        adapter = SignalAdapter(phone_number="+1234567890")
        cb = AsyncMock()
        adapter._callback = cb

        data = {
            "envelope": {
                "source": "+0987654321",
                "sourceName": "Alice",
                "timestamp": 1234567890,
                "dataMessage": {
                    "message": "Hello from Signal!",
                    "groupInfo": {},
                },
            }
        }

        await adapter._process_signal_message(data)
        cb.assert_awaited_once()
        gw_msg = cb.call_args[0][0]
        assert gw_msg.text == "Hello from Signal!"
        assert gw_msg.channel == "signal"
        assert gw_msg.sender_id == "+0987654321"

    @pytest.mark.asyncio
    async def test_process_signal_message_empty(self) -> None:
        from animus_bootstrap.gateway.channels.signal_channel import SignalAdapter

        adapter = SignalAdapter(phone_number="+1234567890")
        cb = AsyncMock()
        adapter._callback = cb

        # Empty message should not trigger callback
        await adapter._process_signal_message({"envelope": {"dataMessage": {}}})
        cb.assert_not_awaited()


# ---------------------------------------------------------------------------
# WhatsApp
# ---------------------------------------------------------------------------


class TestWhatsAppAdapter:
    """WhatsAppAdapter tests."""

    def test_constructor(self) -> None:
        from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(
            phone_number_id="12345",
            access_token="tok",
            verify_token="vtok",
        )
        assert adapter.name == "whatsapp"
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(phone_number_id="12345", access_token="tok")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("animus_bootstrap.gateway.channels.whatsapp.httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_http.aclose = AsyncMock()
            MockClient.return_value = mock_http

            await adapter.connect()
            assert adapter.is_connected is True

            await adapter.disconnect()
            assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_failure(self) -> None:
        from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(phone_number_id="12345", access_token="bad")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("animus_bootstrap.gateway.channels.whatsapp.httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_http.aclose = AsyncMock()
            MockClient.return_value = mock_http

            with pytest.raises(ConnectionError, match="401"):
                await adapter.connect()

    @pytest.mark.asyncio
    async def test_send_message(self) -> None:
        from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(phone_number_id="12345", access_token="tok")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"messages": [{"id": "wamid.abc123"}]}

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        adapter._http = mock_http

        resp = GatewayResponse(
            text="Hello WhatsApp!",
            channel="whatsapp",
            metadata={"to": "+1234567890"},
        )

        msg_id = await adapter.send_message(resp)
        assert msg_id == "wamid.abc123"

    @pytest.mark.asyncio
    async def test_send_message_failure(self) -> None:
        from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(phone_number_id="12345", access_token="tok")

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        adapter._http = mock_http

        resp = GatewayResponse(
            text="Hello!",
            channel="whatsapp",
            metadata={"to": "+1234567890"},
        )
        with pytest.raises(RuntimeError, match="send failed"):
            await adapter.send_message(resp)

    @pytest.mark.asyncio
    async def test_send_message_missing_to(self) -> None:
        from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(phone_number_id="12345", access_token="tok")
        adapter._http = AsyncMock()

        resp = GatewayResponse(text="Hello!", channel="whatsapp", metadata={})
        with pytest.raises(ValueError, match="to"):
            await adapter.send_message(resp)

    @pytest.mark.asyncio
    async def test_send_message_not_connected(self) -> None:
        from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(phone_number_id="12345", access_token="tok")

        resp = GatewayResponse(text="Hello!", channel="whatsapp", metadata={"to": "+1"})
        with pytest.raises(RuntimeError, match="not connected"):
            await adapter.send_message(resp)

    @pytest.mark.asyncio
    async def test_on_message_stores_callback(self) -> None:
        from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(phone_number_id="12345", access_token="tok")
        cb = AsyncMock()
        await adapter.on_message(cb)
        assert adapter._callback is cb

    @pytest.mark.asyncio
    async def test_handle_webhook(self) -> None:
        from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(phone_number_id="12345", access_token="tok")
        cb = AsyncMock()
        adapter._callback = cb

        webhook_data = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "metadata": {"phone_number_id": "12345"},
                                "contacts": [
                                    {
                                        "wa_id": "+sender",
                                        "profile": {"name": "Alice"},
                                    }
                                ],
                                "messages": [
                                    {
                                        "from": "+sender",
                                        "id": "wamid.xyz",
                                        "type": "text",
                                        "text": {"body": "Hi from WhatsApp!"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }

        await adapter.handle_webhook(webhook_data)
        cb.assert_awaited_once()
        gw_msg = cb.call_args[0][0]
        assert gw_msg.text == "Hi from WhatsApp!"
        assert gw_msg.channel == "whatsapp"
        assert gw_msg.sender_name == "Alice"

    @pytest.mark.asyncio
    async def test_handle_webhook_non_text_ignored(self) -> None:
        from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(phone_number_id="12345", access_token="tok")
        cb = AsyncMock()
        adapter._callback = cb

        webhook_data = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [{"from": "+sender", "id": "wamid.1", "type": "image"}],
                                "contacts": [],
                            }
                        }
                    ]
                }
            ]
        }

        await adapter.handle_webhook(webhook_data)
        cb.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_health_check_connected(self) -> None:
        from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(phone_number_id="12345", access_token="tok")

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        adapter._http = mock_http

        health = await adapter.health_check()
        assert health.connected is True

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self) -> None:
        from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(phone_number_id="12345", access_token="tok")
        health = await adapter.health_check()
        assert health.connected is False

    @pytest.mark.asyncio
    async def test_health_check_error(self) -> None:
        from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(phone_number_id="12345", access_token="tok")

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=RuntimeError("timeout"))
        adapter._http = mock_http

        health = await adapter.health_check()
        assert health.connected is False
        assert "timeout" in health.error


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


class TestEmailAdapter:
    """EmailAdapter tests."""

    def test_constructor(self) -> None:
        from animus_bootstrap.gateway.channels.email_channel import EmailAdapter

        adapter = EmailAdapter(
            imap_host="imap.example.com",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass",
        )
        assert adapter.name == "email"
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        from animus_bootstrap.gateway.channels.email_channel import EmailAdapter

        adapter = EmailAdapter(
            imap_host="imap.example.com",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass",
        )

        with patch.object(adapter, "_test_imap"):
            await adapter.connect()
            assert adapter.is_connected is True

            await adapter.disconnect()
            assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_imap_failure(self) -> None:
        from animus_bootstrap.gateway.channels.email_channel import EmailAdapter

        adapter = EmailAdapter(
            imap_host="imap.example.com",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="badpass",
        )

        with patch.object(adapter, "_test_imap", side_effect=imaplib.IMAP4.error("auth failed")):
            with pytest.raises(imaplib.IMAP4.error):
                await adapter.connect()

    @pytest.mark.asyncio
    async def test_send_message_with_aiosmtplib(self) -> None:
        from animus_bootstrap.gateway.channels.email_channel import EmailAdapter

        adapter = EmailAdapter(
            imap_host="imap.example.com",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass",
        )

        with (
            patch("animus_bootstrap.gateway.channels.email_channel.HAS_AIOSMTPLIB", True),
            patch(
                "animus_bootstrap.gateway.channels.email_channel.aiosmtplib",
                create=True,
            ) as mock_aiosmtp,
        ):
            mock_aiosmtp.send = AsyncMock()

            resp = GatewayResponse(
                text="Hello via email!",
                channel="email",
                metadata={"to": "recipient@example.com", "subject": "Test"},
            )

            msg_id = await adapter.send_message(resp)
            assert "@animus>" in msg_id
            mock_aiosmtp.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_message_smtp_fallback(self) -> None:
        from animus_bootstrap.gateway.channels.email_channel import EmailAdapter

        adapter = EmailAdapter(
            imap_host="imap.example.com",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass",
        )

        with (
            patch("animus_bootstrap.gateway.channels.email_channel.HAS_AIOSMTPLIB", False),
            patch.object(adapter, "_send_smtp_sync"),
        ):
            resp = GatewayResponse(
                text="Hello via email!",
                channel="email",
                metadata={"to": "recipient@example.com"},
            )

            msg_id = await adapter.send_message(resp)
            assert "@animus>" in msg_id

    @pytest.mark.asyncio
    async def test_send_message_missing_to(self) -> None:
        from animus_bootstrap.gateway.channels.email_channel import EmailAdapter

        adapter = EmailAdapter(
            imap_host="imap.example.com",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass",
        )

        resp = GatewayResponse(text="Hello!", channel="email", metadata={})
        with pytest.raises(ValueError, match="to"):
            await adapter.send_message(resp)

    @pytest.mark.asyncio
    async def test_on_message_stores_callback(self) -> None:
        from animus_bootstrap.gateway.channels.email_channel import EmailAdapter

        adapter = EmailAdapter(
            imap_host="imap.example.com",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass",
        )
        cb = AsyncMock()
        await adapter.on_message(cb)
        assert adapter._callback is cb

    @pytest.mark.asyncio
    async def test_health_check_ok(self) -> None:
        from animus_bootstrap.gateway.channels.email_channel import EmailAdapter

        adapter = EmailAdapter(
            imap_host="imap.example.com",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass",
        )
        adapter.is_connected = True

        with patch.object(adapter, "_test_imap"):
            health = await adapter.health_check()
            assert health.connected is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self) -> None:
        from animus_bootstrap.gateway.channels.email_channel import EmailAdapter

        adapter = EmailAdapter(
            imap_host="imap.example.com",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass",
        )

        with patch.object(adapter, "_test_imap", side_effect=RuntimeError("connection refused")):
            health = await adapter.health_check()
            assert health.connected is False
            assert "connection refused" in health.error

    def test_fetch_new_messages_parses_email(self) -> None:
        from animus_bootstrap.gateway.channels.email_channel import EmailAdapter

        adapter = EmailAdapter(
            imap_host="imap.example.com",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass",
        )

        # Build a minimal RFC822 email
        import email.mime.text as mime_text

        msg = mime_text.MIMEText("Test body")
        msg["From"] = "sender@example.com"
        msg["Subject"] = "Test subject"
        msg["Message-ID"] = "<test@example.com>"
        raw_bytes = msg.as_bytes()

        mock_conn = MagicMock()
        mock_conn.login = MagicMock()
        mock_conn.select = MagicMock(return_value=("OK", [b"1"]))
        mock_conn.search = MagicMock(return_value=("OK", [b"1"]))
        mock_conn.fetch = MagicMock(return_value=("OK", [(b"1 (RFC822 {123})", raw_bytes)]))
        mock_conn.logout = MagicMock()

        with patch(
            "animus_bootstrap.gateway.channels.email_channel.imaplib.IMAP4_SSL",
            return_value=mock_conn,
        ):
            messages = adapter._fetch_new_messages()
            assert len(messages) == 1
            assert messages[0].text == "Test body"
            assert messages[0].sender_id == "sender@example.com"

    def test_fetch_new_messages_no_unseen(self) -> None:
        from animus_bootstrap.gateway.channels.email_channel import EmailAdapter

        adapter = EmailAdapter(
            imap_host="imap.example.com",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass",
        )

        mock_conn = MagicMock()
        mock_conn.login = MagicMock()
        mock_conn.select = MagicMock(return_value=("OK", [b"0"]))
        mock_conn.search = MagicMock(return_value=("OK", [b""]))
        mock_conn.logout = MagicMock()

        with patch(
            "animus_bootstrap.gateway.channels.email_channel.imaplib.IMAP4_SSL",
            return_value=mock_conn,
        ):
            messages = adapter._fetch_new_messages()
            assert len(messages) == 0

    def test_send_smtp_sync(self) -> None:
        from animus_bootstrap.gateway.channels.email_channel import EmailAdapter

        adapter = EmailAdapter(
            imap_host="imap.example.com",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass",
        )

        import email.mime.text as mime_text

        msg = mime_text.MIMEText("body")
        msg["Subject"] = "test"
        msg["From"] = "user@example.com"
        msg["To"] = "other@example.com"

        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch(
            "animus_bootstrap.gateway.channels.email_channel.smtplib.SMTP",
            return_value=mock_smtp,
        ):
            adapter._send_smtp_sync(msg)
            mock_smtp.starttls.assert_called_once()
            mock_smtp.login.assert_called_once_with("user@example.com", "pass")
            mock_smtp.send_message.assert_called_once_with(msg)


# ---------------------------------------------------------------------------
# __init__.py exports
# ---------------------------------------------------------------------------


class TestChannelExports:
    """Verify the channels package exports all adapters."""

    def test_all_adapters_exported(self) -> None:
        from animus_bootstrap.gateway.channels import __all__

        expected = {
            "DiscordAdapter",
            "EmailAdapter",
            "MatrixAdapter",
            "SignalAdapter",
            "SlackAdapter",
            "TelegramAdapter",
            "WhatsAppAdapter",
        }
        assert set(__all__) == expected

    def test_import_signal_adapter(self) -> None:
        """Signal adapter has no optional deps, so it should always import."""
        from animus_bootstrap.gateway.channels.signal_channel import SignalAdapter

        adapter = SignalAdapter(phone_number="+1")
        assert adapter.name == "signal"

    def test_import_whatsapp_adapter(self) -> None:
        """WhatsApp adapter uses httpx (core dep), should always import."""
        from animus_bootstrap.gateway.channels.whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(phone_number_id="1", access_token="t")
        assert adapter.name == "whatsapp"

    def test_import_email_adapter(self) -> None:
        """Email adapter uses stdlib, should always import."""
        from animus_bootstrap.gateway.channels.email_channel import EmailAdapter

        adapter = EmailAdapter(
            imap_host="imap.x.com",
            smtp_host="smtp.x.com",
            username="u",
            password="p",
        )
        assert adapter.name == "email"
