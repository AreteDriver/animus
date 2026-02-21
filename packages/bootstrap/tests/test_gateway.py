"""Tests for the Animus Bootstrap gateway module."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from animus_bootstrap.config.schema import (
    AnimusConfig,
    ChannelsSection,
    DiscordChannelConfig,
    EmailChannelConfig,
    GatewaySection,
    MatrixChannelConfig,
    SignalChannelConfig,
    SlackChannelConfig,
    TelegramChannelConfig,
    WebchatChannelConfig,
    WhatsappChannelConfig,
)
from animus_bootstrap.gateway.cognitive import (
    AnthropicBackend,
    ForgeBackend,
    OllamaBackend,
)
from animus_bootstrap.gateway.cognitive_types import CognitiveResponse
from animus_bootstrap.gateway.models import (
    Attachment,
    ChannelHealth,
    GatewayMessage,
    GatewayResponse,
    create_message,
)
from animus_bootstrap.gateway.router import MessageRouter
from animus_bootstrap.gateway.session import Session, SessionManager

_DUMMY_REQUEST = httpx.Request("POST", "https://test")

# ======================================================================
# Models
# ======================================================================


class TestAttachment:
    def test_basic_fields(self) -> None:
        att = Attachment(filename="photo.jpg", content_type="image/jpeg")
        assert att.filename == "photo.jpg"
        assert att.content_type == "image/jpeg"
        assert att.data is None
        assert att.url is None

    def test_with_data(self) -> None:
        att = Attachment(filename="f.bin", content_type="application/octet-stream", data=b"\x00")
        assert att.data == b"\x00"

    def test_with_url(self) -> None:
        att = Attachment(filename="f.png", content_type="image/png", url="https://example.com/f")
        assert att.url == "https://example.com/f"


class TestGatewayMessage:
    def test_required_fields(self) -> None:
        now = datetime.now(UTC)
        msg = GatewayMessage(
            id="abc",
            channel="telegram",
            channel_message_id="123",
            sender_id="user1",
            sender_name="Alice",
            text="hello",
            timestamp=now,
        )
        assert msg.id == "abc"
        assert msg.channel == "telegram"
        assert msg.sender_id == "user1"
        assert msg.text == "hello"
        assert msg.role == "user"
        assert msg.attachments == []
        assert msg.reply_to is None
        assert msg.metadata == {}

    def test_with_attachments(self) -> None:
        att = Attachment(filename="x.txt", content_type="text/plain")
        msg = GatewayMessage(
            id="1",
            channel="discord",
            channel_message_id="2",
            sender_id="u",
            sender_name="U",
            text="see file",
            timestamp=datetime.now(UTC),
            attachments=[att],
        )
        assert len(msg.attachments) == 1
        assert msg.attachments[0].filename == "x.txt"


class TestGatewayResponse:
    def test_defaults(self) -> None:
        resp = GatewayResponse(text="hi", channel="webchat")
        assert resp.text == "hi"
        assert resp.channel == "webchat"
        assert resp.attachments == []
        assert resp.channel_message_id is None
        assert resp.metadata == {}


class TestChannelHealth:
    def test_connected(self) -> None:
        h = ChannelHealth(channel="telegram", connected=True, latency_ms=42.0)
        assert h.connected is True
        assert h.latency_ms == 42.0
        assert h.error is None

    def test_disconnected_with_error(self) -> None:
        h = ChannelHealth(channel="discord", connected=False, error="timeout")
        assert h.connected is False
        assert h.error == "timeout"


class TestCreateMessage:
    def test_auto_generates_id_and_timestamp(self) -> None:
        msg = create_message(
            channel="webchat",
            sender_id="user1",
            sender_name="Alice",
            text="hello",
        )
        # Validate UUID format
        uuid.UUID(msg.id)
        assert msg.channel == "webchat"
        assert msg.sender_id == "user1"
        assert msg.sender_name == "Alice"
        assert msg.text == "hello"
        assert isinstance(msg.timestamp, datetime)
        assert msg.role == "user"

    def test_override_id_and_timestamp(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        msg = create_message(
            channel="telegram",
            sender_id="u2",
            sender_name="Bob",
            text="hi",
            id="custom-id",
            timestamp=ts,
        )
        assert msg.id == "custom-id"
        assert msg.timestamp == ts

    def test_kwargs_passthrough(self) -> None:
        msg = create_message(
            channel="discord",
            sender_id="u3",
            sender_name="Carol",
            text="hey",
            channel_message_id="msg-42",
            role="assistant",
            reply_to="prev-id",
            metadata={"key": "val"},
        )
        assert msg.channel_message_id == "msg-42"
        assert msg.role == "assistant"
        assert msg.reply_to == "prev-id"
        assert msg.metadata == {"key": "val"}


# ======================================================================
# SessionManager
# ======================================================================


class TestSessionManager:
    @pytest.fixture()
    def session_mgr(self, tmp_path: Path) -> SessionManager:
        mgr = SessionManager(db_path=tmp_path / "test.db")
        yield mgr
        mgr.close()

    @pytest.mark.asyncio()
    async def test_create_session(self, session_mgr: SessionManager) -> None:
        msg = create_message("telegram", "user1", "Alice", "hello")
        session = await session_mgr.get_or_create_session(msg)
        assert isinstance(session, Session)
        assert session.user_id == "user1"
        assert session.user_name == "Alice"
        assert session.messages == []
        assert "telegram" in session.channel_ids

    @pytest.mark.asyncio()
    async def test_get_existing_session(self, session_mgr: SessionManager) -> None:
        msg1 = create_message("telegram", "user1", "Alice", "hello")
        session1 = await session_mgr.get_or_create_session(msg1)

        msg2 = create_message("telegram", "user1", "Alice", "world")
        session2 = await session_mgr.get_or_create_session(msg2)

        assert session1.id == session2.id

    @pytest.mark.asyncio()
    async def test_different_users_different_sessions(self, session_mgr: SessionManager) -> None:
        msg1 = create_message("telegram", "user1", "Alice", "hello")
        session1 = await session_mgr.get_or_create_session(msg1)

        msg2 = create_message("telegram", "user2", "Bob", "hello")
        session2 = await session_mgr.get_or_create_session(msg2)

        assert session1.id != session2.id

    @pytest.mark.asyncio()
    async def test_add_message(self, session_mgr: SessionManager) -> None:
        msg = create_message("telegram", "user1", "Alice", "hello")
        session = await session_mgr.get_or_create_session(msg)
        await session_mgr.add_message(session, msg)
        assert len(session.messages) == 1
        assert session.messages[0].text == "hello"

    @pytest.mark.asyncio()
    async def test_get_context(self, session_mgr: SessionManager) -> None:
        msg1 = create_message("telegram", "user1", "Alice", "hello")
        session = await session_mgr.get_or_create_session(msg1)
        await session_mgr.add_message(session, msg1)

        msg2 = create_message("telegram", "animus", "Animus", "hi there", role="assistant")
        await session_mgr.add_message(session, msg2)

        context = await session_mgr.get_context(session)
        assert len(context) == 2
        assert context[0] == {"role": "user", "content": "hello"}
        assert context[1] == {"role": "assistant", "content": "hi there"}

    @pytest.mark.asyncio()
    async def test_get_context_respects_limit(self, session_mgr: SessionManager) -> None:
        msg = create_message("telegram", "user1", "Alice", "first")
        session = await session_mgr.get_or_create_session(msg)
        # Add several messages
        for i in range(5):
            m = create_message("telegram", "user1", "Alice", f"msg-{i}")
            await session_mgr.add_message(session, m)

        context = await session_mgr.get_context(session, max_messages=2)
        assert len(context) == 2
        # Should be the MOST RECENT messages
        assert context[0]["content"] == "msg-3"
        assert context[1]["content"] == "msg-4"

    @pytest.mark.asyncio()
    async def test_link_channel(self, session_mgr: SessionManager) -> None:
        msg = create_message("telegram", "user1", "Alice", "hello")
        session = await session_mgr.get_or_create_session(msg)
        await session_mgr.link_channel(session.id, "discord", "discord-user1")

        # Reload session via discord identity
        msg2 = create_message("discord", "discord-user1", "Alice", "from discord")
        session2 = await session_mgr.get_or_create_session(msg2)
        assert session2.id == session.id

    @pytest.mark.asyncio()
    async def test_link_channel_idempotent(self, session_mgr: SessionManager) -> None:
        msg = create_message("telegram", "user1", "Alice", "hello")
        session = await session_mgr.get_or_create_session(msg)
        # Link same channel twice — should not raise
        await session_mgr.link_channel(session.id, "discord", "d-user1")
        await session_mgr.link_channel(session.id, "discord", "d-user1")

    @pytest.mark.asyncio()
    async def test_get_recent_messages(self, session_mgr: SessionManager) -> None:
        msg = create_message("telegram", "user1", "Alice", "hello")
        session = await session_mgr.get_or_create_session(msg)
        await session_mgr.add_message(session, msg)

        msg2 = create_message("webchat", "user2", "Bob", "hey")
        session2 = await session_mgr.get_or_create_session(msg2)
        await session_mgr.add_message(session2, msg2)

        recent = await session_mgr.get_recent_messages(limit=10)
        assert len(recent) == 2

    @pytest.mark.asyncio()
    async def test_prune_old_sessions(self, session_mgr: SessionManager) -> None:
        # Create a session and backdate it
        msg = create_message("telegram", "user1", "Alice", "old")
        session = await session_mgr.get_or_create_session(msg)
        await session_mgr.add_message(session, msg)

        # Manually backdate the session
        old_time = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        session_mgr._conn.execute(
            "UPDATE gateway_sessions SET last_active = ? WHERE id = ?",
            (old_time, session.id),
        )
        session_mgr._conn.commit()

        pruned = await session_mgr.prune_old_sessions(max_age_days=30)
        assert pruned == 1

    @pytest.mark.asyncio()
    async def test_prune_nothing_to_prune(self, session_mgr: SessionManager) -> None:
        msg = create_message("telegram", "user1", "Alice", "fresh")
        session = await session_mgr.get_or_create_session(msg)
        await session_mgr.add_message(session, msg)

        pruned = await session_mgr.prune_old_sessions(max_age_days=30)
        assert pruned == 0


# ======================================================================
# Cognitive Backends
# ======================================================================


class TestAnthropicBackend:
    @pytest.mark.asyncio()
    async def test_generate_response(self) -> None:
        backend = AnthropicBackend(api_key="sk-test", model="claude-test")
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={
                "content": [{"type": "text", "text": "Hello from Claude"}],
                "model": "claude-test",
                "role": "assistant",
            },
        )

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await backend.generate_response(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="You are helpful.",
            )

        assert result == "Hello from Claude"
        # Verify request format
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["model"] == "claude-test"
        assert payload["system"] == "You are helpful."
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert payload["max_tokens"] == 4096
        # Verify headers
        headers = call_kwargs.kwargs["headers"]
        assert headers["x-api-key"] == "sk-test"
        assert headers["anthropic-version"] == "2023-06-01"

    @pytest.mark.asyncio()
    async def test_no_system_prompt(self) -> None:
        backend = AnthropicBackend(api_key="sk-test")
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={"content": [{"type": "text", "text": "ok"}]},
        )

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await backend.generate_response(
                messages=[{"role": "user", "content": "test"}],
            )

        payload = mock_client.post.call_args.kwargs["json"]
        assert "system" not in payload


class TestOllamaBackend:
    @pytest.mark.asyncio()
    async def test_generate_response(self) -> None:
        backend = OllamaBackend(model="llama3.1:8b", host="http://localhost:11434")
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={"message": {"role": "assistant", "content": "Hello from Ollama"}},
        )

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await backend.generate_response(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="Be concise.",
            )

        assert result == "Hello from Ollama"
        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["model"] == "llama3.1:8b"
        assert payload["stream"] is False
        # System prompt prepended as first message
        assert payload["messages"][0] == {"role": "system", "content": "Be concise."}
        assert payload["messages"][1] == {"role": "user", "content": "hi"}

    @pytest.mark.asyncio()
    async def test_no_system_prompt(self) -> None:
        backend = OllamaBackend()
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={"message": {"role": "assistant", "content": "ok"}},
        )

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await backend.generate_response(
                messages=[{"role": "user", "content": "test"}],
            )

        payload = mock_client.post.call_args.kwargs["json"]
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    @pytest.mark.asyncio()
    async def test_host_trailing_slash_stripped(self) -> None:
        backend = OllamaBackend(host="http://localhost:11434/")
        assert backend._host == "http://localhost:11434"


class TestForgeBackend:
    @pytest.mark.asyncio()
    async def test_generate_response(self) -> None:
        backend = ForgeBackend(host="localhost", port=8000, api_key="forge-key")
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={"response": "Hello from Forge"},
        )

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await backend.generate_response(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="You are Animus.",
                max_tokens=2048,
            )

        assert result == "Hello from Forge"
        call_kwargs = mock_client.post.call_args
        assert "http://localhost:8000/api/v1/chat" in call_kwargs.args[0]
        payload = call_kwargs.kwargs["json"]
        assert payload["system_prompt"] == "You are Animus."
        assert payload["max_tokens"] == 2048
        headers = call_kwargs.kwargs["headers"]
        assert headers["authorization"] == "Bearer forge-key"

    @pytest.mark.asyncio()
    async def test_no_api_key(self) -> None:
        backend = ForgeBackend(host="localhost", port=9000)
        mock_response = httpx.Response(200, request=_DUMMY_REQUEST, json={"text": "ok"})

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await backend.generate_response(
                messages=[{"role": "user", "content": "test"}],
            )

        assert result == "ok"
        headers = mock_client.post.call_args.kwargs["headers"]
        assert "authorization" not in headers

    @pytest.mark.asyncio()
    async def test_fallback_text_key(self) -> None:
        """ForgeBackend falls back to 'text' key if 'response' is absent."""
        backend = ForgeBackend()
        mock_response = httpx.Response(200, request=_DUMMY_REQUEST, json={"text": "fallback"})

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await backend.generate_response(
                messages=[{"role": "user", "content": "test"}],
            )

        assert result == "fallback"


# ======================================================================
# MessageRouter
# ======================================================================


class TestMessageRouter:
    @pytest.fixture()
    def mock_cognitive(self) -> AsyncMock:
        cog = AsyncMock()
        cog.generate_response = AsyncMock(return_value="I'm Animus.")
        return cog

    @pytest.fixture()
    def session_mgr(self, tmp_path: Path) -> SessionManager:
        mgr = SessionManager(db_path=tmp_path / "router-test.db")
        yield mgr
        mgr.close()

    @pytest.fixture()
    def router(self, mock_cognitive: AsyncMock, session_mgr: SessionManager) -> MessageRouter:
        return MessageRouter(cognitive=mock_cognitive, session_manager=session_mgr)

    @pytest.mark.asyncio()
    async def test_handle_message_returns_response(self, router: MessageRouter) -> None:
        msg = create_message("webchat", "user1", "Alice", "hello")
        resp = await router.handle_message(msg)
        assert isinstance(resp, GatewayResponse)
        assert resp.text == "I'm Animus."
        assert resp.channel == "webchat"

    @pytest.mark.asyncio()
    async def test_handle_message_stores_both_messages(
        self, router: MessageRouter, session_mgr: SessionManager
    ) -> None:
        msg = create_message("webchat", "user1", "Alice", "hello")
        await router.handle_message(msg)

        # Session should have 2 messages: user + assistant
        msg2 = create_message("webchat", "user1", "Alice", "check")
        session = await session_mgr.get_or_create_session(msg2)
        context = await session_mgr.get_context(session)
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "hello"
        assert context[1]["role"] == "assistant"
        assert context[1]["content"] == "I'm Animus."

    @pytest.mark.asyncio()
    async def test_handle_message_calls_cognitive(
        self, router: MessageRouter, mock_cognitive: AsyncMock
    ) -> None:
        msg = create_message("webchat", "user1", "Alice", "hi")
        await router.handle_message(msg)
        mock_cognitive.generate_response.assert_awaited_once()

    def test_register_channel(self, router: MessageRouter) -> None:
        adapter = MagicMock()
        router.register_channel("telegram", adapter)
        assert "telegram" in router.channels
        assert router.channels["telegram"] is adapter

    def test_unregister_channel(self, router: MessageRouter) -> None:
        adapter = MagicMock()
        router.register_channel("telegram", adapter)
        router.unregister_channel("telegram")
        assert "telegram" not in router.channels

    def test_unregister_nonexistent_channel(self, router: MessageRouter) -> None:
        # Should not raise
        router.unregister_channel("nonexistent")

    def test_channels_returns_copy(self, router: MessageRouter) -> None:
        adapter = MagicMock()
        router.register_channel("telegram", adapter)
        channels = router.channels
        channels["fake"] = "value"
        assert "fake" not in router.channels

    @pytest.mark.asyncio()
    async def test_broadcast_all_channels(self, router: MessageRouter) -> None:
        adapter1 = AsyncMock()
        adapter2 = AsyncMock()
        router.register_channel("telegram", adapter1)
        router.register_channel("discord", adapter2)

        await router.broadcast("hello everyone")

        adapter1.send.assert_awaited_once_with("hello everyone")
        adapter2.send.assert_awaited_once_with("hello everyone")

    @pytest.mark.asyncio()
    async def test_broadcast_specific_channels(self, router: MessageRouter) -> None:
        adapter1 = AsyncMock()
        adapter2 = AsyncMock()
        router.register_channel("telegram", adapter1)
        router.register_channel("discord", adapter2)

        await router.broadcast("hello telegram", channels=["telegram"])

        adapter1.send.assert_awaited_once_with("hello telegram")
        adapter2.send.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_broadcast_missing_channel_warns(self, router: MessageRouter) -> None:
        """Broadcast to an unregistered channel logs warning but doesn't raise."""
        await router.broadcast("hello", channels=["nonexistent"])

    @pytest.mark.asyncio()
    async def test_broadcast_adapter_error_handled(self, router: MessageRouter) -> None:
        """Adapter failure in broadcast is caught and logged."""
        adapter = AsyncMock()
        adapter.send = AsyncMock(side_effect=RuntimeError("send failed"))
        router.register_channel("broken", adapter)

        # Should not raise
        await router.broadcast("test")


# ======================================================================
# Config Schema Extensions
# ======================================================================


class TestGatewayConfig:
    def test_gateway_section_defaults(self) -> None:
        g = GatewaySection()
        assert g.enabled is True
        assert g.default_backend == "anthropic"
        assert g.system_prompt == ""
        assert g.max_response_tokens == 4096

    def test_gateway_in_animus_config(self) -> None:
        cfg = AnimusConfig()
        assert isinstance(cfg.gateway, GatewaySection)
        assert cfg.gateway.enabled is True

    def test_channels_section_defaults(self) -> None:
        cs = ChannelsSection()
        assert cs.webchat.enabled is True
        assert cs.telegram.enabled is False
        assert cs.discord.enabled is False
        assert cs.slack.enabled is False
        assert cs.matrix.enabled is False
        assert cs.signal.enabled is False
        assert cs.whatsapp.enabled is False
        assert cs.email.enabled is False

    def test_channels_in_animus_config(self) -> None:
        cfg = AnimusConfig()
        assert isinstance(cfg.channels, ChannelsSection)
        assert cfg.channels.webchat.enabled is True

    def test_telegram_config(self) -> None:
        tc = TelegramChannelConfig(enabled=True, bot_token="123:ABC")
        assert tc.enabled is True
        assert tc.bot_token == "123:ABC"

    def test_discord_config(self) -> None:
        dc = DiscordChannelConfig(enabled=True, bot_token="xyz", allowed_guilds=["g1", "g2"])
        assert dc.enabled is True
        assert dc.allowed_guilds == ["g1", "g2"]

    def test_slack_config(self) -> None:
        sc = SlackChannelConfig(enabled=True, bot_token="xoxb-xxx", app_token="xapp-yyy")
        assert sc.bot_token == "xoxb-xxx"
        assert sc.app_token == "xapp-yyy"

    def test_matrix_config(self) -> None:
        mc = MatrixChannelConfig(
            enabled=True,
            homeserver="https://matrix.org",
            access_token="tok",
            room_ids=["!abc:matrix.org"],
        )
        assert mc.homeserver == "https://matrix.org"
        assert mc.room_ids == ["!abc:matrix.org"]

    def test_signal_config(self) -> None:
        sc = SignalChannelConfig(enabled=True, phone_number="+1234567890")
        assert sc.phone_number == "+1234567890"

    def test_whatsapp_config(self) -> None:
        wc = WhatsappChannelConfig(enabled=True, phone_number="+1234567890")
        assert wc.phone_number == "+1234567890"

    def test_email_config(self) -> None:
        ec = EmailChannelConfig(
            enabled=True,
            imap_host="imap.example.com",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="secret",
            poll_interval=120,
        )
        assert ec.imap_host == "imap.example.com"
        assert ec.smtp_host == "smtp.example.com"
        assert ec.poll_interval == 120

    def test_webchat_config_defaults(self) -> None:
        wc = WebchatChannelConfig()
        assert wc.enabled is True

    def test_config_model_dump_includes_new_sections(self) -> None:
        cfg = AnimusConfig()
        data = cfg.model_dump()
        assert "gateway" in data
        assert "channels" in data
        assert data["gateway"]["enabled"] is True
        assert data["channels"]["webchat"]["enabled"] is True


# ======================================================================
# Cognitive Backends — generate_structured
# ======================================================================


class TestAnthropicStructured:
    @pytest.mark.asyncio()
    async def test_generate_structured_text_only(self) -> None:
        backend = AnthropicBackend(api_key="sk-test", model="claude-test")
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={
                "content": [{"type": "text", "text": "Just text, no tools."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 50, "output_tokens": 20},
            },
        )

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await backend.generate_structured(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="You are helpful.",
            )

        assert isinstance(result, CognitiveResponse)
        assert result.text == "Just text, no tools."
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"
        assert result.usage == {"input_tokens": 50, "output_tokens": 20}
        assert result.has_tool_calls is False

    @pytest.mark.asyncio()
    async def test_generate_structured_with_tool_use(self) -> None:
        backend = AnthropicBackend(api_key="sk-test", model="claude-test")
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={
                "content": [
                    {"type": "text", "text": "Let me check that."},
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "web_search",
                        "input": {"query": "test"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        )

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await backend.generate_structured(
                messages=[{"role": "user", "content": "search for test"}],
            )

        assert result.text == "Let me check that."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "toolu_01"
        assert result.tool_calls[0].name == "web_search"
        assert result.tool_calls[0].arguments == {"query": "test"}
        assert result.stop_reason == "tool_use"
        assert result.has_tool_calls is True

    @pytest.mark.asyncio()
    async def test_generate_structured_includes_tools_in_payload(self) -> None:
        backend = AnthropicBackend(api_key="sk-test", model="claude-test")
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {},
            },
        )

        tool_schemas = [
            {
                "name": "web_search",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await backend.generate_structured(
                messages=[{"role": "user", "content": "hi"}],
                tools=tool_schemas,
            )

        payload = mock_client.post.call_args.kwargs["json"]
        assert "tools" in payload
        assert payload["tools"] == tool_schemas

    @pytest.mark.asyncio()
    async def test_generate_structured_no_tools_omits_key(self) -> None:
        backend = AnthropicBackend(api_key="sk-test")
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={
                "content": [{"type": "text", "text": "no tools"}],
                "stop_reason": "end_turn",
                "usage": {},
            },
        )

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await backend.generate_structured(
                messages=[{"role": "user", "content": "hi"}],
            )

        payload = mock_client.post.call_args.kwargs["json"]
        assert "tools" not in payload

    @pytest.mark.asyncio()
    async def test_generate_structured_multiple_tool_calls(self) -> None:
        backend = AnthropicBackend(api_key="sk-test")
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "search",
                        "input": {"q": "a"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_02",
                        "name": "read_file",
                        "input": {"path": "/tmp/x"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 80, "output_tokens": 40},
            },
        )

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await backend.generate_structured(
                messages=[{"role": "user", "content": "do both"}],
            )

        assert result.text == ""
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[1].name == "read_file"

    @pytest.mark.asyncio()
    async def test_generate_structured_with_system_prompt(self) -> None:
        backend = AnthropicBackend(api_key="sk-test")
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {},
            },
        )

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await backend.generate_structured(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="Be helpful.",
            )

        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["system"] == "Be helpful."


class TestOllamaStructured:
    @pytest.mark.asyncio()
    async def test_generate_structured_wraps_text(self) -> None:
        backend = OllamaBackend(model="llama3.1:8b")
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={"message": {"role": "assistant", "content": "Hello from Ollama"}},
        )

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await backend.generate_structured(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="Be concise.",
            )

        assert isinstance(result, CognitiveResponse)
        assert result.text == "Hello from Ollama"
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"
        assert result.has_tool_calls is False

    @pytest.mark.asyncio()
    async def test_generate_structured_ignores_tools(self) -> None:
        """Ollama generate_structured ignores tools param — no native support."""
        backend = OllamaBackend()
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={"message": {"role": "assistant", "content": "ok"}},
        )

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await backend.generate_structured(
                messages=[{"role": "user", "content": "test"}],
                tools=[{"name": "search", "description": "Search", "input_schema": {}}],
            )

        assert result.text == "ok"
        assert result.tool_calls == []


class TestForgeStructured:
    @pytest.mark.asyncio()
    async def test_generate_structured_wraps_text(self) -> None:
        backend = ForgeBackend(host="localhost", port=8000)
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={"response": "Hello from Forge"},
        )

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await backend.generate_structured(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="You are Animus.",
            )

        assert isinstance(result, CognitiveResponse)
        assert result.text == "Hello from Forge"
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"
        assert result.has_tool_calls is False

    @pytest.mark.asyncio()
    async def test_generate_structured_ignores_tools(self) -> None:
        """Forge generate_structured ignores tools param — no native support."""
        backend = ForgeBackend()
        mock_response = httpx.Response(
            200,
            request=_DUMMY_REQUEST,
            json={"text": "forge response"},
        )

        with patch("animus_bootstrap.gateway.cognitive.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await backend.generate_structured(
                messages=[{"role": "user", "content": "test"}],
                tools=[{"name": "t", "description": "T", "input_schema": {}}],
            )

        assert result.text == "forge response"
        assert result.tool_calls == []
