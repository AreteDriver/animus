"""Tests for gateway middleware wiring (auth allowlist, rate limit, logging)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from animus_bootstrap.gateway.middleware.auth import GatewayAuthMiddleware
from animus_bootstrap.gateway.middleware.logging import MessageLogger
from animus_bootstrap.gateway.middleware.ratelimit import RateLimiter
from animus_bootstrap.gateway.models import create_message
from animus_bootstrap.gateway.router import MessageRouter
from animus_bootstrap.gateway.session import SessionManager

pytestmark = pytest.mark.asyncio


@pytest.fixture()
def session_manager(tmp_path) -> Iterator[SessionManager]:  # type: ignore[no-untyped-def]
    manager = SessionManager(tmp_path / "sessions.db")
    try:
        yield manager
    finally:
        manager.close()


def _cognitive(text: str = "hi there"):  # type: ignore[no-untyped-def]
    from unittest.mock import AsyncMock

    cog = AsyncMock()
    cog.generate_response = AsyncMock(return_value=text)
    return cog


def _msg(text: str = "hello", channel: str = "webchat", sender: str = "user1"):  # type: ignore[no-untyped-def]
    return create_message(channel, sender, "Alice", text)


class TestNoMiddleware:
    async def test_open_by_default(self, session_manager: SessionManager) -> None:
        router = MessageRouter(_cognitive(), session_manager)
        resp = await router.handle_message(_msg())
        assert resp.text == "hi there"


class TestAuthAllowlist:
    async def test_blocks_unlisted_sender(self, session_manager: SessionManager) -> None:
        auth = GatewayAuthMiddleware()
        auth.add_allowed("webchat", "allowed-user")
        router = MessageRouter(_cognitive(), session_manager, auth=auth)

        resp = await router.handle_message(_msg(sender="intruder"))
        assert "not authorized" in resp.text.lower()

    async def test_allows_listed_sender(self, session_manager: SessionManager) -> None:
        auth = GatewayAuthMiddleware()
        auth.add_allowed("webchat", "user1")
        router = MessageRouter(_cognitive(), session_manager, auth=auth)

        resp = await router.handle_message(_msg(sender="user1"))
        assert resp.text == "hi there"


class TestRateLimit:
    async def test_rate_limit_exhaustion(self, session_manager: SessionManager) -> None:
        limiter = RateLimiter(max_tokens=1, refill_rate=0.0)
        router = MessageRouter(_cognitive(), session_manager, rate_limiter=limiter)

        first = await router.handle_message(_msg())
        assert first.text == "hi there"
        second = await router.handle_message(_msg())
        assert "too quickly" in second.text.lower()


class TestMessageLogging:
    async def test_inbound_and_outbound_logged(
        self, session_manager: SessionManager, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        msg_logger = MessageLogger(tmp_path / "gateway_log.db")
        try:
            router = MessageRouter(_cognitive(), session_manager, message_logger=msg_logger)
            await router.handle_message(_msg())
            logs = msg_logger.get_logs()
            directions = {row["direction"] for row in logs}
            assert "inbound" in directions
            assert "outbound" in directions
        finally:
            msg_logger.close()

    async def test_logger_errors_are_swallowed(self, session_manager: SessionManager) -> None:
        from unittest.mock import MagicMock

        bad_logger = MagicMock()
        bad_logger.log_inbound.side_effect = RuntimeError("db down")
        bad_logger.log_outbound.side_effect = RuntimeError("db down")
        router = MessageRouter(_cognitive(), session_manager, message_logger=bad_logger)

        # A failing logger must not break message handling.
        resp = await router.handle_message(_msg())
        assert resp.text == "hi there"
        bad_logger.log_inbound.assert_called_once()
        bad_logger.log_outbound.assert_called_once()
