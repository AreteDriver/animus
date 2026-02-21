"""Tests for gateway middleware — auth, rate-limiting, and message logging."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from animus_bootstrap.gateway.middleware.auth import GatewayAuthMiddleware
from animus_bootstrap.gateway.middleware.logging import MessageLogger
from animus_bootstrap.gateway.middleware.ratelimit import RateLimiter
from animus_bootstrap.gateway.models import (
    GatewayMessage,
    GatewayResponse,
    create_message,
)

# ======================================================================
# Helpers
# ======================================================================


def _msg(
    channel: str = "telegram",
    sender_id: str = "user1",
    sender_name: str = "Alice",
    text: str = "hello",
) -> GatewayMessage:
    return create_message(
        channel=channel,
        sender_id=sender_id,
        sender_name=sender_name,
        text=text,
    )


# ======================================================================
# GatewayAuthMiddleware
# ======================================================================


class TestGatewayAuthMiddleware:
    """Auth middleware: open vs restricted mode."""

    def test_open_mode_by_default(self) -> None:
        auth = GatewayAuthMiddleware()
        assert auth.mode == "open"

    def test_open_mode_allows_all(self) -> None:
        auth = GatewayAuthMiddleware()
        assert auth.is_allowed(_msg(sender_id="anyone")) is True
        assert auth.is_allowed(_msg(channel="discord", sender_id="x")) is True

    def test_add_switches_to_restricted(self) -> None:
        auth = GatewayAuthMiddleware()
        auth.add_allowed("telegram", "user1")
        assert auth.mode == "restricted"

    def test_restricted_allows_listed_sender(self) -> None:
        auth = GatewayAuthMiddleware()
        auth.add_allowed("telegram", "user1")
        assert auth.is_allowed(_msg(channel="telegram", sender_id="user1")) is True

    def test_restricted_blocks_unlisted_sender(self) -> None:
        auth = GatewayAuthMiddleware()
        auth.add_allowed("telegram", "user1")
        assert auth.is_allowed(_msg(channel="telegram", sender_id="user2")) is False

    def test_restricted_blocks_wrong_channel(self) -> None:
        auth = GatewayAuthMiddleware()
        auth.add_allowed("telegram", "user1")
        assert auth.is_allowed(_msg(channel="discord", sender_id="user1")) is False

    def test_remove_returns_to_open(self) -> None:
        auth = GatewayAuthMiddleware()
        auth.add_allowed("telegram", "user1")
        assert auth.mode == "restricted"
        auth.remove_allowed("telegram", "user1")
        assert auth.mode == "open"
        assert auth.is_allowed(_msg(sender_id="anyone")) is True

    def test_remove_nonexistent_is_noop(self) -> None:
        auth = GatewayAuthMiddleware()
        auth.remove_allowed("telegram", "ghost")
        assert auth.mode == "open"

    def test_multiple_entries(self) -> None:
        auth = GatewayAuthMiddleware()
        auth.add_allowed("telegram", "alice")
        auth.add_allowed("discord", "bob")
        assert auth.is_allowed(_msg(channel="telegram", sender_id="alice")) is True
        assert auth.is_allowed(_msg(channel="discord", sender_id="bob")) is True
        assert auth.is_allowed(_msg(channel="telegram", sender_id="bob")) is False

    def test_allowlist_property_returns_copy(self) -> None:
        auth = GatewayAuthMiddleware()
        auth.add_allowed("telegram", "user1")
        snapshot = auth.allowlist
        auth.add_allowed("discord", "user2")
        assert len(snapshot) == 1  # original snapshot unchanged


# ======================================================================
# RateLimiter
# ======================================================================


class TestRateLimiter:
    """Token-bucket rate limiter."""

    def test_defaults(self) -> None:
        rl = RateLimiter()
        assert rl.max_tokens == 10
        assert rl.refill_rate == 1.0

    def test_custom_params(self) -> None:
        rl = RateLimiter(max_tokens=5, refill_rate=2.0)
        assert rl.max_tokens == 5
        assert rl.refill_rate == 2.0

    def test_allows_up_to_max_tokens(self) -> None:
        rl = RateLimiter(max_tokens=3, refill_rate=0.0)
        assert rl.check("u1") is True
        assert rl.check("u1") is True
        assert rl.check("u1") is True
        assert rl.check("u1") is False

    def test_independent_per_sender(self) -> None:
        rl = RateLimiter(max_tokens=1, refill_rate=0.0)
        assert rl.check("a") is True
        assert rl.check("b") is True
        assert rl.check("a") is False
        assert rl.check("b") is False

    def test_refill_restores_tokens(self) -> None:
        rl = RateLimiter(max_tokens=1, refill_rate=1000.0)
        assert rl.check("u1") is True
        assert rl.check("u1") is False
        # Simulate passage of time so refill fires
        time.sleep(0.01)
        assert rl.check("u1") is True

    def test_refill_does_not_exceed_max(self) -> None:
        rl = RateLimiter(max_tokens=2, refill_rate=10000.0)
        # Consume both tokens
        rl.check("u1")
        rl.check("u1")
        # Wait for generous refill
        time.sleep(0.01)
        # Should get exactly 2 back, not more
        assert rl.check("u1") is True
        assert rl.check("u1") is True
        assert rl.check("u1") is False

    def test_reset_restores_full_bucket(self) -> None:
        rl = RateLimiter(max_tokens=2, refill_rate=0.0)
        rl.check("u1")
        rl.check("u1")
        assert rl.check("u1") is False
        rl.reset("u1")
        assert rl.check("u1") is True
        assert rl.check("u1") is True

    def test_reset_nonexistent_is_noop(self) -> None:
        rl = RateLimiter()
        rl.reset("ghost")  # should not raise

    def test_thread_safety(self) -> None:
        rl = RateLimiter(max_tokens=100, refill_rate=0.0)
        results: list[bool] = []
        lock = threading.Lock()

        def drain() -> None:
            local: list[bool] = []
            for _ in range(20):
                local.append(rl.check("shared"))
            with lock:
                results.extend(local)

        threads = [threading.Thread(target=drain) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 100 True, 100 False
        assert results.count(True) == 100
        assert results.count(False) == 100


# ======================================================================
# MessageLogger
# ======================================================================


class TestMessageLogger:
    """SQLite-backed gateway message log."""

    @pytest.fixture()
    def msg_logger(self, tmp_path: Path) -> MessageLogger:
        ml = MessageLogger(tmp_path / "gateway_log.db")
        yield ml
        ml.close()

    def test_log_inbound(self, msg_logger: MessageLogger) -> None:
        msg = _msg(text="hi from user")
        msg_logger.log_inbound(msg)
        logs = msg_logger.get_logs()
        assert len(logs) == 1
        assert logs[0]["direction"] == "inbound"
        assert logs[0]["text"] == "hi from user"
        assert logs[0]["sender_id"] == "user1"
        assert logs[0]["channel"] == "telegram"

    def test_log_outbound(self, msg_logger: MessageLogger) -> None:
        resp = GatewayResponse(text="reply from animus", channel="telegram")
        msg_logger.log_outbound(resp, channel="telegram")
        logs = msg_logger.get_logs()
        assert len(logs) == 1
        assert logs[0]["direction"] == "outbound"
        assert logs[0]["text"] == "reply from animus"
        assert logs[0]["sender_id"] == "animus"
        assert logs[0]["sender_name"] == "Animus"

    def test_log_preserves_message_id(self, msg_logger: MessageLogger) -> None:
        msg = _msg()
        msg_logger.log_inbound(msg)
        logs = msg_logger.get_logs()
        assert logs[0]["id"] == msg.id

    def test_get_logs_filter_by_channel(self, msg_logger: MessageLogger) -> None:
        msg_logger.log_inbound(_msg(channel="telegram"))
        msg_logger.log_inbound(_msg(channel="discord"))
        msg_logger.log_inbound(_msg(channel="telegram"))

        telegram_logs = msg_logger.get_logs(channel="telegram")
        assert len(telegram_logs) == 2
        assert all(entry["channel"] == "telegram" for entry in telegram_logs)

        discord_logs = msg_logger.get_logs(channel="discord")
        assert len(discord_logs) == 1

    def test_get_logs_limit(self, msg_logger: MessageLogger) -> None:
        for i in range(20):
            msg_logger.log_inbound(_msg(text=f"msg {i}"))
        logs = msg_logger.get_logs(limit=5)
        assert len(logs) == 5

    def test_get_logs_ordered_newest_first(self, msg_logger: MessageLogger) -> None:
        msg_logger.log_inbound(
            create_message(
                channel="telegram",
                sender_id="u1",
                sender_name="A",
                text="first",
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            )
        )
        msg_logger.log_inbound(
            create_message(
                channel="telegram",
                sender_id="u1",
                sender_name="A",
                text="second",
                timestamp=datetime(2024, 6, 1, tzinfo=UTC),
            )
        )
        logs = msg_logger.get_logs()
        assert logs[0]["text"] == "second"
        assert logs[1]["text"] == "first"

    def test_get_logs_empty(self, msg_logger: MessageLogger) -> None:
        assert msg_logger.get_logs() == []

    def test_clear_all(self, msg_logger: MessageLogger) -> None:
        msg_logger.log_inbound(_msg())
        msg_logger.log_inbound(_msg())
        assert len(msg_logger.get_logs()) == 2
        msg_logger.clear_logs()
        assert len(msg_logger.get_logs()) == 0

    def test_clear_before_timestamp(self, msg_logger: MessageLogger) -> None:
        old = create_message(
            channel="telegram",
            sender_id="u1",
            sender_name="A",
            text="old",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        )
        recent = create_message(
            channel="telegram",
            sender_id="u1",
            sender_name="A",
            text="recent",
            timestamp=datetime(2025, 6, 1, tzinfo=UTC),
        )
        msg_logger.log_inbound(old)
        msg_logger.log_inbound(recent)

        msg_logger.clear_logs(before=datetime(2025, 1, 1, tzinfo=UTC))
        logs = msg_logger.get_logs()
        assert len(logs) == 1
        assert logs[0]["text"] == "recent"

    def test_mixed_inbound_outbound(self, msg_logger: MessageLogger) -> None:
        msg_logger.log_inbound(_msg(text="user says hi"))
        resp = GatewayResponse(text="bot replies", channel="telegram")
        msg_logger.log_outbound(resp, channel="telegram")

        logs = msg_logger.get_logs()
        assert len(logs) == 2
        directions = {entry["direction"] for entry in logs}
        assert directions == {"inbound", "outbound"}

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        ml = MessageLogger(tmp_path / "wal_test.db")
        cur = ml._conn.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        assert mode == "wal"
        ml.close()
