"""Targeted coverage for webhook_delivery.deliver_async branches on 2c8cde2.

Covers the async delivery path's uncovered branches:
- 437-443: circuit breaker open → DLQ short-circuit
- 448: X-Webhook-Signature header computed when secret provided
- 476: delivery.last_error = "HTTP <code>" on non-2xx response
- 479-480: TimeoutException branch
- 496-499: retry with sleep when attempt_count < max_retries
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, "src")

from animus_forge.state.backends import SQLiteBackend
from animus_forge.webhooks.webhook_delivery import (
    CircuitBreaker,
    CircuitBreakerConfig,
    DeliveryStatus,
    RetryStrategy,
    WebhookDeliveryManager,
)


@pytest.fixture
def backend():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        b = SQLiteBackend(db_path=db_path)
        yield b
        b.close()


@pytest.fixture
def manager(backend):
    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, recovery_timeout=0.1))
    with patch("animus_forge.webhooks.webhook_delivery.get_database", return_value=backend):
        yield WebhookDeliveryManager(
            backend=backend,
            retry_strategy=RetryStrategy(max_retries=2, base_delay=0.001, jitter=False),
            timeout=1.0,
            circuit_breaker=cb,
        )


def _mock_client_ctx(post_mock):
    """Build an async-context-manager mock whose .post returns `post_mock`."""
    client = AsyncMock()
    client.post = post_mock
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm, client


class TestDeliverAsyncCircuitBreaker:
    """Lines 437-443 — circuit-breaker-open shortcut."""

    @pytest.mark.asyncio
    async def test_open_circuit_goes_straight_to_dlq(self, manager):
        # Force the breaker open.
        manager.circuit_breaker.allow_request = MagicMock(return_value=False)

        # Patch get_async_client so the post path is never reached (would fail).
        with patch("animus_forge.webhooks.webhook_delivery.get_async_client") as mock_get:
            mock_get.side_effect = AssertionError("client should not be built when circuit is open")
            delivery = await manager.deliver_async(
                "https://example.invalid/hook", {"event": "ping"}
            )

        assert delivery.status == DeliveryStatus.CIRCUIT_BROKEN
        assert delivery.last_error == "Circuit breaker open"
        assert delivery.completed_at is not None


class TestDeliverAsyncSignature:
    """Line 448 — signature header written when secret is provided."""

    @pytest.mark.asyncio
    async def test_signature_header_added_when_secret_provided(self, manager):
        response = MagicMock()
        response.status_code = 200
        response.is_success = True
        response.text = "ok"

        post = AsyncMock(return_value=response)
        cm, client = _mock_client_ctx(post)

        with patch(
            "animus_forge.webhooks.webhook_delivery.get_async_client",
            return_value=cm,
        ):
            delivery = await manager.deliver_async(
                "https://example.invalid/hook",
                {"event": "ping"},
                secret="topsecret",
            )

        assert delivery.status == DeliveryStatus.SUCCESS
        # Check the headers dict sent to client.post included a signature.
        call_kwargs = post.await_args.kwargs
        headers = call_kwargs["headers"]
        assert "X-Webhook-Signature" in headers
        assert headers["Content-Type"] == "application/json"


class TestDeliverAsyncRetry:
    """Lines 476 + 496-499 — non-2xx response populates last_error and
    schedules a retry with asyncio.sleep."""

    @pytest.mark.asyncio
    async def test_http_error_then_success_reschedules(self, manager):
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        fail_resp.is_success = False
        fail_resp.text = "boom"

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.is_success = True
        ok_resp.text = "ok"

        post = AsyncMock(side_effect=[fail_resp, ok_resp])
        cm, _ = _mock_client_ctx(post)

        # Patch asyncio.sleep so the retry is instant.
        with (
            patch(
                "animus_forge.webhooks.webhook_delivery.get_async_client",
                return_value=cm,
            ),
            patch(
                "animus_forge.webhooks.webhook_delivery.asyncio.sleep",
                new=AsyncMock(return_value=None),
            ),
        ):
            delivery = await manager.deliver_async(
                "https://example.invalid/hook", {"event": "ping"}
            )

        assert delivery.status == DeliveryStatus.SUCCESS
        assert delivery.attempt_count == 2
        # First attempt set last_error to HTTP 503; success cleared no error
        # but the response_status records the last call's status.
        assert delivery.response_status == 200
        assert post.await_count == 2


class TestDeliverAsyncTimeout:
    """Lines 478-480 — httpx.TimeoutException is caught and reported."""

    @pytest.mark.asyncio
    async def test_timeout_becomes_request_timeout_error(self, manager):
        # Force all retries to timeout so the final status is DEAD_LETTER.
        post = AsyncMock(side_effect=httpx.TimeoutException("slow"))
        cm, _ = _mock_client_ctx(post)

        with (
            patch(
                "animus_forge.webhooks.webhook_delivery.get_async_client",
                return_value=cm,
            ),
            patch(
                "animus_forge.webhooks.webhook_delivery.asyncio.sleep",
                new=AsyncMock(return_value=None),
            ),
        ):
            delivery = await manager.deliver_async(
                "https://example.invalid/hook", {"event": "ping"}
            )

        assert delivery.last_error == "Request timeout"
        # Exhausted retries → DLQ.
        assert delivery.status == DeliveryStatus.DEAD_LETTER
