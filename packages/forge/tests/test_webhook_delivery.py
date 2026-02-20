"""Tests for WebhookDeliveryManager with retries and DLQ."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from animus_forge.state.backends import SQLiteBackend
from animus_forge.webhooks.webhook_delivery import (
    DeliveryStatus,
    RetryStrategy,
    WebhookDelivery,
    WebhookDeliveryManager,
)


class TestRetryStrategy:
    """Tests for RetryStrategy."""

    def test_default_values(self):
        """Default strategy has reasonable defaults."""
        strategy = RetryStrategy()
        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0
        assert strategy.max_delay == 60.0

    def test_exponential_delay(self):
        """Delays grow exponentially."""
        strategy = RetryStrategy(base_delay=1.0, exponential_base=2.0, jitter=False)
        assert strategy.get_delay(0) == 1.0
        assert strategy.get_delay(1) == 2.0
        assert strategy.get_delay(2) == 4.0
        assert strategy.get_delay(3) == 8.0

    def test_max_delay_cap(self):
        """Delay is capped at max_delay."""
        strategy = RetryStrategy(base_delay=1.0, max_delay=5.0, jitter=False)
        assert strategy.get_delay(10) == 5.0

    def test_jitter_varies_delay(self):
        """Jitter adds variation to delay."""
        strategy = RetryStrategy(base_delay=1.0, jitter=True)
        delays = [strategy.get_delay(1) for _ in range(10)]
        # With jitter, delays should vary
        assert len(set(delays)) > 1


class TestWebhookDeliveryManager:
    """Tests for WebhookDeliveryManager."""

    @pytest.fixture
    def backend(self):
        """Create a temporary SQLite backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            backend = SQLiteBackend(db_path=db_path)
            yield backend
            backend.close()

    @pytest.fixture
    def manager(self, backend):
        """Create a WebhookDeliveryManager."""
        with patch("animus_forge.webhooks.webhook_delivery.get_database", return_value=backend):
            manager = WebhookDeliveryManager(
                backend=backend,
                retry_strategy=RetryStrategy(max_retries=2, base_delay=0.01, jitter=False),
                timeout=1.0,
            )
            yield manager

    def test_init_creates_schema(self, backend):
        """Manager creates tables on init."""
        with patch("animus_forge.webhooks.webhook_delivery.get_database", return_value=backend):
            WebhookDeliveryManager(backend=backend)

        # Check tables exist
        tables = backend.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [t["name"] for t in tables]
        assert "webhook_deliveries" in table_names
        assert "webhook_dead_letter" in table_names

    def test_successful_delivery(self, manager):
        """Successful delivery is recorded correctly."""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.text = '{"ok": true}'

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=mock_client,
        ):
            delivery = manager.deliver(
                url="https://example.com/webhook",
                payload={"event": "test"},
            )

        assert delivery.status == DeliveryStatus.SUCCESS
        assert delivery.attempt_count == 1
        assert delivery.completed_at is not None
        assert delivery.response_status == 200

    def test_retry_on_failure(self, manager):
        """Failed delivery is retried."""
        import requests

        call_count = [0]
        mock_client = MagicMock()

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise requests.exceptions.ConnectionError("Connection refused")
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.ok = True
            mock_response.text = '{"ok": true}'
            return mock_response

        mock_client.post.side_effect = side_effect

        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=mock_client,
        ):
            delivery = manager.deliver(
                url="https://example.com/webhook",
                payload={"event": "test"},
            )

        assert delivery.status == DeliveryStatus.SUCCESS
        assert delivery.attempt_count == 2  # First failed, second succeeded

    def test_dlq_on_max_retries(self, manager):
        """Failed delivery after max retries goes to DLQ."""
        import requests

        mock_client = MagicMock()
        mock_client.post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=mock_client,
        ):
            delivery = manager.deliver(
                url="https://example.com/webhook",
                payload={"event": "test"},
                max_retries=2,
            )

        assert delivery.status == DeliveryStatus.DEAD_LETTER
        assert delivery.attempt_count == 2  # max_retries=2 means 2 total attempts

        # Check DLQ
        dlq_items = manager.get_dlq_items()
        assert len(dlq_items) == 1
        assert dlq_items[0]["webhook_url"] == "https://example.com/webhook"

    def test_signature_generation(self, manager):
        """Webhook signature is generated correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.text = '{"ok": true}'

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=mock_client,
        ):
            manager.deliver(
                url="https://example.com/webhook",
                payload={"event": "test"},
                secret="my-secret",
            )

        # Check signature header was added
        call_args = mock_client.post.call_args
        headers = call_args.kwargs.get("headers", {})
        assert "X-Webhook-Signature" in headers
        assert headers["X-Webhook-Signature"].startswith("sha256=")

    def test_reprocess_dlq_item(self, manager):
        """DLQ items can be reprocessed."""
        import requests

        mock_client = MagicMock()

        # First, create a failed delivery
        mock_client.post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=mock_client,
        ):
            manager.deliver(
                url="https://example.com/webhook",
                payload={"event": "test"},
                max_retries=0,
            )

        dlq_items = manager.get_dlq_items()
        assert len(dlq_items) == 1

        # Now reprocess with success
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.text = '{"ok": true}'
        mock_client.post.side_effect = None
        mock_client.post.return_value = mock_response

        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=mock_client,
        ):
            delivery = manager.reprocess_dlq_item(dlq_items[0]["id"])

        assert delivery.status == DeliveryStatus.SUCCESS

        # DLQ item should be marked as reprocessed
        remaining = manager.get_dlq_items()
        assert len(remaining) == 0

    def test_delivery_stats(self, manager):
        """Delivery statistics are tracked correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.text = '{"ok": true}'

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=mock_client,
        ):
            # Create 2 successful deliveries
            manager.deliver(url="https://example.com/1", payload={"n": 1})
            manager.deliver(url="https://example.com/2", payload={"n": 2})

        stats = manager.get_delivery_stats()
        assert stats["success_count"] == 2
        assert stats["pending_count"] == 0


class TestAsyncDelivery:
    """Tests for async webhook delivery."""

    @pytest.fixture
    def backend(self):
        """Create a temporary SQLite backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            backend = SQLiteBackend(db_path=db_path)
            yield backend
            backend.close()

    @pytest.fixture
    def manager(self, backend):
        """Create a WebhookDeliveryManager."""
        with patch("animus_forge.webhooks.webhook_delivery.get_database", return_value=backend):
            manager = WebhookDeliveryManager(
                backend=backend,
                retry_strategy=RetryStrategy(max_retries=1, base_delay=0.01, jitter=False),
                timeout=1.0,
            )
            yield manager

    @pytest.mark.asyncio
    async def test_async_successful_delivery(self, manager):
        """Async delivery works correctly."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.text = '{"ok": true}'

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)

        with patch("animus_forge.webhooks.webhook_delivery.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client.return_value.__aexit__.return_value = None

            delivery = await manager.deliver_async(
                url="https://example.com/webhook",
                payload={"event": "test"},
            )

        assert delivery.status == DeliveryStatus.SUCCESS
        assert delivery.response_status == 200

    @pytest.mark.asyncio
    async def test_async_dlq_on_failure(self, manager):
        """Async failed delivery goes to DLQ."""
        from unittest.mock import AsyncMock

        import httpx

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(side_effect=httpx.RequestError("Connection failed"))

        with patch("animus_forge.webhooks.webhook_delivery.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client.return_value.__aexit__.return_value = None

            delivery = await manager.deliver_async(
                url="https://example.com/webhook",
                payload={"event": "test"},
                max_retries=0,
            )

        assert delivery.status == DeliveryStatus.DEAD_LETTER

        dlq_items = manager.get_dlq_items()
        assert len(dlq_items) == 1


class TestDeliveryStatus:
    """Tests for DeliveryStatus enum."""

    def test_all_statuses_defined(self):
        """All expected statuses exist."""
        assert DeliveryStatus.PENDING.value == "pending"
        assert DeliveryStatus.SUCCESS.value == "success"
        assert DeliveryStatus.FAILED.value == "failed"
        assert DeliveryStatus.RETRYING.value == "retrying"
        assert DeliveryStatus.DEAD_LETTER.value == "dead_letter"


class TestWebhookDeliveryModel:
    """Tests for WebhookDelivery model."""

    def test_default_values(self):
        """Delivery model has correct defaults."""
        delivery = WebhookDelivery(
            webhook_url="https://example.com",
            payload={"test": True},
        )
        assert delivery.status == DeliveryStatus.PENDING
        assert delivery.attempt_count == 0
        assert delivery.max_retries == 3
        assert delivery.headers == {}
        assert delivery.created_at is not None

    def test_custom_values(self):
        """Delivery model accepts custom values."""
        delivery = WebhookDelivery(
            webhook_url="https://example.com",
            payload={"test": True},
            headers={"Authorization": "Bearer token"},
            max_retries=5,
        )
        assert delivery.max_retries == 5
        assert delivery.headers["Authorization"] == "Bearer token"
