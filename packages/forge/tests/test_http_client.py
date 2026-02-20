"""Tests for HTTP client with connection pooling."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from animus_forge.http.client import (
    HTTPClientConfig,
    PooledHTTPClient,
    close_sync_client,
    get_async_client,
    get_pool_stats,
    get_sync_client,
)


class TestHTTPClientConfig:
    """Tests for HTTPClientConfig."""

    def test_default_values(self):
        """Config has reasonable defaults."""
        config = HTTPClientConfig()
        assert config.pool_connections == 10
        assert config.pool_maxsize == 20
        assert config.max_retries == 3
        assert config.timeout == 30.0
        assert 429 in config.retry_statuses

    def test_custom_values(self):
        """Config accepts custom values."""
        config = HTTPClientConfig(
            pool_connections=5,
            pool_maxsize=10,
            timeout=60.0,
        )
        assert config.pool_connections == 5
        assert config.pool_maxsize == 10
        assert config.timeout == 60.0


class TestSyncClient:
    """Tests for sync HTTP client."""

    def setup_method(self):
        """Reset client before each test."""
        close_sync_client()

    def teardown_method(self):
        """Clean up after each test."""
        close_sync_client()

    def test_get_sync_client_returns_session(self):
        """get_sync_client returns a requests.Session."""
        import requests

        client = get_sync_client()
        assert isinstance(client, requests.Session)

    def test_get_sync_client_is_singleton(self):
        """Same client instance is returned on multiple calls."""
        client1 = get_sync_client()
        client2 = get_sync_client()
        assert client1 is client2

    def test_close_sync_client(self):
        """close_sync_client releases the client."""
        client1 = get_sync_client()
        close_sync_client()
        client2 = get_sync_client()
        assert client1 is not client2

    def test_client_has_http_adapter(self):
        """Client has HTTP adapter mounted."""
        from requests.adapters import HTTPAdapter

        client = get_sync_client()
        adapter = client.get_adapter("https://")
        assert isinstance(adapter, HTTPAdapter)

    def test_client_has_retry_config(self):
        """Client adapter has retry configuration."""
        client = get_sync_client()
        adapter = client.get_adapter("https://")
        assert adapter.max_retries is not None


class TestAsyncClient:
    """Tests for async HTTP client."""

    @pytest.mark.asyncio
    async def test_get_async_client_context_manager(self):
        """get_async_client works as context manager."""
        import httpx

        async with get_async_client() as client:
            assert isinstance(client, httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_async_client_is_configured(self):
        """Async client is configured with limits."""
        async with get_async_client() as client:
            # Client should be an AsyncClient instance
            assert hasattr(client, "get")
            assert hasattr(client, "post")

    @pytest.mark.asyncio
    async def test_async_client_with_custom_config(self):
        """Async client accepts custom configuration."""
        config = HTTPClientConfig(
            pool_connections=5,
            pool_maxsize=15,
            timeout=10.0,
        )
        async with get_async_client(config=config) as client:
            # Verify client was created (can't easily check internal limits)
            assert client is not None


class TestPooledHTTPClient:
    """Tests for PooledHTTPClient context manager."""

    def test_context_manager_creates_session(self):
        """Context manager creates a session."""
        import requests

        with PooledHTTPClient() as client:
            assert isinstance(client, requests.Session)

    def test_context_manager_cleanup(self):
        """Context manager cleans up on exit."""
        pooled = PooledHTTPClient()
        with pooled as client:
            assert client is not None
        # After exit, session should be None
        assert pooled._session is None

    def test_custom_config(self):
        """Context manager accepts custom config."""
        config = HTTPClientConfig(timeout=5.0)
        with PooledHTTPClient(config=config) as client:
            assert client is not None


class TestPoolStats:
    """Tests for connection pool statistics."""

    def setup_method(self):
        """Reset client before each test."""
        close_sync_client()

    def teardown_method(self):
        """Clean up after each test."""
        close_sync_client()

    def test_stats_with_no_clients(self):
        """Stats show no active clients."""
        stats = get_pool_stats()
        assert stats["sync_client_active"] is False
        assert stats["async_client_active"] is False

    def test_stats_with_sync_client(self):
        """Stats show active sync client."""
        get_sync_client()
        stats = get_pool_stats()
        assert stats["sync_client_active"] is True


class TestIntegrationWithWebhook:
    """Integration tests with webhook delivery."""

    def setup_method(self):
        """Reset client before each test."""
        close_sync_client()

    def teardown_method(self):
        """Clean up after each test."""
        close_sync_client()

    def test_webhook_uses_pooled_client(self):
        """Webhook delivery uses the pooled client."""
        import os
        import tempfile

        from animus_forge.state.backends import SQLiteBackend
        from animus_forge.webhooks.webhook_delivery import (
            RetryStrategy,
            WebhookDeliveryManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            backend = SQLiteBackend(db_path=db_path)

            with patch("animus_forge.webhooks.webhook_delivery.get_database", return_value=backend):
                manager = WebhookDeliveryManager(
                    backend=backend,
                    retry_strategy=RetryStrategy(max_retries=0),
                )

                # Mock the pooled client
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.ok = True
                mock_response.text = '{"ok": true}'

                with patch("animus_forge.http.client._sync_client") as mock_client:
                    mock_client.post.return_value = mock_response

                    # Patch get_sync_client to return our mock
                    with patch(
                        "animus_forge.webhooks.webhook_delivery.get_sync_client",
                        return_value=mock_client,
                    ):
                        manager.deliver(
                            url="https://example.com/webhook",
                            payload={"event": "test"},
                        )

                        # Verify pooled client was used
                        mock_client.post.assert_called_once()

            backend.close()
