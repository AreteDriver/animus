"""Tests for circuit breaker, DLQ batch management, and DLQ API endpoints."""

import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from animus_forge.state.backends import SQLiteBackend
from animus_forge.webhooks.webhook_delivery import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    DeliveryStatus,
    RetryStrategy,
    WebhookDeliveryManager,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend():
    """Create a temporary SQLite backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        b = SQLiteBackend(db_path=db_path)
        yield b
        b.close()


@pytest.fixture
def manager(backend):
    """Create a WebhookDeliveryManager with fast retries and low circuit threshold."""
    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, recovery_timeout=0.1))
    with patch("animus_forge.webhooks.webhook_delivery.get_database", return_value=backend):
        mgr = WebhookDeliveryManager(
            backend=backend,
            retry_strategy=RetryStrategy(max_retries=1, base_delay=0.001, jitter=False),
            timeout=1.0,
            circuit_breaker=cb,
        )
        yield mgr


def _mock_client_fail():
    """Return a mock HTTP client that always raises ConnectionError."""
    import requests

    mock = MagicMock()
    mock.post.side_effect = requests.exceptions.ConnectionError("refused")
    return mock


def _mock_client_ok():
    """Return a mock HTTP client that succeeds with 200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.text = '{"ok":true}'
    mock = MagicMock()
    mock.post.return_value = mock_resp
    return mock


# ===================================================================
# Circuit Breaker unit tests
# ===================================================================


class TestCircuitBreaker:
    """Test CircuitBreaker state transitions."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.get_state("https://example.com") == "closed"

    def test_allow_request_when_closed(self):
        cb = CircuitBreaker()
        assert cb.allow_request("https://example.com") is True

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        url = "https://example.com"
        cb.record_failure(url)
        cb.record_failure(url)
        assert cb.get_state(url) == "closed"
        assert cb.allow_request(url) is True

    def test_trips_to_open_at_threshold(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        url = "https://example.com"
        for _ in range(3):
            cb.record_failure(url)
        assert cb.get_state(url) == "open"
        assert cb.allow_request(url) is False

    def test_open_to_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.05))
        url = "https://example.com"
        cb.record_failure(url)
        assert cb.get_state(url) == "open"
        assert cb.allow_request(url) is False

        # Wait for recovery timeout
        time.sleep(0.06)
        assert cb.allow_request(url) is True
        assert cb.get_state(url) == "half_open"

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.01))
        url = "https://example.com"
        cb.record_failure(url)
        time.sleep(0.02)
        cb.allow_request(url)  # transitions to half_open
        assert cb.get_state(url) == "half_open"

        cb.record_success(url)
        assert cb.get_state(url) == "closed"
        assert cb.allow_request(url) is True

    def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.01))
        url = "https://example.com"
        cb.record_failure(url)
        time.sleep(0.02)
        cb.allow_request(url)  # transitions to half_open

        cb.record_failure(url)
        assert cb.get_state(url) == "open"

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        url = "https://example.com"
        cb.record_failure(url)
        cb.record_failure(url)
        cb.record_success(url)
        # Counter reset — need 3 more to trip
        cb.record_failure(url)
        cb.record_failure(url)
        assert cb.get_state(url) == "closed"

    def test_per_url_isolation(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))
        cb.record_failure("https://a.com")
        cb.record_failure("https://a.com")
        assert cb.get_state("https://a.com") == "open"
        assert cb.get_state("https://b.com") == "closed"

    def test_get_all_states(self):
        cb = CircuitBreaker()
        cb.record_failure("https://a.com")
        cb.record_success("https://b.com")
        states = cb.get_all_states()
        url_a = "https://a.com"
        url_b = "https://b.com"
        assert url_a in states
        assert url_b in states
        assert states[url_a]["failures"] == 1
        assert states[url_b]["failures"] == 0

    def test_reset(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        url = "https://example.com"
        cb.record_failure(url)
        assert cb.get_state(url) == "open"
        cb.reset(url)
        assert cb.get_state(url) == "closed"

    def test_reset_nonexistent_url(self):
        cb = CircuitBreaker()
        cb.reset("https://nonexistent.com")  # should not raise


class TestCircuitBreakerState:
    """Test the CircuitBreakerState dataclass."""

    def test_defaults(self):
        s = CircuitBreakerState()
        assert s.failures == 0
        assert s.state == "closed"
        assert s.last_failure_at is None


class TestCircuitBreakerConfig:
    """Test the CircuitBreakerConfig dataclass."""

    def test_defaults(self):
        c = CircuitBreakerConfig()
        assert c.failure_threshold == 5
        assert c.recovery_timeout == 300.0

    def test_custom(self):
        c = CircuitBreakerConfig(failure_threshold=10, recovery_timeout=60.0)
        assert c.failure_threshold == 10
        assert c.recovery_timeout == 60.0


# ===================================================================
# Circuit breaker integration with delivery
# ===================================================================


class TestCircuitBreakerDelivery:
    """Test circuit breaker integration in WebhookDeliveryManager.deliver()."""

    def test_circuit_broken_goes_to_dlq(self, manager):
        """When circuit is open, delivery goes straight to DLQ."""
        url = "https://fail.example.com"
        # Force the circuit open
        for _ in range(3):
            manager.circuit_breaker.record_failure(url)
        assert manager.circuit_breaker.get_state(url) == "open"

        delivery = manager.deliver(url=url, payload={"test": True})
        assert delivery.status == DeliveryStatus.CIRCUIT_BROKEN
        assert delivery.last_error == "Circuit breaker open"
        assert delivery.completed_at is not None

        dlq = manager.get_dlq_items()
        assert len(dlq) == 1
        assert dlq[0]["webhook_url"] == url

    def test_success_resets_circuit(self, manager):
        """Successful delivery resets the circuit breaker."""
        url = "https://ok.example.com"
        manager.circuit_breaker.record_failure(url)
        manager.circuit_breaker.record_failure(url)

        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=_mock_client_ok(),
        ):
            delivery = manager.deliver(url=url, payload={"e": 1})

        assert delivery.status == DeliveryStatus.SUCCESS
        assert manager.circuit_breaker.get_state(url) == "closed"

    def test_failed_delivery_increments_circuit(self, manager):
        """Failed delivery (after retries) increments circuit breaker."""
        url = "https://flaky.example.com"
        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=_mock_client_fail(),
        ):
            manager.deliver(url=url, payload={"e": 1})

        # 1 failure recorded
        state = manager.circuit_breaker._states[url]
        assert state.failures == 1

    def test_circuit_trips_after_repeated_failures(self, manager):
        """Circuit trips after threshold (3) failed delivery rounds."""
        url = "https://dead.example.com"
        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=_mock_client_fail(),
        ):
            for _ in range(3):
                manager.deliver(url=url, payload={"e": 1})

        assert manager.circuit_breaker.get_state(url) == "open"

        # Next attempt should be circuit-broken, no HTTP call
        delivery = manager.deliver(url=url, payload={"e": 1})
        assert delivery.status == DeliveryStatus.CIRCUIT_BROKEN


# ===================================================================
# DLQ batch management
# ===================================================================


class TestDLQBatchManagement:
    """Test reprocess_all_dlq, purge_dlq, get_dlq_stats, delete_dlq_item."""

    def _populate_dlq(self, manager, count=3):
        """Helper: create *count* DLQ items."""
        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=_mock_client_fail(),
        ):
            for i in range(count):
                manager.deliver(
                    url=f"https://fail{i}.example.com",
                    payload={"n": i},
                    max_retries=0,
                )

    def test_reprocess_all_dlq_success(self, manager):
        """reprocess_all_dlq retries every pending item."""
        self._populate_dlq(manager, 3)
        assert len(manager.get_dlq_items()) == 3

        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=_mock_client_ok(),
        ):
            results = manager.reprocess_all_dlq(max_items=50)

        assert len(results) == 3
        for r in results:
            assert r["status"] == "success"

        # Original items marked as reprocessed
        assert len(manager.get_dlq_items()) == 0

    def test_reprocess_all_dlq_partial_failure(self, manager):
        """reprocess_all_dlq continues if one item fails."""
        self._populate_dlq(manager, 2)

        call_count = [0]
        ok_client = _mock_client_ok()
        fail_client = _mock_client_fail()

        def alternating(*_a, **_k):
            call_count[0] += 1
            if call_count[0] <= 1:
                return ok_client
            return fail_client

        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            side_effect=alternating,
        ):
            results = manager.reprocess_all_dlq()

        assert len(results) == 2
        statuses = {r["status"] for r in results}
        assert "success" in statuses

    def test_reprocess_all_dlq_empty(self, manager):
        """reprocess_all_dlq returns empty list when no items."""
        results = manager.reprocess_all_dlq()
        assert results == []

    def test_get_dlq_stats(self, manager):
        """get_dlq_stats returns counts by URL and oldest age."""
        self._populate_dlq(manager, 3)

        stats = manager.get_dlq_stats()
        assert stats["total_pending"] == 3
        assert isinstance(stats["by_url"], dict)
        assert len(stats["by_url"]) == 3
        assert stats["oldest_age_seconds"] is not None
        assert stats["oldest_age_seconds"] >= 0

    def test_get_dlq_stats_empty(self, manager):
        """get_dlq_stats returns zeros when empty."""
        stats = manager.get_dlq_stats()
        assert stats["total_pending"] == 0
        assert stats["by_url"] == {}
        assert stats["oldest_age_seconds"] is None

    def test_purge_dlq(self, manager, backend):
        """purge_dlq removes items older than threshold."""
        self._populate_dlq(manager, 2)
        assert len(manager.get_dlq_items()) == 2

        # Backdate items so they're "old"
        with backend.transaction():
            backend.execute(
                "UPDATE webhook_dead_letter SET created_at = ?",
                ((datetime.now() - timedelta(days=60)).isoformat(),),
            )

        deleted = manager.purge_dlq(older_than_days=30)
        assert deleted == 2
        assert len(manager.get_dlq_items()) == 0

    def test_purge_dlq_keeps_recent(self, manager):
        """purge_dlq keeps items newer than threshold."""
        self._populate_dlq(manager, 2)
        deleted = manager.purge_dlq(older_than_days=30)
        assert deleted == 0
        assert len(manager.get_dlq_items()) == 2

    def test_delete_dlq_item(self, manager):
        """delete_dlq_item removes a single item."""
        self._populate_dlq(manager, 2)
        items = manager.get_dlq_items()
        assert len(items) == 2

        result = manager.delete_dlq_item(items[0]["id"])
        assert result is True
        assert len(manager.get_dlq_items()) == 1

    def test_delete_dlq_item_not_found(self, manager):
        """delete_dlq_item returns False for missing ID."""
        assert manager.delete_dlq_item(999) is False


# ===================================================================
# DLQ API endpoints
# ===================================================================


class TestDLQAPIEndpoints:
    """Test the DLQ-related API routes in /v1/webhooks/dlq."""

    @pytest.fixture
    def api_client(self, backend):
        """Create a FastAPI test client with mocked state."""
        from fastapi.testclient import TestClient

        import animus_forge.api as api_module
        import animus_forge.api_state as api_state

        api_state.schedule_manager = MagicMock()
        api_state.webhook_manager = MagicMock()
        api_state.job_manager = MagicMock()
        api_state.version_manager = MagicMock()

        # Create a real delivery manager backed by the test SQLite
        with patch("animus_forge.webhooks.webhook_delivery.get_database", return_value=backend):
            api_state.delivery_manager = WebhookDeliveryManager(
                backend=backend,
                retry_strategy=RetryStrategy(max_retries=0, base_delay=0.001, jitter=False),
                timeout=1.0,
            )

        api_state._app_state["ready"] = True
        api_state._app_state["shutting_down"] = False
        api_state._app_state["start_time"] = datetime.now()

        client = TestClient(api_module.app, raise_server_exceptions=False)
        yield client

        # Cleanup shared mutable state
        api_state.delivery_manager = None

    @pytest.fixture
    def auth_header(self):
        from animus_forge.auth import create_access_token

        token = create_access_token("testuser")
        return {"Authorization": f"Bearer {token}"}

    def _add_dlq_items(self, api_client, backend, count=2):
        """Insert DLQ items via the delivery manager."""
        import animus_forge.api_state as api_state

        dm = api_state.delivery_manager
        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=_mock_client_fail(),
        ):
            for i in range(count):
                dm.deliver(
                    url=f"https://fail{i}.example.com",
                    payload={"i": i},
                    max_retries=0,
                )

    def test_list_dlq_items(self, api_client, auth_header, backend):
        self._add_dlq_items(api_client, backend)
        r = api_client.get("/v1/webhooks/dlq", headers=auth_header)
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2

    def test_list_dlq_items_unauthorized(self, api_client):
        r = api_client.get("/v1/webhooks/dlq")
        assert r.status_code == 401

    def test_get_dlq_stats(self, api_client, auth_header, backend):
        self._add_dlq_items(api_client, backend, 3)
        r = api_client.get("/v1/webhooks/dlq/stats", headers=auth_header)
        assert r.status_code == 200
        data = r.json()
        assert data["total_pending"] == 3
        assert isinstance(data["by_url"], dict)

    def test_retry_dlq_item(self, api_client, auth_header, backend):
        self._add_dlq_items(api_client, backend, 1)
        import animus_forge.api_state as api_state

        items = api_state.delivery_manager.get_dlq_items()
        dlq_id = items[0]["id"]

        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=_mock_client_ok(),
        ):
            r = api_client.post(f"/v1/webhooks/dlq/{dlq_id}/retry", headers=auth_header)

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["delivery_status"] == "success"

    def test_retry_dlq_item_not_found(self, api_client, auth_header):
        r = api_client.post("/v1/webhooks/dlq/99999/retry", headers=auth_header)
        assert r.status_code == 404

    def test_retry_all_dlq(self, api_client, auth_header, backend):
        self._add_dlq_items(api_client, backend, 2)
        with patch(
            "animus_forge.webhooks.webhook_delivery.get_sync_client",
            return_value=_mock_client_ok(),
        ):
            r = api_client.post("/v1/webhooks/dlq/retry-all", headers=auth_header)

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["processed"] == 2

    def test_delete_dlq_item(self, api_client, auth_header, backend):
        self._add_dlq_items(api_client, backend, 1)
        import animus_forge.api_state as api_state

        items = api_state.delivery_manager.get_dlq_items()
        dlq_id = items[0]["id"]

        r = api_client.delete(f"/v1/webhooks/dlq/{dlq_id}", headers=auth_header)
        assert r.status_code == 200
        assert r.json()["status"] == "success"

        # Verify it's gone
        assert len(api_state.delivery_manager.get_dlq_items()) == 0

    def test_delete_dlq_item_not_found(self, api_client, auth_header):
        r = api_client.delete("/v1/webhooks/dlq/99999", headers=auth_header)
        assert r.status_code == 404


# ===================================================================
# DeliveryStatus enum — verify new CIRCUIT_BROKEN value
# ===================================================================


class TestCircuitBrokenStatus:
    """Ensure the new CIRCUIT_BROKEN status is properly defined."""

    def test_circuit_broken_value(self):
        assert DeliveryStatus.CIRCUIT_BROKEN.value == "circuit_broken"

    def test_all_statuses_count(self):
        # 6 total: pending, success, failed, retrying, dead_letter, circuit_broken
        assert len(DeliveryStatus) == 6
