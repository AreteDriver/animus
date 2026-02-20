"""Tests for webhooks/webhook_manager.py and webhooks/webhook_delivery.py coverage."""

import hashlib
import hmac
import sys
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")

from animus_forge.webhooks.webhook_delivery import (
    DeliveryStatus,
    RetryStrategy,
    WebhookDeliveryManager,
)
from animus_forge.webhooks.webhook_manager import (
    PayloadMapping,
    Webhook,
    WebhookManager,
    WebhookStatus,
)


def _mock_backend():
    """Create a mock database backend with all required methods."""
    backend = MagicMock()
    backend.fetchone.return_value = None
    backend.fetchall.return_value = []
    backend.execute.return_value = None
    backend.executescript.return_value = None

    @contextmanager
    def _txn():
        yield

    backend.transaction = _txn
    return backend


def _make_webhook_manager(backend=None):
    """Construct a WebhookManager with mocked dependencies."""
    backend = backend or _mock_backend()
    with (
        patch(
            "animus_forge.webhooks.webhook_manager.get_settings",
            return_value=MagicMock(),
        ),
        patch(
            "animus_forge.webhooks.webhook_manager.get_database",
            return_value=backend,
        ),
        patch(
            "animus_forge.webhooks.webhook_manager.WorkflowEngineAdapter",
            return_value=MagicMock(),
        ),
    ):
        mgr = WebhookManager(backend=backend)
    return mgr


def _make_webhook(
    wid="wh-1",
    workflow_id="wf-1",
    secret="test-secret-key",
    status=WebhookStatus.ACTIVE,
):
    """Create a Webhook for testing."""
    return Webhook(
        id=wid,
        name="Test Webhook",
        workflow_id=workflow_id,
        secret=secret,
        status=status,
    )


def _make_delivery_manager(backend=None):
    """Construct a WebhookDeliveryManager with mocked dependencies."""
    backend = backend or _mock_backend()
    with patch(
        "animus_forge.webhooks.webhook_delivery.get_database",
        return_value=backend,
    ):
        mgr = WebhookDeliveryManager(
            backend=backend,
            retry_strategy=RetryStrategy(max_retries=2, jitter=False),
        )
    return mgr


# ========== WebhookManager Tests ==========


class TestWebhookCRUD:
    def test_create_webhook_success(self):
        mgr = _make_webhook_manager()
        mgr.workflow_engine.load_workflow.return_value = MagicMock()
        mgr.backend.fetchone.return_value = None
        webhook = _make_webhook()

        result = mgr.create_webhook(webhook)

        assert result is True
        assert "wh-1" in mgr._webhooks

    def test_create_webhook_workflow_not_found(self):
        mgr = _make_webhook_manager()
        mgr.workflow_engine.load_workflow.return_value = None
        webhook = _make_webhook()

        try:
            mgr.create_webhook(webhook)
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "not found" in str(exc)

    def test_create_webhook_duplicate_id(self):
        mgr = _make_webhook_manager()
        mgr.workflow_engine.load_workflow.return_value = MagicMock()
        mgr._webhooks["wh-1"] = _make_webhook()
        webhook = _make_webhook()

        try:
            mgr.create_webhook(webhook)
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "already exists" in str(exc)

    def test_update_webhook_preserves_metadata(self):
        mgr = _make_webhook_manager()
        original = _make_webhook()
        original.created_at = datetime(2024, 1, 1)
        original.trigger_count = 10
        original.last_triggered = datetime(2025, 5, 1)
        mgr._webhooks["wh-1"] = original
        mgr.backend.fetchone.return_value = {"id": "wh-1"}

        updated = _make_webhook()
        updated.name = "Updated Webhook"
        updated.created_at = datetime(2099, 12, 31)  # Should be overwritten
        updated.trigger_count = 0  # Should be overwritten

        result = mgr.update_webhook(updated)

        assert result is True
        saved = mgr._webhooks["wh-1"]
        assert saved.name == "Updated Webhook"
        assert saved.created_at == datetime(2024, 1, 1)
        assert saved.trigger_count == 10
        assert saved.last_triggered == datetime(2025, 5, 1)

    def test_update_webhook_not_found(self):
        mgr = _make_webhook_manager()

        try:
            mgr.update_webhook(_make_webhook(wid="nonexistent"))
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "not found" in str(exc)

    def test_delete_webhook(self):
        mgr = _make_webhook_manager()
        mgr._webhooks["wh-1"] = _make_webhook()

        result = mgr.delete_webhook("wh-1")

        assert result is True
        assert "wh-1" not in mgr._webhooks

    def test_delete_nonexistent_returns_false(self):
        mgr = _make_webhook_manager()

        assert mgr.delete_webhook("missing") is False

    def test_get_webhook(self):
        mgr = _make_webhook_manager()
        wh = _make_webhook()
        mgr._webhooks["wh-1"] = wh

        assert mgr.get_webhook("wh-1") is wh
        assert mgr.get_webhook("missing") is None

    def test_list_webhooks_excludes_secrets(self):
        mgr = _make_webhook_manager()
        wh = _make_webhook()
        wh.last_triggered = datetime(2025, 6, 1)
        mgr._webhooks["wh-1"] = wh

        result = mgr.list_webhooks()

        assert len(result) == 1
        entry = result[0]
        assert entry["id"] == "wh-1"
        assert "secret" not in entry
        assert entry["trigger_count"] == 0
        assert entry["last_triggered"] is not None


class TestSignatureVerification:
    def test_verify_signature_valid(self):
        mgr = _make_webhook_manager()
        wh = _make_webhook(secret="my-secret")
        mgr._webhooks["wh-1"] = wh

        payload = b'{"event": "push"}'
        expected_sig = hmac.new(b"my-secret", payload, hashlib.sha256).hexdigest()

        assert mgr.verify_signature("wh-1", payload, expected_sig) is True

    def test_verify_signature_with_prefix(self):
        mgr = _make_webhook_manager()
        wh = _make_webhook(secret="my-secret")
        mgr._webhooks["wh-1"] = wh

        payload = b'{"event": "push"}'
        expected_sig = "sha256=" + hmac.new(b"my-secret", payload, hashlib.sha256).hexdigest()

        assert mgr.verify_signature("wh-1", payload, expected_sig) is True

    def test_verify_signature_invalid(self):
        mgr = _make_webhook_manager()
        wh = _make_webhook(secret="my-secret")
        mgr._webhooks["wh-1"] = wh

        payload = b'{"event": "push"}'

        assert mgr.verify_signature("wh-1", payload, "bad-signature") is False

    def test_verify_signature_unknown_webhook(self):
        mgr = _make_webhook_manager()

        assert mgr.verify_signature("nonexistent", b"data", "sig") is False

    def test_generate_signature(self):
        mgr = _make_webhook_manager()
        wh = _make_webhook(secret="my-secret")
        mgr._webhooks["wh-1"] = wh

        payload = b'{"test": true}'
        sig = mgr.generate_signature("wh-1", payload)

        assert sig.startswith("sha256=")
        expected = hmac.new(b"my-secret", payload, hashlib.sha256).hexdigest()
        assert sig == f"sha256={expected}"

    def test_generate_signature_unknown_webhook(self):
        mgr = _make_webhook_manager()

        try:
            mgr.generate_signature("nonexistent", b"data")
            assert False, "Expected ValueError"
        except ValueError:
            pass


class TestWebhookTrigger:
    def test_trigger_executes_workflow(self):
        mgr = _make_webhook_manager()
        wh = _make_webhook()
        mgr._webhooks["wh-1"] = wh

        mock_workflow = MagicMock()
        mock_workflow.variables = {}
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.model_dump.return_value = {"status": "completed"}
        mgr.workflow_engine.load_workflow.return_value = mock_workflow
        mgr.workflow_engine.execute_workflow.return_value = mock_result
        mgr.backend.fetchone.return_value = {"id": "wh-1"}

        result = mgr.trigger("wh-1", {"event": "push"}, source_ip="127.0.0.1")

        assert result["status"] == "completed"
        assert result["webhook_id"] == "wh-1"
        assert result["error"] is None
        assert mgr._webhooks["wh-1"].trigger_count == 1

    def test_trigger_disabled_webhook_raises(self):
        mgr = _make_webhook_manager()
        wh = _make_webhook(status=WebhookStatus.DISABLED)
        mgr._webhooks["wh-1"] = wh

        try:
            mgr.trigger("wh-1", {"event": "push"})
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "disabled" in str(exc)

    def test_trigger_nonexistent_webhook_raises(self):
        mgr = _make_webhook_manager()

        try:
            mgr.trigger("missing", {"event": "push"})
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "not found" in str(exc)

    def test_trigger_with_payload_mapping(self):
        mgr = _make_webhook_manager()
        wh = _make_webhook()
        wh.payload_mappings = [
            PayloadMapping(
                source_path="data.user.id",
                target_variable="user_id",
            ),
            PayloadMapping(
                source_path="data.missing_field",
                target_variable="fallback_var",
                default="default_value",
            ),
        ]
        wh.static_variables = {"env": "prod"}
        mgr._webhooks["wh-1"] = wh

        mock_workflow = MagicMock()
        mock_workflow.variables = {}
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.model_dump.return_value = {"status": "completed"}
        mgr.workflow_engine.load_workflow.return_value = mock_workflow
        mgr.workflow_engine.execute_workflow.return_value = mock_result
        mgr.backend.fetchone.return_value = {"id": "wh-1"}

        payload = {"data": {"user": {"id": "u-123"}}}
        mgr.trigger("wh-1", payload)

        # Check variables were mapped to the workflow
        updated_vars = mock_workflow.variables
        assert updated_vars["user_id"] == "u-123"
        assert updated_vars["fallback_var"] == "default_value"
        assert updated_vars["env"] == "prod"

    def test_trigger_workflow_execution_failure(self):
        mgr = _make_webhook_manager()
        wh = _make_webhook()
        mgr._webhooks["wh-1"] = wh
        mgr.workflow_engine.load_workflow.side_effect = Exception("Engine down")
        mgr.backend.fetchone.return_value = {"id": "wh-1"}

        result = mgr.trigger("wh-1", {"event": "push"})

        assert result["status"] == "failed"
        assert result["error"] is not None
        assert "Engine down" in result["error"]


class TestRegenerateSecret:
    def test_regenerate_secret_returns_new_secret(self):
        mgr = _make_webhook_manager()
        wh = _make_webhook(secret="old-secret")
        mgr._webhooks["wh-1"] = wh
        mgr.backend.fetchone.return_value = {"id": "wh-1"}

        new_secret = mgr.regenerate_secret("wh-1")

        assert new_secret != "old-secret"
        assert len(new_secret) > 20  # token_urlsafe(32) produces ~43 chars
        assert mgr._webhooks["wh-1"].secret == new_secret

    def test_regenerate_secret_nonexistent_raises(self):
        mgr = _make_webhook_manager()

        try:
            mgr.regenerate_secret("missing")
            assert False, "Expected ValueError"
        except ValueError:
            pass


class TestTriggerHistory:
    def test_get_trigger_history(self):
        mgr = _make_webhook_manager()
        mgr.backend.fetchall.return_value = [
            {
                "webhook_id": "wh-1",
                "workflow_id": "wf-1",
                "triggered_at": "2025-06-15T12:00:00",
                "source_ip": "10.0.0.1",
                "payload_size": 256,
                "status": "success",
                "duration_seconds": 1.2,
                "error": None,
            },
        ]

        logs = mgr.get_trigger_history("wh-1", limit=5)

        assert len(logs) == 1
        assert logs[0].webhook_id == "wh-1"
        assert logs[0].source_ip == "10.0.0.1"
        assert logs[0].status == "success"

    def test_get_trigger_history_empty(self):
        mgr = _make_webhook_manager()
        mgr.backend.fetchall.return_value = []

        assert mgr.get_trigger_history("wh-1") == []


# ========== WebhookDeliveryManager Tests ==========


class TestRetryStrategy:
    def test_exponential_backoff_no_jitter(self):
        strategy = RetryStrategy(base_delay=1.0, exponential_base=2.0, max_delay=60.0, jitter=False)

        assert strategy.get_delay(0) == 1.0  # 1 * 2^0 = 1
        assert strategy.get_delay(1) == 2.0  # 1 * 2^1 = 2
        assert strategy.get_delay(2) == 4.0  # 1 * 2^2 = 4
        assert strategy.get_delay(3) == 8.0  # 1 * 2^3 = 8

    def test_max_delay_cap(self):
        strategy = RetryStrategy(
            base_delay=10.0, exponential_base=10.0, max_delay=50.0, jitter=False
        )

        # 10 * 10^3 = 10000, but capped at 50
        assert strategy.get_delay(3) == 50.0

    def test_jitter_adds_randomization(self):
        strategy = RetryStrategy(
            base_delay=10.0, exponential_base=2.0, max_delay=1000.0, jitter=True
        )

        # With jitter, delay should be between 50% and 150% of base calculation
        # For attempt=0: base = 10*2^0 = 10, so range is 5..15
        delays = {strategy.get_delay(0) for _ in range(20)}
        # With randomization, we should see variation (not all identical)
        assert len(delays) > 1


class TestDeliveryManagerDeliver:
    def test_successful_delivery(self):
        dm = _make_delivery_manager()
        dm.backend.fetchone.return_value = {"id": 1}

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.text = "OK"

        with patch("animus_forge.webhooks.webhook_delivery.get_sync_client") as mock_client_fn:
            client = MagicMock()
            client.post.return_value = mock_response
            mock_client_fn.return_value = client

            delivery = dm.deliver(
                url="https://example.com/hook",
                payload={"event": "test"},
            )

        assert delivery.status == DeliveryStatus.SUCCESS
        assert delivery.response_status == 200
        assert delivery.attempt_count == 1

    def test_delivery_with_secret_adds_signature_header(self):
        dm = _make_delivery_manager()
        dm.backend.fetchone.return_value = {"id": 1}

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.text = "OK"

        captured_headers = {}

        with patch("animus_forge.webhooks.webhook_delivery.get_sync_client") as mock_client_fn:
            client = MagicMock()

            def capture_post(url, data, headers, timeout):
                captured_headers.update(headers)
                return mock_response

            client.post.side_effect = capture_post
            mock_client_fn.return_value = client

            dm.deliver(
                url="https://example.com/hook",
                payload={"event": "test"},
                secret="webhook-secret",
            )

        assert "X-Webhook-Signature" in captured_headers
        assert captured_headers["X-Webhook-Signature"].startswith("sha256=")

    def test_delivery_failure_moves_to_dead_letter(self):
        dm = _make_delivery_manager()
        dm.backend.fetchone.return_value = {"id": 1}

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("animus_forge.webhooks.webhook_delivery.get_sync_client") as mock_client_fn:
            client = MagicMock()
            client.post.return_value = mock_response
            mock_client_fn.return_value = client

            delivery = dm.deliver(
                url="https://example.com/hook",
                payload={"event": "test"},
                max_retries=0,
            )

        assert delivery.status == DeliveryStatus.DEAD_LETTER
        assert delivery.last_error == "HTTP 500"


class TestDeliveryManagerDLQ:
    def test_get_dlq_items(self):
        dm = _make_delivery_manager()
        dm.backend.fetchall.return_value = [
            {
                "id": 1,
                "delivery_id": 10,
                "webhook_url": "https://example.com/hook",
                "payload": '{"event": "test"}',
                "headers": "{}",
                "error": "HTTP 500",
                "attempt_count": 3,
                "created_at": "2025-06-15T12:00:00",
                "reprocessed_at": None,
            },
        ]

        items = dm.get_dlq_items(limit=10)

        assert len(items) == 1
        assert items[0]["webhook_url"] == "https://example.com/hook"

    def test_get_delivery_stats(self):
        dm = _make_delivery_manager()

        # Use a callable side_effect to avoid fragile list ordering
        status_counts = {
            "pending": 1,
            "success": 5,
            "failed": 2,
            "retrying": 0,
            "dead_letter": 1,
            "circuit_broken": 0,
        }

        def _fetchone(sql, params=None):
            if "AVG" in sql:
                return {"avg_attempts": 1.5}
            if "webhook_dead_letter" in sql:
                return {"count": 1}
            # Status count query
            if params:
                return {"count": status_counts.get(params[0], 0)}
            return {"count": 0}

        dm.backend.fetchone.side_effect = _fetchone

        stats = dm.get_delivery_stats()

        assert stats["success_count"] == 5
        assert stats["failed_count"] == 2
        assert stats["dead_letter_count"] == 1
        assert stats["dlq_pending_count"] == 1
        assert stats["avg_attempts_success"] == 1.5
