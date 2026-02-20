"""Additional coverage tests for webhook manager."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")


def _make_manager():
    """Create a WebhookManager with mocked dependencies."""
    with (
        patch("animus_forge.webhooks.webhook_manager.get_settings") as mock_settings,
        patch("animus_forge.webhooks.webhook_manager.get_database") as mock_db,
        patch("animus_forge.webhooks.webhook_manager.WorkflowEngineAdapter") as mock_engine,
    ):
        settings = MagicMock()
        mock_settings.return_value = settings
        backend = MagicMock()
        backend.fetchall.return_value = []
        mock_db.return_value = backend

        from animus_forge.webhooks.webhook_manager import WebhookManager

        mgr = WebhookManager(backend=backend)
        return mgr, backend, mock_engine.return_value


def _make_webhook(**kwargs):
    from animus_forge.webhooks.webhook_manager import Webhook, WebhookStatus

    defaults = {
        "id": "wh-1",
        "name": "Test Webhook",
        "workflow_id": "wf-1",
        "secret": "test-secret",
        "status": WebhookStatus.ACTIVE,
    }
    defaults.update(kwargs)
    return Webhook(**defaults)


class TestWebhookManagerCreate:
    def test_create_webhook(self):
        mgr, backend, engine = _make_manager()
        engine.load_workflow.return_value = MagicMock()
        backend.fetchone.return_value = None
        backend.transaction.return_value.__enter__ = MagicMock()
        backend.transaction.return_value.__exit__ = MagicMock(return_value=False)

        wh = _make_webhook()
        result = mgr.create_webhook(wh)
        assert result is True
        assert "wh-1" in mgr._webhooks

    def test_create_duplicate_raises(self):
        mgr, backend, engine = _make_manager()
        engine.load_workflow.return_value = MagicMock()
        backend.fetchone.return_value = None
        backend.transaction.return_value.__enter__ = MagicMock()
        backend.transaction.return_value.__exit__ = MagicMock(return_value=False)

        wh = _make_webhook()
        mgr.create_webhook(wh)
        with pytest.raises(ValueError, match="already exists"):
            mgr.create_webhook(wh)

    def test_create_workflow_not_found(self):
        mgr, backend, engine = _make_manager()
        engine.load_workflow.return_value = None

        with pytest.raises(ValueError, match="not found"):
            mgr.create_webhook(_make_webhook())


class TestWebhookManagerOperations:
    def _setup_with_webhook(self):
        mgr, backend, engine = _make_manager()
        wh = _make_webhook()
        mgr._webhooks["wh-1"] = wh
        backend.transaction.return_value.__enter__ = MagicMock()
        backend.transaction.return_value.__exit__ = MagicMock(return_value=False)
        return mgr, backend, engine, wh

    def test_get_webhook(self):
        mgr, _, _, _ = self._setup_with_webhook()
        assert mgr.get_webhook("wh-1") is not None
        assert mgr.get_webhook("nonexistent") is None

    def test_list_webhooks(self):
        mgr, _, _, _ = self._setup_with_webhook()
        webhooks = mgr.list_webhooks()
        assert len(webhooks) == 1
        assert webhooks[0]["id"] == "wh-1"
        assert "secret" not in webhooks[0]

    def test_delete_webhook(self):
        mgr, backend, _, _ = self._setup_with_webhook()
        result = mgr.delete_webhook("wh-1")
        assert result is True
        assert "wh-1" not in mgr._webhooks

    def test_delete_nonexistent(self):
        mgr, _, _, _ = self._setup_with_webhook()
        assert mgr.delete_webhook("nonexistent") is False

    def test_update_webhook(self):
        mgr, backend, _, _ = self._setup_with_webhook()
        backend.fetchone.return_value = {"id": "wh-1"}
        updated = _make_webhook(name="Updated Name")
        result = mgr.update_webhook(updated)
        assert result is True

    def test_update_nonexistent_raises(self):
        mgr, _, _, _ = self._setup_with_webhook()
        with pytest.raises(ValueError, match="not found"):
            mgr.update_webhook(_make_webhook(id="nonexistent"))


class TestWebhookSignature:
    def test_verify_and_generate(self):
        mgr, _, _, _ = TestWebhookManagerOperations()._setup_with_webhook()
        payload = b'{"test": true}'
        sig = mgr.generate_signature("wh-1", payload)
        assert sig.startswith("sha256=")
        assert mgr.verify_signature("wh-1", payload, sig) is True

    def test_verify_raw_hex(self):
        mgr, _, _, _ = TestWebhookManagerOperations()._setup_with_webhook()
        payload = b'{"test": true}'
        sig = mgr.generate_signature("wh-1", payload)
        raw = sig.replace("sha256=", "")
        assert mgr.verify_signature("wh-1", payload, raw) is True

    def test_verify_nonexistent(self):
        mgr, _, _, _ = TestWebhookManagerOperations()._setup_with_webhook()
        assert mgr.verify_signature("nonexistent", b"x", "sig") is False

    def test_generate_nonexistent_raises(self):
        mgr, _, _, _ = TestWebhookManagerOperations()._setup_with_webhook()
        with pytest.raises(ValueError):
            mgr.generate_signature("nonexistent", b"x")


class TestPayloadExtraction:
    def test_extract_nested(self):
        mgr, _, _, _ = TestWebhookManagerOperations()._setup_with_webhook()
        payload = {"data": {"user": {"id": 42}}}
        assert mgr._extract_payload_value(payload, "data.user.id") == 42

    def test_extract_missing(self):
        mgr, _, _, _ = TestWebhookManagerOperations()._setup_with_webhook()
        assert mgr._extract_payload_value({}, "missing.key") is None

    def test_map_payload_to_variables(self):
        from animus_forge.webhooks.webhook_manager import PayloadMapping

        mgr, _, _, _ = TestWebhookManagerOperations()._setup_with_webhook()
        wh = mgr._webhooks["wh-1"]
        wh.payload_mappings = [
            PayloadMapping(source_path="user.name", target_variable="username"),
            PayloadMapping(
                source_path="missing", target_variable="fallback", default="default_val"
            ),
        ]
        wh.static_variables = {"env": "test"}

        payload = {"user": {"name": "alice"}}
        variables = mgr._map_payload_to_variables(wh, payload)
        assert variables["username"] == "alice"
        assert variables["fallback"] == "default_val"
        assert variables["env"] == "test"


class TestWebhookTrigger:
    def test_trigger_success(self):
        mgr, backend, engine = _make_manager()
        wh = _make_webhook()
        mgr._webhooks["wh-1"] = wh
        backend.fetchone.return_value = {"id": "wh-1"}
        backend.transaction.return_value.__enter__ = MagicMock()
        backend.transaction.return_value.__exit__ = MagicMock(return_value=False)

        mock_workflow = MagicMock()
        mock_workflow.variables = {}
        engine.load_workflow.return_value = mock_workflow
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.model_dump.return_value = {"status": "success"}
        engine.execute_workflow.return_value = mock_result

        result = mgr.trigger("wh-1", {"data": "test"})
        assert result["status"] == "success"

    def test_trigger_nonexistent(self):
        mgr, _, _ = _make_manager()
        with pytest.raises(ValueError, match="not found"):
            mgr.trigger("nonexistent", {})

    def test_trigger_disabled(self):
        from animus_forge.webhooks.webhook_manager import WebhookStatus

        mgr, _, _ = _make_manager()
        wh = _make_webhook(status=WebhookStatus.DISABLED)
        mgr._webhooks["wh-1"] = wh
        with pytest.raises(ValueError, match="disabled"):
            mgr.trigger("wh-1", {})


class TestWebhookRegenerateSecret:
    def test_regenerate(self):
        mgr, backend, _ = _make_manager()
        wh = _make_webhook()
        mgr._webhooks["wh-1"] = wh
        old_secret = wh.secret
        backend.fetchone.return_value = {"id": "wh-1"}
        backend.transaction.return_value.__enter__ = MagicMock()
        backend.transaction.return_value.__exit__ = MagicMock(return_value=False)

        new_secret = mgr.regenerate_secret("wh-1")
        assert new_secret != old_secret

    def test_regenerate_nonexistent(self):
        mgr, _, _ = _make_manager()
        with pytest.raises(ValueError):
            mgr.regenerate_secret("nonexistent")


class TestRowToWebhook:
    def test_valid_row(self):
        mgr, _, _ = _make_manager()
        row = {
            "id": "wh-2",
            "name": "Test",
            "description": "",
            "workflow_id": "wf-1",
            "secret": "s",
            "payload_mappings": None,
            "static_variables": None,
            "status": "active",
            "created_at": "2024-01-01T00:00:00",
            "last_triggered": None,
            "trigger_count": 0,
        }
        wh = mgr._row_to_webhook(row)
        assert wh is not None
        assert wh.id == "wh-2"

    def test_row_with_mappings(self):
        mgr, _, _ = _make_manager()
        row = {
            "id": "wh-3",
            "name": "Test",
            "description": "",
            "workflow_id": "wf-1",
            "secret": "s",
            "payload_mappings": json.dumps([{"source_path": "a.b", "target_variable": "c"}]),
            "static_variables": json.dumps({"key": "val"}),
            "status": "active",
            "created_at": "2024-01-01T00:00:00",
            "last_triggered": None,
            "trigger_count": 0,
        }
        wh = mgr._row_to_webhook(row)
        assert len(wh.payload_mappings) == 1
        assert wh.static_variables["key"] == "val"

    def test_invalid_row(self):
        mgr, _, _ = _make_manager()
        wh = mgr._row_to_webhook({"bad": "data"})
        assert wh is None


class TestGetTriggerHistory:
    def test_get_history(self):
        mgr, backend, _ = _make_manager()
        backend.fetchall.return_value = [
            {
                "webhook_id": "wh-1",
                "workflow_id": "wf-1",
                "triggered_at": "2024-01-01T00:00:00",
                "source_ip": "127.0.0.1",
                "payload_size": 50,
                "status": "success",
                "duration_seconds": 1.5,
                "error": None,
            }
        ]
        logs = mgr.get_trigger_history("wh-1")
        assert len(logs) == 1
        assert logs[0].webhook_id == "wh-1"

    def test_empty_history(self):
        mgr, backend, _ = _make_manager()
        backend.fetchall.return_value = []
        logs = mgr.get_trigger_history("wh-1")
        assert len(logs) == 0
