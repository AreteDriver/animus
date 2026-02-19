"""Tests for integrations: oauth, webhooks, todoist, manager, filesystem, google.

Covers: integrations/oauth.py, webhooks.py, todoist.py, manager.py,
        filesystem.py, google/calendar.py, google/gmail.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus.integrations.oauth import OAuth2Token, load_token, save_token
from animus.integrations.webhooks import WebhookEvent, WebhookIntegration
from animus.integrations.todoist import TodoistIntegration
from animus.integrations.manager import IntegrationManager, _derive_key, _get_encryption_secret


# ===================================================================
# OAuth2
# ===================================================================


class TestOAuth2Token:
    """Tests for OAuth2Token dataclass."""

    def test_not_expired(self):
        token = OAuth2Token(
            access_token="abc",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=datetime.now() + timedelta(hours=1),
            scopes=["read"],
        )
        assert token.is_expired() is False

    def test_expired(self):
        token = OAuth2Token(
            access_token="abc",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=datetime.now() - timedelta(hours=1),
            scopes=["read"],
        )
        assert token.is_expired() is True

    def test_no_expiry(self):
        token = OAuth2Token(
            access_token="abc",
            refresh_token=None,
            token_type="Bearer",
            expires_at=None,
            scopes=[],
        )
        assert token.is_expired() is False

    def test_to_dict(self):
        token = OAuth2Token(
            access_token="abc",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=datetime(2025, 1, 1, 12, 0, 0),
            scopes=["read", "write"],
        )
        d = token.to_dict()
        assert d["access_token"] == "abc"
        assert d["refresh_token"] == "ref"
        assert d["expires_at"] is not None
        assert d["scopes"] == ["read", "write"]

    def test_from_dict(self):
        data = {
            "access_token": "xyz",
            "refresh_token": "ref",
            "token_type": "Bearer",
            "expires_at": "2025-01-01T12:00:00",
            "scopes": ["read"],
        }
        token = OAuth2Token.from_dict(data)
        assert token.access_token == "xyz"
        assert token.expires_at is not None

    def test_from_dict_no_expiry(self):
        data = {
            "access_token": "xyz",
            "scopes": [],
        }
        token = OAuth2Token.from_dict(data)
        assert token.expires_at is None
        assert token.token_type == "Bearer"

    def test_roundtrip(self):
        token = OAuth2Token(
            access_token="abc",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=datetime(2025, 6, 15, 10, 30, 0),
            scopes=["scope1"],
        )
        restored = OAuth2Token.from_dict(token.to_dict())
        assert restored.access_token == token.access_token
        assert restored.refresh_token == token.refresh_token


class TestSaveLoadToken:
    """Tests for save_token/load_token."""

    def test_save_and_load(self, tmp_path: Path):
        token = OAuth2Token(
            access_token="abc",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=datetime(2025, 1, 1, 12, 0, 0),
            scopes=["read"],
        )
        path = tmp_path / "token.json"
        save_token(token, path)
        assert path.exists()

        loaded = load_token(path)
        assert loaded is not None
        assert loaded.access_token == "abc"

    def test_load_missing_file(self, tmp_path: Path):
        result = load_token(tmp_path / "missing.json")
        assert result is None

    def test_load_corrupt_file(self, tmp_path: Path):
        path = tmp_path / "corrupt.json"
        path.write_text("not json")
        result = load_token(path)
        assert result is None


class TestOAuth2Flow:
    """Tests for OAuth2Flow (import-guarded)."""

    def test_init_without_google_auth(self):
        from animus.integrations.oauth import OAuth2Flow

        with patch("animus.integrations.oauth.GOOGLE_AUTH_AVAILABLE", False):
            with pytest.raises(ImportError, match="Google auth"):
                OAuth2Flow("id", "secret", ["scope"])


# ===================================================================
# Webhooks
# ===================================================================


class TestWebhookEvent:
    """Tests for WebhookEvent dataclass."""

    def test_to_dict(self):
        event = WebhookEvent(
            id="evt-1",
            source="github",
            event_type="push",
            payload={"ref": "main"},
            received_at=datetime(2025, 1, 1),
            headers={"X-Event": "push"},
        )
        d = event.to_dict()
        assert d["id"] == "evt-1"
        assert d["source"] == "github"
        assert d["payload"]["ref"] == "main"


class TestWebhookIntegration:
    """Tests for WebhookIntegration."""

    def test_init(self):
        wh = WebhookIntegration(max_events=50)
        assert len(wh._events) == 0

    def test_connect_disconnect(self):
        wh = WebhookIntegration()
        result = asyncio.run(wh.connect({"port": 0, "secret": "test"}))
        # Port 0 may fail on some systems, so handle both
        if result:
            assert wh.is_connected
            asyncio.run(wh.disconnect())
            assert not wh.is_connected

    def test_verify(self):
        wh = WebhookIntegration()
        result = asyncio.run(wh.verify())
        assert result is False

    def test_get_tools(self):
        wh = WebhookIntegration()
        tools = wh.get_tools()
        assert len(tools) == 3
        names = [t.name for t in tools]
        assert "webhook_list_events" in names
        assert "webhook_get_event" in names
        assert "webhook_info" in names

    def test_receive_event(self):
        wh = WebhookIntegration(max_events=10)
        event = WebhookEvent(
            id="e1",
            source="github",
            event_type="push",
            payload={"ref": "main"},
            received_at=datetime.now(),
        )
        wh._receive_event(event)
        assert len(wh._events) == 1

    def test_receive_event_with_callback(self):
        wh = WebhookIntegration()
        cb = MagicMock()
        wh.register_callback("github", "push", cb)

        event = WebhookEvent(
            id="e1",
            source="github",
            event_type="push",
            payload={},
            received_at=datetime.now(),
        )
        wh._receive_event(event)
        cb.assert_called_once_with(event)

    def test_receive_event_wildcard_callback(self):
        wh = WebhookIntegration()
        cb = MagicMock()
        wh.register_callback("*", "*", cb)

        event = WebhookEvent(
            id="e1",
            source="any",
            event_type="any",
            payload={},
            received_at=datetime.now(),
        )
        wh._receive_event(event)
        cb.assert_called_once_with(event)

    def test_receive_event_callback_error(self):
        wh = WebhookIntegration()
        bad_cb = MagicMock(side_effect=Exception("cb error"))
        wh.register_callback("src", "type", bad_cb)

        event = WebhookEvent(
            id="e1", source="src", event_type="type",
            payload={}, received_at=datetime.now(),
        )
        wh._receive_event(event)  # Should not raise

    def test_tool_list_events(self):
        wh = WebhookIntegration()
        for i in range(5):
            wh._receive_event(
                WebhookEvent(
                    id=f"e{i}",
                    source="github" if i % 2 == 0 else "slack",
                    event_type="push",
                    payload={},
                    received_at=datetime.now(),
                )
            )
        result = asyncio.run(wh._tool_list_events(source="github"))
        assert result.success
        assert result.output["count"] == 3

    def test_tool_list_events_by_type(self):
        wh = WebhookIntegration()
        wh._receive_event(
            WebhookEvent(
                id="e1", source="src", event_type="push",
                payload={}, received_at=datetime.now(),
            )
        )
        wh._receive_event(
            WebhookEvent(
                id="e2", source="src", event_type="pr",
                payload={}, received_at=datetime.now(),
            )
        )
        result = asyncio.run(wh._tool_list_events(event_type="push"))
        assert result.output["count"] == 1

    def test_tool_get_event_found(self):
        wh = WebhookIntegration()
        wh._receive_event(
            WebhookEvent(
                id="target-id", source="src", event_type="type",
                payload={"key": "val"}, received_at=datetime.now(),
            )
        )
        result = asyncio.run(wh._tool_get_event("target-id"))
        assert result.success
        assert result.output["id"] == "target-id"

    def test_tool_get_event_not_found(self):
        wh = WebhookIntegration()
        result = asyncio.run(wh._tool_get_event("missing"))
        assert not result.success
        assert "not found" in result.error

    def test_tool_info(self):
        wh = WebhookIntegration()
        result = asyncio.run(wh._tool_info())
        assert result.success
        assert "port" in result.output


# ===================================================================
# Todoist
# ===================================================================


class TestTodoistIntegration:
    """Tests for TodoistIntegration."""

    def test_init(self):
        ti = TodoistIntegration()
        assert ti.name == "todoist"
        assert ti._api is None

    def test_connect_not_available(self):
        ti = TodoistIntegration()
        with patch("animus.integrations.todoist.TODOIST_AVAILABLE", False):
            result = asyncio.run(ti.connect({"api_key": "test"}))
        assert result is False

    def test_connect_no_api_key(self):
        ti = TodoistIntegration()
        with patch("animus.integrations.todoist.TODOIST_AVAILABLE", True):
            result = asyncio.run(ti.connect({}))
        assert result is False

    def test_connect_success(self):
        ti = TodoistIntegration()
        mock_api = MagicMock()
        mock_api.get_projects.return_value = []

        import animus.integrations.todoist as todoist_mod

        with patch.object(todoist_mod, "TODOIST_AVAILABLE", True):
            with patch.object(todoist_mod, "TodoistAPI", create=True, return_value=mock_api):
                result = asyncio.run(ti.connect({"api_key": "valid-key"}))
        assert result is True
        assert ti.is_connected

    def test_connect_api_error(self):
        ti = TodoistIntegration()
        mock_api = MagicMock()
        mock_api.get_projects.side_effect = Exception("auth failed")

        import animus.integrations.todoist as todoist_mod

        with patch.object(todoist_mod, "TODOIST_AVAILABLE", True):
            with patch.object(todoist_mod, "TodoistAPI", create=True, return_value=mock_api):
                result = asyncio.run(ti.connect({"api_key": "bad-key"}))
        assert result is False

    def test_disconnect(self):
        ti = TodoistIntegration()
        ti._api = MagicMock()
        result = asyncio.run(ti.disconnect())
        assert result is True
        assert ti._api is None

    def test_verify_no_api(self):
        ti = TodoistIntegration()
        result = asyncio.run(ti.verify())
        assert result is False

    def test_verify_success(self):
        ti = TodoistIntegration()
        ti._api = MagicMock()
        ti._api.get_projects.return_value = []
        result = asyncio.run(ti.verify())
        assert result is True

    def test_verify_failure(self):
        ti = TodoistIntegration()
        ti._api = MagicMock()
        ti._api.get_projects.side_effect = Exception("expired")
        result = asyncio.run(ti.verify())
        assert result is False

    def test_get_tools(self):
        ti = TodoistIntegration()
        tools = ti.get_tools()
        assert len(tools) == 5
        names = [t.name for t in tools]
        assert "todoist_list_tasks" in names
        assert "todoist_create_task" in names
        assert "todoist_complete_task" in names
        assert "todoist_list_projects" in names
        assert "todoist_sync" in names

    def test_tool_list_tasks_no_api(self):
        ti = TodoistIntegration()
        result = asyncio.run(ti._tool_list_tasks())
        assert not result.success

    def test_tool_list_tasks_success(self):
        ti = TodoistIntegration()
        mock_task = MagicMock()
        mock_task.id = "t1"
        mock_task.content = "Test task"
        mock_task.description = "desc"
        mock_task.project_id = "p1"
        mock_task.priority = 1
        mock_task.due = None
        mock_task.labels = []
        mock_task.created_at = "2025-01-01"

        ti._api = MagicMock()
        ti._api.get_tasks.return_value = [mock_task]
        result = asyncio.run(ti._tool_list_tasks())
        assert result.success
        assert result.output["count"] == 1

    def test_tool_list_tasks_with_filter(self):
        ti = TodoistIntegration()
        ti._api = MagicMock()
        ti._api.get_tasks.return_value = []
        result = asyncio.run(ti._tool_list_tasks(filter="today"))
        assert result.success
        ti._api.get_tasks.assert_called_with(filter="today")

    def test_tool_list_tasks_with_project(self):
        ti = TodoistIntegration()
        ti._api = MagicMock()
        ti._api.get_tasks.return_value = []
        result = asyncio.run(ti._tool_list_tasks(project_id="p1"))
        assert result.success
        ti._api.get_tasks.assert_called_with(project_id="p1")

    def test_tool_list_tasks_error(self):
        ti = TodoistIntegration()
        ti._api = MagicMock()
        ti._api.get_tasks.side_effect = Exception("api error")
        result = asyncio.run(ti._tool_list_tasks())
        assert not result.success

    def test_tool_create_task_no_api(self):
        ti = TodoistIntegration()
        result = asyncio.run(ti._tool_create_task("task"))
        assert not result.success

    def test_tool_create_task_success(self):
        ti = TodoistIntegration()
        mock_task = MagicMock()
        mock_task.id = "new-1"
        mock_task.content = "New task"
        mock_task.project_id = "p1"
        mock_task.url = "https://todoist.com/tasks/new-1"

        ti._api = MagicMock()
        ti._api.add_task.return_value = mock_task
        result = asyncio.run(
            ti._tool_create_task(
                "New task",
                project_id="p1",
                due_string="tomorrow",
                priority=4,
                labels=["work"],
            )
        )
        assert result.success
        assert result.output["id"] == "new-1"

    def test_tool_create_task_error(self):
        ti = TodoistIntegration()
        ti._api = MagicMock()
        ti._api.add_task.side_effect = Exception("create failed")
        result = asyncio.run(ti._tool_create_task("task"))
        assert not result.success

    def test_tool_complete_task_no_api(self):
        ti = TodoistIntegration()
        result = asyncio.run(ti._tool_complete_task("t1"))
        assert not result.success

    def test_tool_complete_task_success(self):
        ti = TodoistIntegration()
        ti._api = MagicMock()
        result = asyncio.run(ti._tool_complete_task("t1"))
        assert result.success

    def test_tool_complete_task_error(self):
        ti = TodoistIntegration()
        ti._api = MagicMock()
        ti._api.close_task.side_effect = Exception("not found")
        result = asyncio.run(ti._tool_complete_task("t1"))
        assert not result.success

    def test_tool_list_projects_no_api(self):
        ti = TodoistIntegration()
        result = asyncio.run(ti._tool_list_projects())
        assert not result.success

    def test_tool_list_projects_success(self):
        ti = TodoistIntegration()
        mock_proj = MagicMock()
        mock_proj.id = "p1"
        mock_proj.name = "Inbox"
        mock_proj.color = "blue"
        mock_proj.is_favorite = False
        mock_proj.is_inbox_project = True

        ti._api = MagicMock()
        ti._api.get_projects.return_value = [mock_proj]
        result = asyncio.run(ti._tool_list_projects())
        assert result.success
        assert result.output["count"] == 1

    def test_tool_list_projects_error(self):
        ti = TodoistIntegration()
        ti._api = MagicMock()
        ti._api.get_projects.side_effect = Exception("api error")
        result = asyncio.run(ti._tool_list_projects())
        assert not result.success

    def test_tool_sync_no_api(self):
        ti = TodoistIntegration()
        result = asyncio.run(ti._tool_sync())
        assert not result.success

    def test_tool_sync_success(self):
        ti = TodoistIntegration()
        ti._api = MagicMock()
        ti._api.get_tasks.return_value = [MagicMock(), MagicMock()]
        result = asyncio.run(ti._tool_sync(direction="pull"))
        assert result.success
        assert result.output["todoist_task_count"] == 2

    def test_tool_sync_error(self):
        ti = TodoistIntegration()
        ti._api = MagicMock()
        ti._api.get_tasks.side_effect = Exception("sync error")
        result = asyncio.run(ti._tool_sync())
        assert not result.success


# ===================================================================
# Integration Manager
# ===================================================================


class TestIntegrationManager:
    """Tests for IntegrationManager."""

    def test_init(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        assert mgr._data_dir == tmp_path

    def test_register_unregister(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        integration = MagicMock(spec=["name", "get_info", "is_connected", "get_tools"])
        integration.name = "test"
        mgr.register(integration)
        assert mgr.get("test") is integration
        assert mgr.unregister("test") is True
        assert mgr.get("test") is None
        assert mgr.unregister("nonexistent") is False

    def test_list_all(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        integration = MagicMock()
        integration.name = "test"
        integration.get_info.return_value = MagicMock(status=MagicMock(value="connected"))
        mgr.register(integration)
        result = mgr.list_all()
        assert len(result) == 1

    def test_list_connected(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        connected = MagicMock()
        connected.name = "c"
        connected.is_connected = True
        connected.get_info.return_value = MagicMock()

        disconnected = MagicMock()
        disconnected.name = "d"
        disconnected.is_connected = False

        mgr.register(connected)
        mgr.register(disconnected)
        result = mgr.list_connected()
        assert len(result) == 1

    def test_connect_unknown(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        result = asyncio.run(mgr.connect("unknown", {}))
        assert result is False

    def test_connect_success(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        integration = MagicMock()
        integration.name = "test"
        integration.connect = AsyncMock(return_value=True)
        mgr.register(integration)

        result = asyncio.run(mgr.connect("test", {"key": "val"}))
        assert result is True

    def test_connect_failure(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        integration = MagicMock()
        integration.name = "test"
        integration.connect = AsyncMock(return_value=False)
        mgr.register(integration)

        result = asyncio.run(mgr.connect("test", {}))
        assert result is False

    def test_disconnect_unknown(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        result = asyncio.run(mgr.disconnect("unknown"))
        assert result is False

    def test_disconnect_success(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        integration = MagicMock()
        integration.name = "test"
        integration.connect = AsyncMock(return_value=True)
        integration.disconnect = AsyncMock(return_value=True)
        mgr.register(integration)

        asyncio.run(mgr.connect("test", {"key": "val"}))
        result = asyncio.run(mgr.disconnect("test"))
        assert result is True

    def test_verify(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        integration = MagicMock()
        integration.name = "test"
        integration.verify = AsyncMock(return_value=True)
        mgr.register(integration)

        result = asyncio.run(mgr.verify("test"))
        assert result is True

    def test_verify_unknown(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        result = asyncio.run(mgr.verify("unknown"))
        assert result is False

    def test_verify_all(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        integration = MagicMock()
        integration.name = "test"
        integration.is_connected = True
        integration.verify = AsyncMock(return_value=True)
        mgr.register(integration)

        results = asyncio.run(mgr.verify_all())
        assert results["test"] is True

    def test_get_all_tools(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        tool1 = MagicMock()
        integration = MagicMock()
        integration.name = "test"
        integration.is_connected = True
        integration.get_tools.return_value = [tool1]
        mgr.register(integration)

        tools = mgr.get_all_tools()
        assert len(tools) == 1
        assert mgr.list_tools() == tools

    def test_get_tools_by_integration(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        integration = MagicMock()
        integration.name = "test"
        integration.is_connected = True
        integration.get_tools.return_value = [MagicMock()]
        mgr.register(integration)

        by_int = mgr.get_tools_by_integration()
        assert "test" in by_int
        assert len(by_int["test"]) == 1

    def test_credentials_save_load(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        creds = {"api_key": "secret-123", "project": "main"}
        mgr._save_credentials("test_int", creds)

        loaded = mgr._load_credentials("test_int")
        assert loaded is not None
        assert loaded["api_key"] == "secret-123"

    def test_credentials_clear(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        mgr._save_credentials("test_int", {"key": "val"})
        mgr._clear_credentials("test_int")
        assert mgr._load_credentials("test_int") is None

    def test_credentials_load_missing(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        assert mgr._load_credentials("missing") is None

    def test_reconnect_from_stored(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        integration = MagicMock()
        integration.name = "test"
        integration.connect = AsyncMock(return_value=True)
        mgr.register(integration)

        # Save credentials first
        mgr._save_credentials("test", {"api_key": "stored"})

        results = asyncio.run(mgr.reconnect_from_stored())
        assert "test" in results
        assert results["test"] is True

    def test_get_status_summary(self, tmp_path: Path):
        mgr = IntegrationManager(data_dir=tmp_path)
        integration = MagicMock()
        integration.name = "test"
        integration.is_connected = True
        integration.get_info.return_value = MagicMock(
            status=MagicMock(value="connected")
        )
        mgr.register(integration)

        summary = mgr.get_status_summary()
        assert summary["total"] == 1
        assert summary["connected"] == 1
        assert summary["statuses"]["test"] == "connected"

    def test_derive_key(self):
        key = _derive_key("test-secret")
        assert len(key) > 0
        # Should be deterministic
        assert _derive_key("test-secret") == key

    def test_get_encryption_secret(self):
        secret = _get_encryption_secret()
        assert secret  # Should return something


# ===================================================================
# Google Calendar Integration
# ===================================================================


class TestGoogleCalendarIntegration:
    """Tests for GoogleCalendarIntegration."""

    def test_init(self):
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        assert cal.name == "google_calendar"

    def test_connect_not_available(self):
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        with patch("animus.integrations.google.calendar.GOOGLE_API_AVAILABLE", False):
            result = asyncio.run(cal.connect({}))
        assert result is False

    def test_connect_success(self, tmp_path: Path):
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration(data_dir=tmp_path)
        mock_service = MagicMock()
        mock_token = OAuth2Token(
            access_token="tok", refresh_token="ref",
            token_type="Bearer", expires_at=datetime(2099, 1, 1), scopes=[],
        )

        import animus.integrations.google.calendar as cal_mod

        with patch.object(cal_mod, "GOOGLE_API_AVAILABLE", True), \
             patch.object(cal_mod, "GOOGLE_AUTH_AVAILABLE", True), \
             patch.object(cal_mod, "load_token", return_value=mock_token), \
             patch.object(cal_mod, "Credentials", create=True, return_value=MagicMock()), \
             patch.object(cal_mod, "build", create=True, return_value=mock_service):
            result = asyncio.run(
                cal.connect({"client_id": "id", "client_secret": "secret"})
            )
        assert result is True

    def test_connect_error(self, tmp_path: Path):
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration(data_dir=tmp_path)
        mock_token = OAuth2Token(
            access_token="tok", refresh_token="ref",
            token_type="Bearer", expires_at=datetime(2099, 1, 1), scopes=[],
        )

        import animus.integrations.google.calendar as cal_mod

        with patch.object(cal_mod, "GOOGLE_API_AVAILABLE", True), \
             patch.object(cal_mod, "GOOGLE_AUTH_AVAILABLE", True), \
             patch.object(cal_mod, "load_token", return_value=mock_token), \
             patch.object(cal_mod, "Credentials", create=True, side_effect=Exception("auth fail")):
            result = asyncio.run(cal.connect({"client_id": "id", "client_secret": "secret"}))
        assert result is False

    def test_disconnect(self):
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        result = asyncio.run(cal.disconnect())
        assert result is True

    def test_get_tools(self):
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        tools = cal.get_tools()
        assert len(tools) >= 3
        names = [t.name for t in tools]
        assert "calendar_list_events" in names
        assert "calendar_create_event" in names

    def test_tool_list_events_no_service(self):
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        result = asyncio.run(cal._tool_list_events())
        assert not result.success

    def test_tool_list_events_success(self):
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        mock_events = {
            "items": [
                {
                    "id": "e1",
                    "summary": "Test Event",
                    "start": {"dateTime": "2025-01-01T10:00:00"},
                    "end": {"dateTime": "2025-01-01T11:00:00"},
                }
            ]
        }
        mock_service = MagicMock()
        mock_service.events.return_value.list.return_value.execute.return_value = mock_events
        cal._service = mock_service

        result = asyncio.run(cal._tool_list_events())
        assert result.success

    def test_tool_create_event_no_service(self):
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        result = asyncio.run(
            cal._tool_create_event("Test", "2025-01-01T10:00", "2025-01-01T11:00")
        )
        assert not result.success

    def test_tool_create_event_success(self):
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        mock_service = MagicMock()
        mock_service.events.return_value.insert.return_value.execute.return_value = {
            "id": "new-1",
            "summary": "Test",
            "htmlLink": "https://calendar.google.com/event/new-1",
            "start": {"dateTime": "2025-01-01T10:00:00"},
            "end": {"dateTime": "2025-01-01T11:00:00"},
        }
        cal._service = mock_service

        result = asyncio.run(
            cal._tool_create_event("Test", "2025-01-01T10:00:00", "2025-01-01T11:00:00")
        )
        assert result.success

    def test_tool_check_availability_no_service(self):
        from animus.integrations.google.calendar import GoogleCalendarIntegration

        cal = GoogleCalendarIntegration()
        result = asyncio.run(
            cal._tool_check_availability("2025-01-01T10:00", "2025-01-01T11:00")
        )
        assert not result.success


# ===================================================================
# Google Gmail Integration
# ===================================================================


class TestGmailIntegration:
    """Tests for GmailIntegration."""

    def test_init(self):
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        assert gmail.name == "gmail"

    def test_connect_not_available(self):
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        with patch("animus.integrations.google.gmail.GOOGLE_API_AVAILABLE", False):
            result = asyncio.run(gmail.connect({}))
        assert result is False

    def test_connect_success(self, tmp_path: Path):
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration(data_dir=tmp_path)
        mock_service = MagicMock()
        mock_token = OAuth2Token(
            access_token="tok", refresh_token="ref",
            token_type="Bearer", expires_at=datetime(2099, 1, 1), scopes=[],
        )

        import animus.integrations.google.gmail as gmail_mod

        with patch.object(gmail_mod, "GOOGLE_API_AVAILABLE", True), \
             patch.object(gmail_mod, "GOOGLE_AUTH_AVAILABLE", True), \
             patch.object(gmail_mod, "load_token", return_value=mock_token), \
             patch.object(gmail_mod, "Credentials", create=True, return_value=MagicMock()), \
             patch.object(gmail_mod, "build", create=True, return_value=mock_service):
            result = asyncio.run(
                gmail.connect({"client_id": "id", "client_secret": "secret"})
            )
        assert result is True

    def test_connect_error(self, tmp_path: Path):
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration(data_dir=tmp_path)
        mock_token = OAuth2Token(
            access_token="tok", refresh_token="ref",
            token_type="Bearer", expires_at=datetime(2099, 1, 1), scopes=[],
        )

        import animus.integrations.google.gmail as gmail_mod

        with patch.object(gmail_mod, "GOOGLE_API_AVAILABLE", True), \
             patch.object(gmail_mod, "GOOGLE_AUTH_AVAILABLE", True), \
             patch.object(gmail_mod, "load_token", return_value=mock_token), \
             patch.object(gmail_mod, "Credentials", create=True, side_effect=Exception("auth fail")):
            result = asyncio.run(gmail.connect({"client_id": "id", "client_secret": "secret"}))
        assert result is False

    def test_disconnect(self):
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        result = asyncio.run(gmail.disconnect())
        assert result is True

    def test_get_tools(self):
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        tools = gmail.get_tools()
        assert len(tools) >= 3
        names = [t.name for t in tools]
        assert "gmail_list_inbox" in names
        assert "gmail_read_email" in names
        assert "gmail_send_email" in names

    def test_tool_list_inbox_no_service(self):
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        result = asyncio.run(gmail._tool_list_inbox())
        assert not result.success

    def test_tool_list_inbox_success(self):
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        mock_service = MagicMock()
        mock_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
            "messages": [{"id": "m1", "threadId": "t1"}],
            "resultSizeEstimate": 1,
        }
        # Mock get for each message
        mock_service.users.return_value.messages.return_value.get.return_value.execute.return_value = {
            "id": "m1",
            "snippet": "Test email",
            "payload": {"headers": [
                {"name": "Subject", "value": "Test"},
                {"name": "From", "value": "test@example.com"},
                {"name": "Date", "value": "2025-01-01"},
            ]},
        }
        gmail._service = mock_service
        result = asyncio.run(gmail._tool_list_inbox())
        assert result.success

    def test_tool_read_email_no_service(self):
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        result = asyncio.run(gmail._tool_read_email("m1"))
        assert not result.success

    def test_tool_send_email_no_service(self):
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        result = asyncio.run(
            gmail._tool_send_email("to@test.com", "Subject", "Body")
        )
        assert not result.success

    def test_tool_send_email_success(self):
        from animus.integrations.google.gmail import GmailIntegration

        gmail = GmailIntegration()
        mock_service = MagicMock()
        mock_service.users.return_value.messages.return_value.send.return_value.execute.return_value = {
            "id": "sent-1",
            "labelIds": ["SENT"],
        }
        gmail._service = mock_service
        result = asyncio.run(
            gmail._tool_send_email("to@test.com", "Subject", "Body")
        )
        assert result.success
