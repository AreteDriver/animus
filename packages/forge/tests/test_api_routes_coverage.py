"""Tests for API route modules: executions, budgets, settings, health.

Covers uncovered lines in:
- src/animus_forge/api_routes/executions.py
- src/animus_forge/api_routes/budgets.py
- src/animus_forge/api_routes/settings.py
- src/animus_forge/api_routes/health.py
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from animus_forge.auth import create_access_token
from animus_forge.state.backends import SQLiteBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend():
    """Create a temporary SQLite backend for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        be = SQLiteBackend(db_path=db_path)

        schema = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        );
        """
        be.executescript(schema)
        yield be
        be.close()


@pytest.fixture
def client(backend, monkeypatch):
    """Create a TestClient with mocked api_state managers."""
    monkeypatch.setenv("ALLOW_DEMO_AUTH", "true")

    from animus_forge.config.settings import get_settings

    get_settings.cache_clear()

    with patch("animus_forge.api.get_database", return_value=backend):
        with patch("animus_forge.api.run_migrations", return_value=[]):
            import animus_forge.api_state as api_state
            from animus_forge.api import app
            from animus_forge.api_state import limiter
            from animus_forge.security.brute_force import get_brute_force_protection

            limiter.enabled = False

            protection = get_brute_force_protection()
            protection._attempts.clear()
            protection._total_blocked = 0
            protection._total_allowed = 0

            test_client = TestClient(app)

            # Reset shutting_down flag set by previous test files
            api_state._app_state["shutting_down"] = False
            api_state._app_state["ready"] = True

            # Store references for per-test patching
            test_client._api_state = api_state

            yield test_client


@pytest.fixture
def auth_headers():
    """Create valid auth headers."""
    token = create_access_token("test-user")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Execution mock helpers
# ---------------------------------------------------------------------------


def _make_execution(
    execution_id="exec-1",
    status_value="running",
    workflow_id="wf-1",
):
    """Build a mock Execution object."""
    from animus_forge.executions import ExecutionStatus

    mock = MagicMock()
    mock.id = execution_id
    mock.workflow_id = workflow_id
    mock.workflow_name = "Test Workflow"
    mock.status = ExecutionStatus(status_value)
    mock.started_at = datetime.now()
    mock.completed_at = None
    mock.current_step = "step-1"
    mock.progress = 50
    mock.checkpoint_id = None
    mock.variables = {}
    mock.error = None
    mock.created_at = datetime.now()
    mock.logs = []
    mock.metrics = None
    mock.model_dump.return_value = {
        "id": execution_id,
        "workflow_id": workflow_id,
        "workflow_name": "Test Workflow",
        "status": status_value,
        "progress": 50,
    }
    return mock


def _make_paginated(data, total=1, page=1, page_size=20, total_pages=1):
    """Build a mock PaginatedResponse."""
    mock = MagicMock()
    mock.data = data
    mock.total = total
    mock.page = page
    mock.page_size = page_size
    mock.total_pages = total_pages
    return mock


def _make_log_entry(execution_id="exec-1"):
    """Build a mock ExecutionLog."""
    mock = MagicMock()
    mock.execution_id = execution_id
    mock.model_dump.return_value = {
        "id": 1,
        "execution_id": execution_id,
        "level": "info",
        "message": "Step started",
    }
    return mock


def _make_metrics(execution_id="exec-1"):
    """Build a mock ExecutionMetrics."""
    mock = MagicMock()
    mock.execution_id = execution_id
    mock.model_dump.return_value = {
        "execution_id": execution_id,
        "total_tokens": 100,
        "total_cost_cents": 5,
    }
    return mock


# ---------------------------------------------------------------------------
# Execution endpoint tests
# ---------------------------------------------------------------------------


class TestExecutionEndpoints:
    """Tests for /v1/executions endpoints."""

    def test_list_executions_requires_auth(self, client):
        response = client.get("/v1/executions")
        assert response.status_code == 401

    def test_list_executions_success(self, client, auth_headers):
        exec_mock = _make_execution()
        paginated = _make_paginated([exec_mock])

        mock_mgr = MagicMock()
        mock_mgr.list_executions.return_value = paginated

        client._api_state.execution_manager = mock_mgr

        response = client.get("/v1/executions", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["total"] == 1
        assert data["page"] == 1

    def test_list_executions_with_status_filter(self, client, auth_headers):
        paginated = _make_paginated([], total=0)
        mock_mgr = MagicMock()
        mock_mgr.list_executions.return_value = paginated

        client._api_state.execution_manager = mock_mgr

        response = client.get("/v1/executions?status=running", headers=auth_headers)
        assert response.status_code == 200

    def test_list_executions_invalid_status(self, client, auth_headers):
        mock_mgr = MagicMock()
        client._api_state.execution_manager = mock_mgr

        response = client.get("/v1/executions?status=bogus", headers=auth_headers)
        assert response.status_code == 400

    def test_list_executions_with_workflow_filter(self, client, auth_headers):
        paginated = _make_paginated([], total=0)
        mock_mgr = MagicMock()
        mock_mgr.list_executions.return_value = paginated

        client._api_state.execution_manager = mock_mgr

        response = client.get("/v1/executions?workflow_id=wf-1", headers=auth_headers)
        assert response.status_code == 200

    def test_get_execution_success(self, client, auth_headers):
        exec_mock = _make_execution()
        metrics = _make_metrics()

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.get_logs.return_value = [_make_log_entry()]
        mock_mgr.get_metrics.return_value = metrics

        client._api_state.execution_manager = mock_mgr

        response = client.get("/v1/executions/exec-1", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["id"] == "exec-1"

    def test_get_execution_not_found(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = None

        client._api_state.execution_manager = mock_mgr

        response = client.get("/v1/executions/nope", headers=auth_headers)
        assert response.status_code == 404

    def test_get_execution_logs_success(self, client, auth_headers):
        exec_mock = _make_execution()
        log_entry = _make_log_entry()

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.get_logs.return_value = [log_entry]

        client._api_state.execution_manager = mock_mgr

        response = client.get("/v1/executions/exec-1/logs", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_execution_logs_with_level_filter(self, client, auth_headers):
        exec_mock = _make_execution()

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.get_logs.return_value = []

        client._api_state.execution_manager = mock_mgr

        response = client.get("/v1/executions/exec-1/logs?level=error", headers=auth_headers)
        assert response.status_code == 200

    def test_get_execution_logs_invalid_level(self, client, auth_headers):
        exec_mock = _make_execution()

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock

        client._api_state.execution_manager = mock_mgr

        response = client.get("/v1/executions/exec-1/logs?level=bogus", headers=auth_headers)
        assert response.status_code == 400

    def test_get_execution_logs_not_found(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = None

        client._api_state.execution_manager = mock_mgr

        response = client.get("/v1/executions/nope/logs", headers=auth_headers)
        assert response.status_code == 404

    def test_pause_execution_success(self, client, auth_headers):
        exec_mock = _make_execution(status_value="running")
        updated = _make_execution(status_value="paused")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.pause_execution.return_value = updated

        client._api_state.execution_manager = mock_mgr

        response = client.post("/v1/executions/exec-1/pause", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_pause_execution_not_found(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = None

        client._api_state.execution_manager = mock_mgr

        response = client.post("/v1/executions/nope/pause", headers=auth_headers)
        assert response.status_code == 404

    def test_pause_execution_wrong_status(self, client, auth_headers):
        exec_mock = _make_execution(status_value="completed")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock

        client._api_state.execution_manager = mock_mgr

        response = client.post("/v1/executions/exec-1/pause", headers=auth_headers)
        assert response.status_code == 400

    def test_pause_execution_returns_none(self, client, auth_headers):
        """Pause succeeds but pause_execution returns None (unknown status)."""
        exec_mock = _make_execution(status_value="running")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.pause_execution.return_value = None

        client._api_state.execution_manager = mock_mgr

        response = client.post("/v1/executions/exec-1/pause", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["execution_status"] == "unknown"

    def test_resume_execution_success(self, client, auth_headers):
        exec_mock = _make_execution(status_value="paused")
        updated = _make_execution(status_value="running")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.resume_execution.return_value = updated

        client._api_state.execution_manager = mock_mgr

        response = client.post("/v1/executions/exec-1/resume", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_resume_execution_not_found(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = None

        client._api_state.execution_manager = mock_mgr

        response = client.post("/v1/executions/nope/resume", headers=auth_headers)
        assert response.status_code == 404

    def test_resume_execution_wrong_status(self, client, auth_headers):
        exec_mock = _make_execution(status_value="running")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock

        client._api_state.execution_manager = mock_mgr

        response = client.post("/v1/executions/exec-1/resume", headers=auth_headers)
        assert response.status_code == 400

    def test_resume_execution_returns_none(self, client, auth_headers):
        """Resume succeeds but resume_execution returns None."""
        exec_mock = _make_execution(status_value="paused")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.resume_execution.return_value = None

        client._api_state.execution_manager = mock_mgr

        response = client.post("/v1/executions/exec-1/resume", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["execution_status"] == "unknown"

    def test_cancel_execution_success(self, client, auth_headers):
        exec_mock = _make_execution(status_value="running")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.cancel_execution.return_value = True

        client._api_state.execution_manager = mock_mgr

        response = client.post("/v1/executions/exec-1/cancel", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["message"] == "Execution cancelled"

    def test_cancel_execution_not_found(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = None

        client._api_state.execution_manager = mock_mgr

        response = client.post("/v1/executions/nope/cancel", headers=auth_headers)
        assert response.status_code == 404

    def test_cancel_execution_fails(self, client, auth_headers):
        exec_mock = _make_execution(status_value="completed")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.cancel_execution.return_value = False

        client._api_state.execution_manager = mock_mgr

        response = client.post("/v1/executions/exec-1/cancel", headers=auth_headers)
        assert response.status_code == 400

    def test_delete_execution_success(self, client, auth_headers):
        exec_mock = _make_execution(status_value="completed")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.delete_execution.return_value = True

        client._api_state.execution_manager = mock_mgr

        response = client.delete("/v1/executions/exec-1", headers=auth_headers)
        assert response.status_code == 200

    def test_delete_execution_not_found(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = None

        client._api_state.execution_manager = mock_mgr

        response = client.delete("/v1/executions/nope", headers=auth_headers)
        assert response.status_code == 404

    def test_delete_execution_active_status(self, client, auth_headers):
        """Cannot delete running/pending/paused executions."""
        exec_mock = _make_execution(status_value="running")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock

        client._api_state.execution_manager = mock_mgr

        response = client.delete("/v1/executions/exec-1", headers=auth_headers)
        assert response.status_code == 400

    def test_delete_execution_pending_status(self, client, auth_headers):
        exec_mock = _make_execution(status_value="pending")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock

        client._api_state.execution_manager = mock_mgr

        response = client.delete("/v1/executions/exec-1", headers=auth_headers)
        assert response.status_code == 400

    def test_delete_execution_paused_status(self, client, auth_headers):
        exec_mock = _make_execution(status_value="paused")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock

        client._api_state.execution_manager = mock_mgr

        response = client.delete("/v1/executions/exec-1", headers=auth_headers)
        assert response.status_code == 400

    def test_delete_execution_failed_status(self, client, auth_headers):
        """Can delete failed executions."""
        exec_mock = _make_execution(status_value="failed")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.delete_execution.return_value = True

        client._api_state.execution_manager = mock_mgr

        response = client.delete("/v1/executions/exec-1", headers=auth_headers)
        assert response.status_code == 200

    def test_delete_execution_cancelled_status(self, client, auth_headers):
        """Can delete cancelled executions."""
        exec_mock = _make_execution(status_value="cancelled")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.delete_execution.return_value = True

        client._api_state.execution_manager = mock_mgr

        response = client.delete("/v1/executions/exec-1", headers=auth_headers)
        assert response.status_code == 200

    def test_delete_execution_internal_error(self, client, auth_headers):
        """delete_execution returns False -> 500."""
        exec_mock = _make_execution(status_value="completed")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.delete_execution.return_value = False

        client._api_state.execution_manager = mock_mgr

        response = client.delete("/v1/executions/exec-1", headers=auth_headers)
        assert response.status_code == 500

    def test_cleanup_executions_success(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.cleanup_old_executions.return_value = 5

        client._api_state.execution_manager = mock_mgr

        response = client.post("/v1/executions/cleanup", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["deleted"] == 5

    def test_cleanup_executions_custom_hours(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.cleanup_old_executions.return_value = 0

        client._api_state.execution_manager = mock_mgr

        response = client.post("/v1/executions/cleanup?max_age_hours=24", headers=auth_headers)
        assert response.status_code == 200
        mock_mgr.cleanup_old_executions.assert_called_with(24)

    def test_cleanup_executions_requires_auth(self, client):
        response = client.post("/v1/executions/cleanup")
        assert response.status_code == 401

    def test_stream_execution_not_found(self, client, auth_headers):
        """SSE stream returns 404 for missing execution."""
        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = None

        client._api_state.execution_manager = mock_mgr

        response = client.get("/v1/executions/nope/stream", headers=auth_headers)
        assert response.status_code == 404

    def test_stream_execution_completed(self, client, auth_headers):
        """SSE stream for already-completed execution sends snapshot+done."""
        exec_mock = _make_execution(status_value="completed")
        log_entry = _make_log_entry()
        metrics_mock = _make_metrics()

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.get_logs.return_value = [log_entry]
        mock_mgr.get_metrics.return_value = metrics_mock
        mock_mgr.register_callback = MagicMock()
        mock_mgr.unregister_callback = MagicMock()

        client._api_state.execution_manager = mock_mgr

        response = client.get("/v1/executions/exec-1/stream", headers=auth_headers)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = response.text
        assert "event: snapshot" in body
        assert "event: done" in body

    def test_stream_execution_no_metrics(self, client, auth_headers):
        """SSE stream for completed execution with no metrics."""
        exec_mock = _make_execution(status_value="failed")

        mock_mgr = MagicMock()
        mock_mgr.get_execution.return_value = exec_mock
        mock_mgr.get_logs.return_value = []
        mock_mgr.get_metrics.return_value = None
        mock_mgr.register_callback = MagicMock()
        mock_mgr.unregister_callback = MagicMock()

        client._api_state.execution_manager = mock_mgr

        response = client.get("/v1/executions/exec-1/stream", headers=auth_headers)
        assert response.status_code == 200
        body = response.text
        assert "event: snapshot" in body
        assert "event: done" in body


# ---------------------------------------------------------------------------
# Budget mock helpers
# ---------------------------------------------------------------------------


def _make_budget(budget_id="bud-1", name="Test Budget", period="monthly"):
    """Build a mock Budget object."""
    mock = MagicMock()
    mock.id = budget_id
    mock.name = name
    mock.total_amount = 100.0
    mock.used_amount = 25.0
    mock.period = period
    mock.agent_id = None
    mock.model_dump.return_value = {
        "id": budget_id,
        "name": name,
        "total_amount": 100.0,
        "used_amount": 25.0,
        "period": period,
    }
    return mock


def _make_budget_summary():
    """Build a mock BudgetSummary object."""
    mock = MagicMock()
    mock.model_dump.return_value = {
        "total_budget": 500.0,
        "total_used": 100.0,
        "total_remaining": 400.0,
        "percent_used": 20.0,
        "budget_count": 3,
        "exceeded_count": 0,
        "warning_count": 0,
    }
    return mock


# ---------------------------------------------------------------------------
# Budget endpoint tests
# ---------------------------------------------------------------------------


class TestBudgetEndpoints:
    """Tests for /v1/budgets endpoints."""

    def test_list_budgets_requires_auth(self, client):
        response = client.get("/v1/budgets")
        assert response.status_code == 401

    def test_list_budgets_success(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.list_budgets.return_value = [_make_budget()]

        client._api_state.budget_manager = mock_mgr

        response = client.get("/v1/budgets", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) == 1

    def test_list_budgets_with_period_filter(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.list_budgets.return_value = []

        client._api_state.budget_manager = mock_mgr

        response = client.get("/v1/budgets?period=daily", headers=auth_headers)
        assert response.status_code == 200

    def test_list_budgets_with_agent_filter(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.list_budgets.return_value = []

        client._api_state.budget_manager = mock_mgr

        response = client.get("/v1/budgets?agent_id=agent-1", headers=auth_headers)
        assert response.status_code == 200

    def test_list_budgets_invalid_period(self, client, auth_headers):
        mock_mgr = MagicMock()
        client._api_state.budget_manager = mock_mgr

        response = client.get("/v1/budgets?period=hourly", headers=auth_headers)
        assert response.status_code == 400

    def test_get_budget_summary(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.get_summary.return_value = _make_budget_summary()

        client._api_state.budget_manager = mock_mgr

        response = client.get("/v1/budgets/summary", headers=auth_headers)
        assert response.status_code == 200
        assert "total_budget" in response.json()

    def test_get_budget_success(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.get_budget.return_value = _make_budget()

        client._api_state.budget_manager = mock_mgr

        response = client.get("/v1/budgets/bud-1", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["id"] == "bud-1"

    def test_get_budget_not_found(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.get_budget.return_value = None

        client._api_state.budget_manager = mock_mgr

        response = client.get("/v1/budgets/nope", headers=auth_headers)
        assert response.status_code == 404

    def test_create_budget_success(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.create_budget.return_value = _make_budget()

        client._api_state.budget_manager = mock_mgr

        response = client.post(
            "/v1/budgets",
            json={"name": "Test", "total_amount": 100, "period": "monthly"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_create_budget_invalid_period(self, client, auth_headers):
        mock_mgr = MagicMock()
        client._api_state.budget_manager = mock_mgr

        response = client.post(
            "/v1/budgets",
            json={"name": "Test", "total_amount": 100, "period": "hourly"},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_create_budget_empty_name(self, client, auth_headers):
        mock_mgr = MagicMock()
        client._api_state.budget_manager = mock_mgr

        response = client.post(
            "/v1/budgets",
            json={"name": "", "total_amount": 100, "period": "daily"},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_create_budget_negative_amount(self, client, auth_headers):
        mock_mgr = MagicMock()
        client._api_state.budget_manager = mock_mgr

        response = client.post(
            "/v1/budgets",
            json={"name": "Test", "total_amount": -10, "period": "daily"},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_create_budget_value_error(self, client, auth_headers):
        """Manager raises ValueError during creation."""
        mock_mgr = MagicMock()
        mock_mgr.create_budget.side_effect = ValueError("Duplicate name")

        client._api_state.budget_manager = mock_mgr

        response = client.post(
            "/v1/budgets",
            json={"name": "Dupe", "total_amount": 100, "period": "daily"},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_create_budget_with_agent_id(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.create_budget.return_value = _make_budget()

        client._api_state.budget_manager = mock_mgr

        response = client.post(
            "/v1/budgets",
            json={
                "name": "Agent Budget",
                "total_amount": 50,
                "period": "weekly",
                "agent_id": "agent-1",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_update_budget_success(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.update_budget.return_value = _make_budget(name="Updated")

        client._api_state.budget_manager = mock_mgr

        response = client.patch(
            "/v1/budgets/bud-1",
            json={"name": "Updated"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_update_budget_not_found(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.update_budget.return_value = None

        client._api_state.budget_manager = mock_mgr

        response = client.patch(
            "/v1/budgets/nope",
            json={"name": "Updated"},
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_update_budget_invalid_period(self, client, auth_headers):
        mock_mgr = MagicMock()
        client._api_state.budget_manager = mock_mgr

        response = client.patch(
            "/v1/budgets/bud-1",
            json={"period": "hourly"},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_update_budget_negative_total(self, client, auth_headers):
        mock_mgr = MagicMock()
        client._api_state.budget_manager = mock_mgr

        response = client.patch(
            "/v1/budgets/bud-1",
            json={"total_amount": -5},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_update_budget_negative_used(self, client, auth_headers):
        mock_mgr = MagicMock()
        client._api_state.budget_manager = mock_mgr

        response = client.patch(
            "/v1/budgets/bud-1",
            json={"used_amount": -1},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_update_budget_with_valid_period(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.update_budget.return_value = _make_budget(period="weekly")

        client._api_state.budget_manager = mock_mgr

        response = client.patch(
            "/v1/budgets/bud-1",
            json={"period": "weekly"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_delete_budget_success(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.delete_budget.return_value = True

        client._api_state.budget_manager = mock_mgr

        response = client.delete("/v1/budgets/bud-1", headers=auth_headers)
        assert response.status_code == 200

    def test_delete_budget_not_found(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.delete_budget.return_value = False

        client._api_state.budget_manager = mock_mgr

        response = client.delete("/v1/budgets/nope", headers=auth_headers)
        assert response.status_code == 404

    def test_add_budget_usage_success(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.add_usage.return_value = _make_budget()

        client._api_state.budget_manager = mock_mgr

        response = client.post("/v1/budgets/bud-1/add-usage?amount=10.5", headers=auth_headers)
        assert response.status_code == 200

    def test_add_budget_usage_not_found(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.add_usage.return_value = None

        client._api_state.budget_manager = mock_mgr

        response = client.post("/v1/budgets/nope/add-usage?amount=5", headers=auth_headers)
        assert response.status_code == 404

    def test_reset_budget_usage_success(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.reset_usage.return_value = _make_budget()

        client._api_state.budget_manager = mock_mgr

        response = client.post("/v1/budgets/bud-1/reset", headers=auth_headers)
        assert response.status_code == 200

    def test_reset_budget_usage_not_found(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.reset_usage.return_value = None

        client._api_state.budget_manager = mock_mgr

        response = client.post("/v1/budgets/nope/reset", headers=auth_headers)
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Settings mock helpers
# ---------------------------------------------------------------------------


def _make_preferences(user_id="test-user"):
    """Build a mock UserPreferences object."""
    mock = MagicMock()
    mock.user_id = user_id
    mock.theme = "dark"
    mock.compact_view = False
    mock.show_costs = True
    mock.default_page_size = 20
    mock.model_dump.return_value = {
        "user_id": user_id,
        "theme": "dark",
        "compact_view": False,
        "show_costs": True,
        "default_page_size": 20,
        "notifications": {
            "execution_complete": True,
            "execution_failed": True,
            "budget_alert": True,
        },
    }
    return mock


def _make_api_key_info(provider="openai"):
    """Build a mock APIKeyInfo object."""
    mock = MagicMock()
    mock.id = 1
    mock.provider = provider
    mock.key_prefix = "sk-...abc"
    mock.model_dump.return_value = {
        "id": 1,
        "provider": provider,
        "key_prefix": "sk-...abc",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
    }
    return mock


def _make_api_key_status():
    """Build a mock APIKeyStatus object."""
    mock = MagicMock()
    mock.model_dump.return_value = {
        "openai": True,
        "anthropic": False,
        "github": False,
    }
    return mock


# ---------------------------------------------------------------------------
# Settings endpoint tests
# ---------------------------------------------------------------------------


class TestSettingsEndpoints:
    """Tests for /v1/settings endpoints."""

    def test_get_preferences_requires_auth(self, client):
        response = client.get("/v1/settings/preferences")
        assert response.status_code == 401

    def test_get_preferences_success(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.get_preferences.return_value = _make_preferences()

        client._api_state.settings_manager = mock_mgr

        response = client.get("/v1/settings/preferences", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["theme"] == "dark"

    def test_update_preferences_theme(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.update_preferences.return_value = _make_preferences()

        client._api_state.settings_manager = mock_mgr

        response = client.post(
            "/v1/settings/preferences",
            json={"theme": "dark"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_update_preferences_invalid_theme(self, client, auth_headers):
        mock_mgr = MagicMock()
        client._api_state.settings_manager = mock_mgr

        response = client.post(
            "/v1/settings/preferences",
            json={"theme": "neon"},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_update_preferences_compact_view(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.update_preferences.return_value = _make_preferences()

        client._api_state.settings_manager = mock_mgr

        response = client.post(
            "/v1/settings/preferences",
            json={"compact_view": True},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_update_preferences_show_costs(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.update_preferences.return_value = _make_preferences()

        client._api_state.settings_manager = mock_mgr

        response = client.post(
            "/v1/settings/preferences",
            json={"show_costs": False},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_update_preferences_page_size(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.update_preferences.return_value = _make_preferences()

        client._api_state.settings_manager = mock_mgr

        response = client.post(
            "/v1/settings/preferences",
            json={"default_page_size": 50},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_update_preferences_page_size_too_small(self, client, auth_headers):
        mock_mgr = MagicMock()
        client._api_state.settings_manager = mock_mgr

        response = client.post(
            "/v1/settings/preferences",
            json={"default_page_size": 5},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_update_preferences_page_size_too_large(self, client, auth_headers):
        mock_mgr = MagicMock()
        client._api_state.settings_manager = mock_mgr

        response = client.post(
            "/v1/settings/preferences",
            json={"default_page_size": 200},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_update_preferences_notifications(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.update_preferences.return_value = _make_preferences()

        client._api_state.settings_manager = mock_mgr

        response = client.post(
            "/v1/settings/preferences",
            json={
                "notifications": {
                    "execution_complete": False,
                    "execution_failed": True,
                    "budget_alert": False,
                }
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_api_keys_success(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.get_api_keys.return_value = [_make_api_key_info()]

        client._api_state.settings_manager = mock_mgr

        response = client.get("/v1/settings/api-keys", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_api_key_status(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.get_api_key_status.return_value = _make_api_key_status()

        client._api_state.settings_manager = mock_mgr

        response = client.get("/v1/settings/api-keys/status", headers=auth_headers)
        assert response.status_code == 200
        assert "openai" in response.json()

    def test_set_api_key_success(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.set_api_key.return_value = _make_api_key_info()

        client._api_state.settings_manager = mock_mgr

        response = client.post(
            "/v1/settings/api-keys",
            json={"provider": "openai", "key": "sk-test-key-12345"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_set_api_key_invalid_provider(self, client, auth_headers):
        mock_mgr = MagicMock()
        client._api_state.settings_manager = mock_mgr

        response = client.post(
            "/v1/settings/api-keys",
            json={"provider": "azure", "key": "sk-test-key-12345"},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_set_api_key_too_short(self, client, auth_headers):
        mock_mgr = MagicMock()
        client._api_state.settings_manager = mock_mgr

        response = client.post(
            "/v1/settings/api-keys",
            json={"provider": "openai", "key": "short"},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_set_api_key_anthropic(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.set_api_key.return_value = _make_api_key_info("anthropic")

        client._api_state.settings_manager = mock_mgr

        response = client.post(
            "/v1/settings/api-keys",
            json={"provider": "anthropic", "key": "sk-ant-1234567890"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_set_api_key_github(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.set_api_key.return_value = _make_api_key_info("github")

        client._api_state.settings_manager = mock_mgr

        response = client.post(
            "/v1/settings/api-keys",
            json={"provider": "github", "key": "ghp_1234567890"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_delete_api_key_success(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.delete_api_key.return_value = True

        client._api_state.settings_manager = mock_mgr

        response = client.delete("/v1/settings/api-keys/openai", headers=auth_headers)
        assert response.status_code == 200

    def test_delete_api_key_not_found(self, client, auth_headers):
        mock_mgr = MagicMock()
        mock_mgr.delete_api_key.return_value = False

        client._api_state.settings_manager = mock_mgr

        response = client.delete("/v1/settings/api-keys/openai", headers=auth_headers)
        assert response.status_code == 404

    def test_delete_api_key_invalid_provider(self, client, auth_headers):
        mock_mgr = MagicMock()
        client._api_state.settings_manager = mock_mgr

        response = client.delete("/v1/settings/api-keys/azure", headers=auth_headers)
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    """Tests for health check endpoints (no /v1 prefix)."""

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["app"] == "AI Workflow Orchestrator"
        assert "version" in data

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "timestamp" in response.json()

    def test_liveness_check(self, client):
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_check_ready(self, client):
        client._api_state._app_state["ready"] = True
        client._api_state._app_state["shutting_down"] = False

        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_readiness_check_not_ready(self, client):
        client._api_state._app_state["ready"] = False

        response = client.get("/health/ready")
        assert response.status_code == 503

    def test_readiness_check_shutting_down(self, client):
        client._api_state._app_state["ready"] = True
        client._api_state._app_state["shutting_down"] = True

        response = client.get("/health/ready")
        assert response.status_code == 503

        # Reset for other tests
        client._api_state._app_state["shutting_down"] = False

    def test_database_health_check_sqlite(self, client, backend):
        with patch("animus_forge.api_routes.health.get_database", return_value=backend):
            with patch(
                "animus_forge.api_routes.health.get_migration_status",
                return_value={"applied": 5, "pending": 0},
            ):
                response = client.get("/health/db")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["backend"] == "sqlite"
                assert data["database"] == "connected"

    def test_database_health_check_postgres(self, client):
        from animus_forge.state.backends import PostgresBackend

        # Create a mock that passes isinstance(x, PostgresBackend)
        mock_backend = MagicMock(spec=PostgresBackend)
        mock_backend.fetchone.return_value = {"ping": 1}

        with patch("animus_forge.api_routes.health.get_database", return_value=mock_backend):
            with patch(
                "animus_forge.api_routes.health.get_migration_status",
                return_value={"applied": 3},
            ):
                response = client.get("/health/db")
                assert response.status_code == 200
                assert response.json()["backend"] == "postgresql"

    def test_database_health_check_unknown_backend(self, client):
        """Unknown backend type."""
        mock_backend = MagicMock(spec=["fetchone"])
        mock_backend.fetchone.return_value = {"ping": 1}

        with patch("animus_forge.api_routes.health.get_database", return_value=mock_backend):
            with patch("animus_forge.api_routes.health.PostgresBackend", new=type("PG", (), {})):
                with patch(
                    "animus_forge.api_routes.health.SQLiteBackend",
                    new=type("SL", (), {}),
                ):
                    with patch(
                        "animus_forge.api_routes.health.get_migration_status",
                        return_value={},
                    ):
                        response = client.get("/health/db")
                        assert response.status_code == 200
                        assert response.json()["backend"] == "unknown"

    def test_database_health_check_failure(self, client):
        with patch(
            "animus_forge.api_routes.health.get_database",
            side_effect=Exception("Connection refused"),
        ):
            response = client.get("/health/db")
            assert response.status_code == 503
            assert response.json()["detail"]["status"] == "unhealthy"

    def test_full_health_check(self, client, backend):
        client._api_state._app_state["ready"] = True
        client._api_state._app_state["shutting_down"] = False
        client._api_state._app_state["start_time"] = datetime(2024, 1, 1)

        with patch("animus_forge.api_routes.health.get_database", return_value=backend):
            with patch(
                "animus_forge.api_routes.health.get_all_circuit_stats",
                return_value={},
            ):
                with patch(
                    "animus_forge.api_routes.health.get_all_provider_stats",
                    return_value={},
                ):
                    response = client.get("/health/full")
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "healthy"
                    assert data["uptime_seconds"] is not None
                    assert data["uptime_seconds"] > 0
                    assert data["database"]["status"] == "connected"

    def test_full_health_check_no_start_time(self, client, backend):
        client._api_state._app_state["ready"] = True
        client._api_state._app_state["shutting_down"] = False
        client._api_state._app_state["start_time"] = None

        with patch("animus_forge.api_routes.health.get_database", return_value=backend):
            with patch(
                "animus_forge.api_routes.health.get_all_circuit_stats",
                return_value={},
            ):
                with patch(
                    "animus_forge.api_routes.health.get_all_provider_stats",
                    return_value={},
                ):
                    response = client.get("/health/full")
                    assert response.status_code == 200
                    data = response.json()
                    assert data["uptime_seconds"] is None

    def test_full_health_check_db_failure(self, client):
        client._api_state._app_state["ready"] = True
        client._api_state._app_state["shutting_down"] = False
        client._api_state._app_state["start_time"] = None

        with patch(
            "animus_forge.api_routes.health.get_database",
            side_effect=Exception("DB down"),
        ):
            with patch(
                "animus_forge.api_routes.health.get_all_circuit_stats",
                return_value={},
            ):
                with patch(
                    "animus_forge.api_routes.health.get_all_provider_stats",
                    return_value={},
                ):
                    response = client.get("/health/full")
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "degraded"
                    assert data["database"]["status"] == "disconnected"

    def test_full_health_check_circuit_breaker_open(self, client, backend):
        """Open circuit breaker degrades health status."""
        client._api_state._app_state["ready"] = True
        client._api_state._app_state["shutting_down"] = False
        client._api_state._app_state["start_time"] = None

        with patch("animus_forge.api_routes.health.get_database", return_value=backend):
            with patch(
                "animus_forge.api_routes.health.get_all_circuit_stats",
                return_value={
                    "openai": {
                        "state": "open",
                        "failure_count": 5,
                    }
                },
            ):
                with patch(
                    "animus_forge.api_routes.health.get_all_provider_stats",
                    return_value={},
                ):
                    response = client.get("/health/full")
                    assert response.status_code == 200
                    assert response.json()["status"] == "degraded"

    def test_full_health_check_shutting_down(self, client, backend):
        """Shutting down overrides health status."""
        client._api_state._app_state["ready"] = True
        client._api_state._app_state["shutting_down"] = True
        client._api_state._app_state["start_time"] = None

        with patch("animus_forge.api_routes.health.get_database", return_value=backend):
            with patch(
                "animus_forge.api_routes.health.get_all_circuit_stats",
                return_value={},
            ):
                with patch(
                    "animus_forge.api_routes.health.get_all_provider_stats",
                    return_value={},
                ):
                    response = client.get("/health/full")
                    assert response.status_code == 200
                    assert response.json()["status"] == "shutting_down"

        # Reset for other tests
        client._api_state._app_state["shutting_down"] = False

    def test_metrics_endpoint(self, client):
        client._api_state._app_state["ready"] = True
        client._api_state._app_state["shutting_down"] = False
        client._api_state._app_state["start_time"] = datetime(2024, 1, 1)

        mock_collector = MagicMock()
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = "# HELP test\n# TYPE test gauge\ntest 1"

        with patch(
            "animus_forge.api_routes.health.get_all_circuit_stats",
            return_value={
                "openai": {"state": "closed", "failure_count": 0},
            },
        ):
            with patch("animus_forge.metrics.get_collector", return_value=mock_collector):
                with patch(
                    "animus_forge.metrics.PrometheusExporter",
                    return_value=mock_exporter,
                ):
                    response = client.get("/metrics")
                    assert response.status_code == 200
                    body = response.text
                    assert "gorgon_app_ready" in body
                    assert "gorgon_app_shutting_down" in body
                    assert "gorgon_active_requests" in body
                    assert "gorgon_uptime_seconds" in body
                    assert "gorgon_circuit_breaker_openai_state" in body
                    assert "gorgon_circuit_breaker_openai_failures" in body

    def test_metrics_endpoint_no_start_time(self, client):
        """Metrics endpoint without start_time skips uptime."""
        client._api_state._app_state["ready"] = False
        client._api_state._app_state["shutting_down"] = False
        client._api_state._app_state["start_time"] = None

        mock_collector = MagicMock()
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = ""

        with patch(
            "animus_forge.api_routes.health.get_all_circuit_stats",
            return_value={},
        ):
            with patch("animus_forge.metrics.get_collector", return_value=mock_collector):
                with patch(
                    "animus_forge.metrics.PrometheusExporter",
                    return_value=mock_exporter,
                ):
                    response = client.get("/metrics")
                    assert response.status_code == 200
                    body = response.text
                    assert "gorgon_uptime_seconds" not in body

    def test_websocket_stats_no_manager(self, client):
        client._api_state.ws_manager = None

        response = client.get("/ws/stats")
        assert response.status_code == 200
        assert response.json()["error"] == "WebSocket not initialized"

    def test_websocket_stats_with_manager(self, client):
        mock_ws = MagicMock()
        mock_ws.get_stats.return_value = {
            "connections": 2,
            "messages_sent": 100,
        }

        client._api_state.ws_manager = mock_ws

        response = client.get("/ws/stats")
        assert response.status_code == 200
        assert response.json()["connections"] == 2
