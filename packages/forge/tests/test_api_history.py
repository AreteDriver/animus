"""Tests for task history API endpoints."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "src")

from animus_forge.state.backends import SQLiteBackend


@pytest.fixture
def backend():
    """Create a temporary SQLite backend with task_history tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        backend = SQLiteBackend(db_path=db_path)
        yield backend
        backend.close()


@pytest.fixture
def client(backend, monkeypatch):
    """Create a test client with TaskStore initialized."""
    from animus_forge.config.settings import get_settings
    from animus_forge.state.migrations import run_migrations as actual_run_migrations

    monkeypatch.setenv("ALLOW_DEMO_AUTH", "true")
    get_settings.cache_clear()

    actual_run_migrations(backend)

    # Apply task_history migration
    migration_path = os.path.join(
        os.path.dirname(__file__), "..", "migrations", "010_task_history.sql"
    )
    with open(migration_path) as f:
        sql = f.read()
    backend.executescript(sql)

    with patch("animus_forge.api.get_database", return_value=backend):
        with patch("animus_forge.api.run_migrations", return_value=[]):
            with patch(
                "animus_forge.scheduler.schedule_manager.WorkflowEngineAdapter"
            ) as mock_sched:
                with patch(
                    "animus_forge.webhooks.webhook_manager.WorkflowEngineAdapter"
                ) as mock_wh:
                    with patch("animus_forge.jobs.job_manager.WorkflowEngineAdapter") as mock_job:
                        for m in (mock_sched, mock_wh, mock_job):
                            mock_wf = MagicMock()
                            mock_wf.variables = {}
                            m.return_value.load_workflow.return_value = mock_wf
                            r = MagicMock()
                            r.status = "completed"
                            r.errors = []
                            r.model_dump.return_value = {"status": "completed"}
                            m.return_value.execute_workflow.return_value = r

                        from animus_forge.api import app
                        from animus_forge.api_state import limiter
                        from animus_forge.security.brute_force import (
                            get_brute_force_protection,
                        )

                        limiter.enabled = False
                        protection = get_brute_force_protection()
                        protection._attempts.clear()
                        protection._total_blocked = 0
                        protection._total_allowed = 0

                        with TestClient(app) as test_client:
                            yield test_client

                        limiter.enabled = True
                        get_settings.cache_clear()


@pytest.fixture
def auth_headers(client):
    """Get authentication headers."""
    response = client.post("/v1/auth/login", json={"user_id": "test", "password": "demo"})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def seeded_client(client, auth_headers):
    """Client with some task history data seeded."""
    from animus_forge import api_state as state

    store = state.task_store
    store.record_task(
        job_id="step-1",
        workflow_id="wf-alpha",
        status="completed",
        agent_role="builder",
        model="claude-sonnet",
        total_tokens=1000,
        cost_usd=0.01,
        duration_ms=500,
    )
    store.record_task(
        job_id="step-2",
        workflow_id="wf-alpha",
        status="failed",
        agent_role="tester",
        model="gpt-4o",
        total_tokens=800,
        cost_usd=0.008,
        duration_ms=300,
        error="Test assertion failed",
    )
    store.record_task(
        job_id="step-3",
        workflow_id="wf-beta",
        status="completed",
        agent_role="builder",
        model="claude-sonnet",
        total_tokens=1200,
        cost_usd=0.012,
        duration_ms=700,
    )
    return client


# =============================================================================
# GET /v1/history
# =============================================================================


class TestListHistory:
    def test_list_returns_tasks(self, seeded_client, auth_headers):
        resp = seeded_client.get("/v1/history", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3

    def test_list_filter_by_status(self, seeded_client, auth_headers):
        resp = seeded_client.get("/v1/history", params={"status": "failed"}, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["status"] == "failed"

    def test_list_filter_by_agent_role(self, seeded_client, auth_headers):
        resp = seeded_client.get(
            "/v1/history", params={"agent_role": "builder"}, headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_list_limit_and_offset(self, seeded_client, auth_headers):
        resp = seeded_client.get(
            "/v1/history", params={"limit": 1, "offset": 0}, headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_list_empty(self, client, auth_headers):
        resp = client.get("/v1/history", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_requires_auth(self, client):
        resp = client.get("/v1/history")
        assert resp.status_code == 401


# =============================================================================
# GET /v1/history/summary
# =============================================================================


class TestHistorySummary:
    def test_summary_returns_stats(self, seeded_client, auth_headers):
        resp = seeded_client.get("/v1/history/summary", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tasks"] == 3
        assert data["successful"] == 2
        assert data["failed"] == 1
        assert data["total_tokens"] == 3000
        assert "success_rate" in data
        assert "top_agents" in data

    def test_summary_empty(self, client, auth_headers):
        resp = client.get("/v1/history/summary", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tasks"] == 0


# =============================================================================
# GET /v1/history/stats
# =============================================================================


class TestHistoryStats:
    def test_stats_all_agents(self, seeded_client, auth_headers):
        resp = seeded_client.get("/v1/history/stats", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2  # builder + tester

    def test_stats_single_agent(self, seeded_client, auth_headers):
        resp = seeded_client.get(
            "/v1/history/stats", params={"agent": "builder"}, headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["agent_role"] == "builder"
        assert data[0]["total_tasks"] == 2

    def test_stats_unknown_agent(self, seeded_client, auth_headers):
        resp = seeded_client.get(
            "/v1/history/stats",
            params={"agent": "nonexistent"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json() == []


# =============================================================================
# GET /v1/history/budget
# =============================================================================


class TestHistoryBudget:
    def test_budget_returns_rollups(self, seeded_client, auth_headers):
        resp = seeded_client.get("/v1/history/budget", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    def test_budget_custom_days(self, seeded_client, auth_headers):
        resp = seeded_client.get("/v1/history/budget", params={"days": 1}, headers=auth_headers)
        assert resp.status_code == 200

    def test_budget_filter_by_agent(self, seeded_client, auth_headers):
        resp = seeded_client.get(
            "/v1/history/budget", params={"agent": "builder"}, headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        for entry in data:
            assert entry["agent_role"] == "builder"


# =============================================================================
# GET /v1/history/{task_id}
# =============================================================================


class TestHistoryTaskDetail:
    def test_get_task(self, seeded_client, auth_headers):
        resp = seeded_client.get("/v1/history/1", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == 1
        assert data["job_id"] == "step-1"

    def test_get_task_not_found(self, seeded_client, auth_headers):
        resp = seeded_client.get("/v1/history/9999", headers=auth_headers)
        assert resp.status_code == 404
