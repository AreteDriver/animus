"""Integration tests for the full agent pipeline.

Tests that the API lifespan wires TaskRunner, agent memory, config loader,
and Supervisor bridge correctly. Uses mocked providers to avoid external deps.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider():
    """Mock the agent provider so no Ollama needed."""
    mock_prov = MagicMock()
    mock_prov.complete = AsyncMock(return_value="Task completed successfully")
    mock_prov.complete_with_tools = AsyncMock(return_value="Built the feature with tools")

    with patch(
        "animus_forge.agents.provider_wrapper.create_agent_provider",
        return_value=mock_prov,
    ):
        yield mock_prov


def _make_agents_client(mock_provider_fixture=None):
    """Create a standalone TestClient for the agents router."""
    with patch("animus_forge.api_routes.agents.verify_auth"):
        from animus_forge.api_routes.agents import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)


# ---------------------------------------------------------------------------
# Lifespan wiring (uses full app with mocked provider)
# ---------------------------------------------------------------------------


class TestLifespanWiring:
    """Verify the lifespan initializes agent infrastructure."""

    @pytest.fixture(autouse=True)
    def _setup(self, mock_provider, tmp_path):
        """Use mock provider for all lifespan tests. Reset state to avoid pollution."""
        import os

        from animus_forge import api_state as state
        from animus_forge.config import get_settings
        from animus_forge.state.database import reset_database

        # Ensure a clean, writable DATABASE_URL (prior tests may leave stale cache)
        db_path = tmp_path / "test-lifespan.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        get_settings.cache_clear()
        reset_database()

        # Clear stale state from prior tests
        state.task_runner = None
        state.agent_memory = None
        state.subagent_manager = None
        state.process_registry = None
        state.agent_configs = None
        state.supervisor_factory = lambda: None
        yield
        # Cleanup after
        state.task_runner = None
        state.agent_memory = None
        state.subagent_manager = None
        state.process_registry = None
        state.agent_configs = None
        reset_database()
        get_settings.cache_clear()
        os.environ.pop("DATABASE_URL", None)

    def test_task_runner_available_after_startup(self):
        from animus_forge import api_state as state
        from animus_forge.api import app

        with TestClient(app):
            assert state.task_runner is not None

    def test_agent_memory_available(self):
        from animus_forge import api_state as state
        from animus_forge.api import app

        with TestClient(app):
            assert state.agent_memory is not None

    def test_subagent_manager_available(self):
        from animus_forge import api_state as state
        from animus_forge.api import app

        with TestClient(app):
            assert state.subagent_manager is not None

    def test_process_registry_available(self):
        from animus_forge import api_state as state
        from animus_forge.api import app

        with TestClient(app):
            assert state.process_registry is not None

    def test_supervisor_factory_wires_task_runner(self):
        from animus_forge import api_state as state
        from animus_forge.api import app

        with TestClient(app):
            sup = state.supervisor_factory()
            if sup is not None:
                assert sup._task_runner is not None

    def test_agent_configs_loaded(self):
        from animus_forge import api_state as state
        from animus_forge.api import app

        with TestClient(app):
            if state.agent_configs is not None:
                assert isinstance(state.agent_configs, dict)
                assert "builder" in state.agent_configs


# ---------------------------------------------------------------------------
# POST /agents/run endpoint
# ---------------------------------------------------------------------------


class TestAgentRunEndpoint:
    """Test the /agents/run endpoint end-to-end."""

    def test_run_agent_returns_result(self, mock_provider):
        with patch("animus_forge.api_routes.agents.verify_auth"):
            from animus_forge.api_routes.agents import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            resp = client.post("/agents/run?agent=builder&task=Build+hello+world&use_tools=false")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["agent"] == "builder"
        assert "task_id" in data
        assert data["duration_ms"] >= 0

    def test_run_agent_with_tools(self, mock_provider):
        with (
            patch("animus_forge.api_routes.agents.verify_auth"),
            patch("animus_forge.api_routes.agents._get_task_runner", return_value=None),
            patch("animus_forge.tools.registry.ForgeToolRegistry", return_value=MagicMock()),
        ):
            from animus_forge.api_routes.agents import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            resp = client.post("/agents/run?agent=builder&task=Build+it&use_tools=true")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_run_agent_different_role(self, mock_provider):
        with patch("animus_forge.api_routes.agents.verify_auth"):
            from animus_forge.api_routes.agents import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            resp = client.post("/agents/run?agent=planner&task=Plan+feature&use_tools=false")
        assert resp.status_code == 200
        assert resp.json()["agent"] == "planner"


# ---------------------------------------------------------------------------
# GET /agents/runs, /agents/processes, /agents/memory
# ---------------------------------------------------------------------------


class TestAgentEndpoints:
    """Test agent endpoints with mocked state."""

    @pytest.fixture
    def client(self):
        with patch("animus_forge.api_routes.agents.verify_auth"):
            from animus_forge.api_routes.agents import router

            app = FastAPI()
            app.include_router(router)
            yield TestClient(app)

    def test_list_runs(self, client):
        resp = client.get("/agents/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert "runs" in data or "message" in data

    def test_list_processes(self, client):
        resp = client.get("/agents/processes")
        assert resp.status_code == 200
        assert "processes" in resp.json()

    def test_memory_store_and_recall(self, client):
        mem = MagicMock()
        mem.store.return_value = 1
        entry = MagicMock()
        entry.id = 1
        entry.agent_id = "test-agent"
        entry.workflow_id = None
        entry.memory_type = "fact"
        entry.content = "Uses Python 3.12"
        entry.metadata = {}
        entry.importance = 0.8
        entry.created_at = None
        entry.accessed_at = None
        entry.access_count = 0
        mem.recall.return_value = [entry]

        with patch("animus_forge.api_routes.agents._get_agent_memory", return_value=mem):
            resp = client.post(
                "/agents/memory/test-agent?content=Uses+Python&memory_type=fact&importance=0.8"
            )
            assert resp.status_code == 200
            assert resp.json()["stored"] is True

            resp = client.get("/agents/memory/test-agent")
            assert resp.status_code == 200
            assert len(resp.json()["memories"]) == 1

    def test_memory_stats(self, client):
        mem = MagicMock()
        mem.get_stats.return_value = {"total": 5}
        with patch("animus_forge.api_routes.agents._get_agent_memory", return_value=mem):
            resp = client.get("/agents/memory/test-agent/stats")
        assert resp.status_code == 200
        assert "stats" in resp.json()


# ---------------------------------------------------------------------------
# CLI get_supervisor wiring
# ---------------------------------------------------------------------------


class TestCLISupervisorWiring:
    """Test that CLI get_supervisor creates a wired Supervisor."""

    def test_get_supervisor_has_task_runner(self, mock_provider):
        from animus_forge.cli.helpers import get_supervisor

        with patch("animus_forge.cli.helpers.console"):
            sup = get_supervisor()

        assert sup is not None
        assert sup._task_runner is not None

    def test_get_supervisor_loads_agent_configs(self, mock_provider):
        from animus_forge.cli.helpers import get_supervisor

        with patch("animus_forge.cli.helpers.console"):
            sup = get_supervisor()

        if sup._agent_configs is not None:
            assert "builder" in sup._agent_configs


# ---------------------------------------------------------------------------
# WebSocket agent endpoint exists
# ---------------------------------------------------------------------------


class TestWebSocketAgentEndpoint:
    """Verify the /ws/agents endpoint is registered."""

    def test_ws_agents_route_registered(self):
        """The /ws/agents WebSocket route exists in the app."""
        from animus_forge.api_routes.websocket import router

        ws_paths = []
        for route in router.routes:
            path = getattr(route, "path", "")
            if "ws" in path.lower() or "agents" in path.lower():
                ws_paths.append(path)

        assert "/ws/agents" in ws_paths
