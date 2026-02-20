"""Tests for graph workflow API endpoints and CLI commands."""

import json
import os
import tempfile
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
    """Create a temporary SQLite backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        backend = SQLiteBackend(db_path=db_path)

        schema = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        );
        """
        backend.executescript(schema)
        yield backend
        backend.close()


@pytest.fixture
def client(backend, monkeypatch):
    """Create a test client with graph routes available."""
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

            # Reset shutting_down flag
            api_state._app_state["shutting_down"] = False

            yield test_client


@pytest.fixture
def auth_headers():
    """Create auth headers with valid token."""
    token = create_access_token("test-user")
    return {"Authorization": f"Bearer {token}"}


def _simple_graph():
    """Build a minimal valid graph for testing."""
    return {
        "id": "test-graph-1",
        "name": "Test Graph",
        "nodes": [
            {
                "id": "start-1",
                "type": "start",
                "data": {},
                "position": {"x": 0, "y": 0},
            },
            {"id": "end-1", "type": "end", "data": {}, "position": {"x": 200, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "start-1", "target": "end-1"},
        ],
        "variables": {},
    }


def _graph_with_cycle():
    """Build a graph that contains a non-loop cycle."""
    return {
        "id": "cycle-graph",
        "name": "Cycle Graph",
        "nodes": [
            {"id": "a", "type": "agent", "data": {}, "position": {"x": 0, "y": 0}},
            {"id": "b", "type": "agent", "data": {}, "position": {"x": 100, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "a", "target": "b"},
            {"id": "e2", "source": "b", "target": "a"},
        ],
        "variables": {},
    }


def _disconnected_graph():
    """Build a graph with a disconnected node."""
    return {
        "id": "discon-graph",
        "name": "Disconnected",
        "nodes": [
            {
                "id": "start-1",
                "type": "start",
                "data": {},
                "position": {"x": 0, "y": 0},
            },
            {"id": "end-1", "type": "end", "data": {}, "position": {"x": 100, "y": 0}},
            {
                "id": "orphan",
                "type": "agent",
                "data": {},
                "position": {"x": 200, "y": 0},
            },
        ],
        "edges": [
            {"id": "e1", "source": "start-1", "target": "end-1"},
        ],
        "variables": {},
    }


# ---------------------------------------------------------------------------
# API Endpoint Tests
# ---------------------------------------------------------------------------


class TestGraphExecuteEndpoint:
    """Tests for POST /v1/graph/execute."""

    def test_requires_auth(self, client):
        response = client.post("/v1/graph/execute", json={"graph": _simple_graph()})
        assert response.status_code == 401

    def test_execute_simple_graph(self, client, auth_headers):
        """Execute a start->end graph and get completed status."""
        body = {"graph": _simple_graph()}

        with patch("animus_forge.api_routes.graph.state.execution_manager", new=None):
            response = client.post("/v1/graph/execute", json=body, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["workflow_id"] == "test-graph-1"
        assert "execution_id" in data

    def test_execute_with_variables(self, client, auth_headers):
        """Execute with override variables."""
        body = {
            "graph": _simple_graph(),
            "variables": {"input_text": "hello"},
        }

        with patch("animus_forge.api_routes.graph.state.execution_manager", new=None):
            response = client.post("/v1/graph/execute", json=body, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

    def test_execute_invalid_graph(self, client, auth_headers):
        """Invalid graph body returns 400."""
        body = {"graph": {"nodes": "not-a-list"}}
        response = client.post("/v1/graph/execute", json=body, headers=auth_headers)
        # Pydantic validation rejects non-list nodes
        assert response.status_code == 422

    def test_execute_cycle_graph_fails(self, client, auth_headers):
        """Graph with non-loop cycle should fail execution."""
        body = {"graph": _graph_with_cycle()}

        with patch("animus_forge.api_routes.graph.state.execution_manager", new=None):
            response = client.post("/v1/graph/execute", json=body, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "cycle" in data["error"].lower()


class TestGraphExecuteAsyncEndpoint:
    """Tests for POST /v1/graph/execute/async."""

    def test_requires_auth(self, client):
        response = client.post("/v1/graph/execute/async", json={"graph": _simple_graph()})
        assert response.status_code == 401

    def test_async_returns_execution_id(self, client, auth_headers):
        """Async execute returns immediately with a poll URL."""
        body = {"graph": _simple_graph()}

        with patch("animus_forge.api_routes.graph.state.execution_manager", new=None):
            response = client.post("/v1/graph/execute/async", json=body, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "execution_id" in data
        assert "poll_url" in data
        assert data["execution_id"] in data["poll_url"]


class TestGraphExecutionStatusEndpoint:
    """Tests for GET /v1/graph/executions/{id}."""

    def test_requires_auth(self, client):
        response = client.get("/v1/graph/executions/nonexistent")
        assert response.status_code == 401

    def test_not_found(self, client, auth_headers):
        """Getting a nonexistent execution returns 404."""
        response = client.get("/v1/graph/executions/nonexistent", headers=auth_headers)
        assert response.status_code == 404

    def test_get_after_async_submit(self, client, auth_headers):
        """Submit async then poll for status."""
        body = {"graph": _simple_graph()}

        with patch("animus_forge.api_routes.graph.state.execution_manager", new=None):
            submit_resp = client.post("/v1/graph/execute/async", json=body, headers=auth_headers)
        execution_id = submit_resp.json()["execution_id"]

        response = client.get(f"/v1/graph/executions/{execution_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["execution_id"] == execution_id


class TestGraphPauseEndpoint:
    """Tests for POST /v1/graph/executions/{id}/pause."""

    def test_requires_auth(self, client):
        response = client.post("/v1/graph/executions/x/pause")
        assert response.status_code == 401

    def test_not_found(self, client, auth_headers):
        response = client.post("/v1/graph/executions/nonexistent/pause", headers=auth_headers)
        assert response.status_code == 404

    def test_pause_non_running_fails(self, client, auth_headers):
        """Cannot pause an execution that is not running."""
        # Inject a fake completed execution
        from animus_forge.api_routes.graph import _async_executions

        _async_executions["test-done"] = {
            "execution_id": "test-done",
            "workflow_id": "w",
            "status": "completed",
            "outputs": {},
            "node_results": {},
            "total_duration_ms": 0,
            "total_tokens": 0,
            "error": None,
        }

        response = client.post("/v1/graph/executions/test-done/pause", headers=auth_headers)
        assert response.status_code == 400

        # Cleanup
        del _async_executions["test-done"]


class TestGraphResumeEndpoint:
    """Tests for POST /v1/graph/executions/{id}/resume."""

    def test_requires_auth(self, client):
        response = client.post(
            "/v1/graph/executions/x/resume",
            json=_simple_graph(),
        )
        assert response.status_code == 401

    def test_not_found(self, client, auth_headers):
        response = client.post(
            "/v1/graph/executions/nonexistent/resume",
            json=_simple_graph(),
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_resume_non_paused_fails(self, client, auth_headers):
        """Cannot resume an execution that is not paused."""
        from animus_forge.api_routes.graph import _async_executions

        _async_executions["test-running"] = {
            "execution_id": "test-running",
            "workflow_id": "w",
            "status": "running",
            "outputs": {},
            "node_results": {},
            "total_duration_ms": 0,
            "total_tokens": 0,
            "error": None,
        }

        response = client.post(
            "/v1/graph/executions/test-running/resume",
            json=_simple_graph(),
            headers=auth_headers,
        )
        assert response.status_code == 400

        del _async_executions["test-running"]


class TestGraphValidateEndpoint:
    """Tests for POST /v1/graph/validate."""

    def test_requires_auth(self, client):
        response = client.post("/v1/graph/validate", json=_simple_graph())
        assert response.status_code == 401

    def test_valid_graph(self, client, auth_headers):
        """A simple start->end graph is valid."""
        response = client.post("/v1/graph/validate", json=_simple_graph(), headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["node_count"] == 2
        assert data["edge_count"] == 1

    def test_cycle_graph_invalid(self, client, auth_headers):
        """Graph with non-loop cycle is invalid."""
        response = client.post("/v1/graph/validate", json=_graph_with_cycle(), headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        errors = [i for i in data["issues"] if i["severity"] == "error"]
        assert len(errors) > 0

    def test_disconnected_node_warning(self, client, auth_headers):
        """Disconnected node produces a warning."""
        response = client.post(
            "/v1/graph/validate",
            json=_disconnected_graph(),
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        warnings = [i for i in data["issues"] if i["severity"] == "warning"]
        # At least one warning for the orphan node
        assert any("orphan" in w.get("message", "") for w in warnings)

    def test_missing_source_node(self, client, auth_headers):
        """Edge referencing a nonexistent source is an error."""
        graph = {
            "id": "bad-edge",
            "name": "Bad Edge",
            "nodes": [
                {"id": "a", "type": "start", "data": {}, "position": {"x": 0, "y": 0}},
            ],
            "edges": [
                {"id": "e1", "source": "ghost", "target": "a"},
            ],
        }
        response = client.post("/v1/graph/validate", json=graph, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        errors = [i for i in data["issues"] if i["severity"] == "error"]
        assert any("ghost" in e["message"] for e in errors)

    def test_empty_graph_valid(self, client, auth_headers):
        """Empty graph (no nodes) is technically valid."""
        graph = {"id": "empty", "name": "Empty", "nodes": [], "edges": []}
        response = client.post("/v1/graph/validate", json=graph, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["node_count"] == 0


# ---------------------------------------------------------------------------
# CLI Command Tests
# ---------------------------------------------------------------------------


class TestGraphCLI:
    """Tests for gorgon graph CLI commands."""

    def test_validate_valid_graph(self, tmp_path):
        """Validate a valid graph from a JSON file."""
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(_simple_graph()))

        from typer.testing import CliRunner

        from animus_forge.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["graph", "validate", str(graph_file)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower() or "Valid" in result.output

    def test_validate_valid_graph_json_output(self, tmp_path):
        """Validate with --json flag."""
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(_simple_graph()))

        from typer.testing import CliRunner

        from animus_forge.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["graph", "validate", str(graph_file), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["valid"] is True

    def test_validate_invalid_cycle(self, tmp_path):
        """Validate a graph with a cycle reports errors."""
        graph_file = tmp_path / "cycle.json"
        graph_file.write_text(json.dumps(_graph_with_cycle()))

        from typer.testing import CliRunner

        from animus_forge.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["graph", "validate", str(graph_file)])
        assert result.exit_code == 1

    def test_validate_missing_file(self):
        """Missing file exits with error."""
        from typer.testing import CliRunner

        from animus_forge.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["graph", "validate", "/nonexistent/file.json"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_execute_simple_graph(self, tmp_path):
        """Execute a simple graph via CLI."""
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(_simple_graph()))

        from typer.testing import CliRunner

        from animus_forge.cli.main import app

        runner = CliRunner()

        with patch("animus_forge.cli.commands.graph.asyncio.run") as mock_run:
            # Create a mock ExecutionResult
            mock_result = MagicMock()
            mock_result.execution_id = "exec-123"
            mock_result.workflow_id = "test-graph-1"
            mock_result.status = "completed"
            mock_result.outputs = {"result": "ok"}
            mock_result.total_duration_ms = 42
            mock_result.total_tokens = 100
            mock_result.error = None
            mock_result.node_results = {}
            mock_run.return_value = mock_result

            result = runner.invoke(app, ["graph", "execute", str(graph_file)])
            assert result.exit_code == 0
            assert "completed" in result.output.lower() or "exec-123" in result.output

    def test_execute_json_output(self, tmp_path):
        """Execute with --json flag."""
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(_simple_graph()))

        from typer.testing import CliRunner

        from animus_forge.cli.main import app

        runner = CliRunner()

        with patch("animus_forge.cli.commands.graph.asyncio.run") as mock_run:
            mock_result = MagicMock()
            mock_result.execution_id = "exec-456"
            mock_result.workflow_id = "test-graph-1"
            mock_result.status = "completed"
            mock_result.outputs = {}
            mock_result.total_duration_ms = 10
            mock_result.total_tokens = 0
            mock_result.error = None
            mock_result.node_results = {}
            mock_run.return_value = mock_result

            result = runner.invoke(app, ["graph", "execute", str(graph_file), "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["status"] == "completed"

    def test_execute_dry_run(self, tmp_path):
        """Execute with --dry-run only validates."""
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(_simple_graph()))

        from typer.testing import CliRunner

        from animus_forge.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["graph", "execute", str(graph_file), "--dry-run"])
        assert result.exit_code == 0
        assert "dry run" in result.output.lower() or "valid" in result.output.lower()

    def test_execute_with_variables(self, tmp_path):
        """Execute with --var key=value."""
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(_simple_graph()))

        from typer.testing import CliRunner

        from animus_forge.cli.main import app

        runner = CliRunner()

        with patch("animus_forge.cli.commands.graph.asyncio.run") as mock_run:
            mock_result = MagicMock()
            mock_result.execution_id = "exec-789"
            mock_result.workflow_id = "test-graph-1"
            mock_result.status = "completed"
            mock_result.outputs = {}
            mock_result.total_duration_ms = 5
            mock_result.total_tokens = 0
            mock_result.error = None
            mock_result.node_results = {}
            mock_run.return_value = mock_result

            result = runner.invoke(
                app,
                ["graph", "execute", str(graph_file), "--var", "name=test"],
            )
            assert result.exit_code == 0

    def test_execute_bad_variable_format(self, tmp_path):
        """Bad --var format exits with error."""
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(_simple_graph()))

        from typer.testing import CliRunner

        from animus_forge.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["graph", "execute", str(graph_file), "--var", "badformat"],
        )
        assert result.exit_code == 1
