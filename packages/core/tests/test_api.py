"""
Tests for Animus HTTP API.

Uses httpx TestClient with mocked backends. Skips if FastAPI not installed.
"""

import pytest

try:
    from fastapi.testclient import TestClient

    import animus.api as api_module
    from animus.api import AppState, create_app

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")


@pytest.fixture
def mock_state(tmp_path):
    """Create an AppState with mocked backends."""
    from animus.cognitive import CognitiveLayer, ModelConfig
    from animus.config import AnimusConfig
    from animus.decision import DecisionFramework
    from animus.memory import MemoryLayer
    from animus.tasks import TaskTracker
    from animus.tools import create_default_registry

    config = AnimusConfig(data_dir=tmp_path)
    memory = MemoryLayer(tmp_path, backend="json")
    mock_config = ModelConfig.mock(default_response="Test response.")
    cognitive = CognitiveLayer(primary_config=mock_config)
    tools = create_default_registry()
    tasks = TaskTracker(tmp_path / "tasks")
    decisions = DecisionFramework(cognitive)

    state = AppState(
        config=config,
        memory=memory,
        cognitive=cognitive,
        tools=tools,
        tasks=tasks,
        decisions=decisions,
        conversations={},
    )
    return state


@pytest.fixture
def client(mock_state):
    """Create a test client with mocked state."""
    app = create_app()
    # Inject state
    api_module._state = mock_state
    yield TestClient(app)
    api_module._state = None


class TestStatusEndpoint:
    def test_get_status(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert "version" in data
        assert "memory_count" in data

    def test_status_no_state_returns_503(self):
        app = create_app()
        api_module._state = None
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.get("/status")
        assert resp.status_code == 503


class TestChatEndpoint:
    def test_chat_basic(self, client):
        resp = client.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Test response."
        assert "conversation_id" in data
        assert "mode_used" in data

    def test_chat_with_mode(self, client):
        resp = client.post("/chat", json={"message": "Hello", "mode": "quick"})
        assert resp.status_code == 200

    def test_chat_continue_conversation(self, client):
        resp1 = client.post("/chat", json={"message": "Hello"})
        cid = resp1.json()["conversation_id"]
        resp2 = client.post("/chat", json={"message": "Follow up", "conversation_id": cid})
        assert resp2.json()["conversation_id"] == cid


class TestMemoryEndpoints:
    def test_create_memory(self, client):
        resp = client.post(
            "/memory",
            json={
                "content": "Test memory",
                "memory_type": "semantic",
                "tags": ["test"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "Test memory"
        assert "test" in data["tags"]

    def test_create_memory_invalid_type(self, client):
        resp = client.post(
            "/memory",
            json={
                "content": "Test",
                "memory_type": "invalid_type",
            },
        )
        assert resp.status_code == 400

    def test_search_memories(self, client):
        # Create a memory first
        client.post("/memory", json={"content": "Python is great", "tags": ["dev"]})
        resp = client.get("/memory/search", params={"query": "Python"})
        assert resp.status_code == 200
        data = resp.json()
        assert "memories" in data
        assert "count" in data

    def test_get_memory_not_found(self, client):
        resp = client.get("/memory/nonexistent-id")
        assert resp.status_code == 404

    def test_delete_memory_not_found(self, client):
        resp = client.delete("/memory/nonexistent-id")
        assert resp.status_code == 404

    def test_create_and_get_memory(self, client):
        create_resp = client.post("/memory", json={"content": "Retrievable memory"})
        mem_id = create_resp.json()["id"]
        get_resp = client.get(f"/memory/{mem_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["content"] == "Retrievable memory"

    def test_create_and_delete_memory(self, client):
        create_resp = client.post("/memory", json={"content": "Delete me"})
        mem_id = create_resp.json()["id"]
        del_resp = client.delete(f"/memory/{mem_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "deleted"


class TestToolEndpoints:
    def test_list_tools(self, client):
        resp = client.get("/tools")
        assert resp.status_code == 200
        assert "tools" in resp.json()

    def test_execute_tool_get_datetime(self, client):
        resp = client.post("/tools/get_datetime", json={"params": {}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_execute_tool_not_found(self, client):
        resp = client.post("/tools/nonexistent_tool", json={"params": {}})
        assert resp.status_code == 404

    def test_execute_tool_requires_approval(self, client):
        resp = client.post("/tools/run_command", json={"params": {"command": "ls"}})
        assert resp.status_code == 403


class TestTaskEndpoints:
    def test_create_task(self, client):
        resp = client.post("/tasks", json={"description": "Test task", "tags": ["test"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["description"] == "Test task"

    def test_list_tasks(self, client):
        client.post("/tasks", json={"description": "Task 1"})
        resp = client.get("/tasks")
        assert resp.status_code == 200
        assert len(resp.json()["tasks"]) >= 1

    def test_update_task(self, client):
        create_resp = client.post("/tasks", json={"description": "Update me"})
        task_id = create_resp.json()["id"]
        update_resp = client.patch(f"/tasks/{task_id}", json={"status": "in_progress"})
        assert update_resp.status_code == 200
        assert update_resp.json()["status"] == "in_progress"

    def test_delete_task(self, client):
        create_resp = client.post("/tasks", json={"description": "Delete me"})
        task_id = create_resp.json()["id"]
        del_resp = client.delete(f"/tasks/{task_id}")
        assert del_resp.status_code == 200

    def test_update_task_not_found(self, client):
        resp = client.patch("/tasks/nonexistent", json={"status": "completed"})
        assert resp.status_code == 404

    def test_delete_task_not_found(self, client):
        resp = client.delete("/tasks/nonexistent")
        assert resp.status_code == 404


class TestDecisionEndpoint:
    def test_analyze_decision(self, client):
        resp = client.post(
            "/decide",
            json={
                "question": "Which DB?",
                "options": ["Postgres", "SQLite"],
                "criteria": ["Speed"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["question"] == "Which DB?"
        assert "recommendation" in data


class TestBriefEndpoint:
    def test_brief_no_memories(self, client):
        resp = client.get("/brief")
        assert resp.status_code == 200
        data = resp.json()
        assert "briefing" in data

    def test_brief_with_topic(self, client):
        resp = client.get("/brief", params={"topic": "work"})
        assert resp.status_code == 200


class TestIntegrationEndpoints:
    def test_list_integrations_no_manager(self, client):
        resp = client.get("/integrations")
        assert resp.status_code == 200
        assert resp.json()["connected_count"] == 0


class TestLearningEndpoints:
    def test_learning_status_not_available(self, client):
        resp = client.get("/learning/status")
        assert resp.status_code == 503

    def test_learning_items_not_available(self, client):
        resp = client.get("/learning/items")
        assert resp.status_code == 503

    def test_learning_scan_not_available(self, client):
        resp = client.post("/learning/scan")
        assert resp.status_code == 503


class TestAPIKeyAuth:
    def test_auth_required_when_configured(self, mock_state):
        mock_state.config.api.api_key = "test-secret-key"
        app = create_app()
        api_module._state = mock_state
        c = TestClient(app, raise_server_exceptions=False)

        # No key -> 401
        resp = c.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 401

        # Wrong key -> 401
        resp = c.post("/chat", json={"message": "Hello"}, headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

        # Correct key -> 200
        resp = c.post(
            "/chat",
            json={"message": "Hello"},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert resp.status_code == 200

        api_module._state = None
