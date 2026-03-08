"""Tests for the agents API routes (process registry + agent memory)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with auth disabled."""
    with patch("animus_forge.api_routes.agents.verify_auth"):
        from fastapi import FastAPI

        from animus_forge.api_routes.agents import router

        app = FastAPI()
        app.include_router(router)
        yield TestClient(app)


# ===========================================================================
# Process registry endpoints
# ===========================================================================


class TestListProcesses:
    """Test GET /agents/processes."""

    def test_no_registry(self, client):
        with patch("animus_forge.api_routes.agents._get_process_registry", return_value=None):
            resp = client.get("/agents/processes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["processes"] == []
        assert "not available" in data["message"]

    def test_empty_registry(self, client):
        registry = MagicMock()
        registry.list_all.return_value = []
        with patch("animus_forge.api_routes.agents._get_process_registry", return_value=registry):
            resp = client.get("/agents/processes")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_with_processes(self, client):
        entry = MagicMock()
        entry.to_dict.return_value = {"id": "p1", "type": "agent", "status": "running"}
        registry = MagicMock()
        registry.list_all.return_value = [entry]
        with patch("animus_forge.api_routes.agents._get_process_registry", return_value=registry):
            resp = client.get("/agents/processes")
        data = resp.json()
        assert data["total"] == 1
        assert data["processes"][0]["id"] == "p1"

    def test_invalid_filter(self, client):
        registry = MagicMock()
        registry.list_all.side_effect = ValueError("bad type")
        with patch("animus_forge.api_routes.agents._get_process_registry", return_value=registry):
            with patch("animus_forge.agents.process_registry.ProcessType", side_effect=ValueError):
                resp = client.get("/agents/processes?process_type=bad")
        data = resp.json()
        assert data["processes"] == []

    def test_limit_param(self, client):
        entries = [MagicMock() for _ in range(5)]
        for i, e in enumerate(entries):
            e.to_dict.return_value = {"id": f"p{i}"}
        registry = MagicMock()
        registry.list_all.return_value = entries
        with patch("animus_forge.api_routes.agents._get_process_registry", return_value=registry):
            resp = client.get("/agents/processes?limit=3")
        assert resp.json()["total"] == 3


class TestGetProcess:
    """Test GET /agents/processes/{process_id}."""

    def test_no_registry(self, client):
        with patch("animus_forge.api_routes.agents._get_process_registry", return_value=None):
            resp = client.get("/agents/processes/abc")
        assert resp.status_code == 404

    def test_not_found(self, client):
        registry = MagicMock()
        registry.get.return_value = None
        with patch("animus_forge.api_routes.agents._get_process_registry", return_value=registry):
            resp = client.get("/agents/processes/missing")
        assert resp.status_code == 404

    def test_found(self, client):
        entry = MagicMock()
        entry.to_dict.return_value = {"id": "p1", "status": "completed"}
        registry = MagicMock()
        registry.get.return_value = entry
        with patch("animus_forge.api_routes.agents._get_process_registry", return_value=registry):
            resp = client.get("/agents/processes/p1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "p1"


# ===========================================================================
# Agent memory endpoints
# ===========================================================================


@dataclass
class FakeMemoryEntry:
    id: int = 1
    agent_id: str = "agent-1"
    workflow_id: str | None = None
    memory_type: str = "fact"
    content: str = "test content"
    metadata: dict | None = None
    importance: float = 0.5
    created_at: str = "2026-01-01"
    accessed_at: str = "2026-01-01"
    access_count: int = 0


class TestGetAgentMemory:
    """Test GET /agents/memory/{agent_id}."""

    def test_no_memory_store(self, client):
        with patch("animus_forge.api_routes.agents._get_agent_memory", return_value=None):
            resp = client.get("/agents/memory/agent-1")
        assert resp.status_code == 200
        assert resp.json()["memories"] == []

    def test_with_memories(self, client):
        mem = MagicMock()
        mem.recall.return_value = [FakeMemoryEntry()]
        with patch("animus_forge.api_routes.agents._get_agent_memory", return_value=mem):
            resp = client.get("/agents/memory/agent-1")
        data = resp.json()
        assert data["total"] == 1
        assert data["memories"][0]["content"] == "test content"

    def test_filters_passed(self, client):
        mem = MagicMock()
        mem.recall.return_value = []
        with patch("animus_forge.api_routes.agents._get_agent_memory", return_value=mem):
            client.get("/agents/memory/a1?memory_type=fact&limit=5&min_importance=0.8")
        mem.recall.assert_called_once_with(
            agent_id="a1",
            memory_type="fact",
            limit=5,
            min_importance=0.8,
        )


class TestGetAgentMemoryStats:
    """Test GET /agents/memory/{agent_id}/stats."""

    def test_no_memory_store(self, client):
        with patch("animus_forge.api_routes.agents._get_agent_memory", return_value=None):
            resp = client.get("/agents/memory/agent-1/stats")
        assert resp.status_code == 200
        assert resp.json()["stats"] == {}

    def test_with_stats(self, client):
        mem = MagicMock()
        mem.get_stats.return_value = {"total_memories": 5, "by_type": {"fact": 3}}
        with patch("animus_forge.api_routes.agents._get_agent_memory", return_value=mem):
            resp = client.get("/agents/memory/agent-1/stats")
        data = resp.json()
        assert data["stats"]["total_memories"] == 5


class TestStoreAgentMemory:
    """Test POST /agents/memory/{agent_id}."""

    def test_no_memory_store(self, client):
        with patch("animus_forge.api_routes.agents._get_agent_memory", return_value=None):
            resp = client.post("/agents/memory/agent-1?content=hello")
        assert resp.status_code == 404

    def test_store_success(self, client):
        mem = MagicMock()
        mem.store.return_value = 42
        with patch("animus_forge.api_routes.agents._get_agent_memory", return_value=mem):
            resp = client.post(
                "/agents/memory/agent-1?content=hello&memory_type=fact&importance=0.9"
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["memory_id"] == 42
        assert data["stored"] is True


class TestForgetAgentMemory:
    """Test DELETE /agents/memory/{agent_id}."""

    def test_no_memory_store(self, client):
        with patch("animus_forge.api_routes.agents._get_agent_memory", return_value=None):
            resp = client.delete("/agents/memory/agent-1")
        assert resp.status_code == 404

    def test_forget_success(self, client):
        mem = MagicMock()
        mem.forget.return_value = 3
        with patch("animus_forge.api_routes.agents._get_agent_memory", return_value=mem):
            resp = client.delete("/agents/memory/agent-1?memory_type=conversation")
        data = resp.json()
        assert data["removed"] == 3

    def test_forget_by_id(self, client):
        mem = MagicMock()
        mem.forget.return_value = 1
        with patch("animus_forge.api_routes.agents._get_agent_memory", return_value=mem):
            client.delete("/agents/memory/agent-1?memory_id=5")
        mem.forget.assert_called_once_with(
            agent_id="agent-1",
            memory_id=5,
            memory_type=None,
        )
