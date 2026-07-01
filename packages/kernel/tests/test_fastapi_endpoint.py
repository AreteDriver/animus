"""Tests for the FastAPI chat/budget/queue endpoints."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from animus_kernel.server.app import app


@pytest.fixture
def client():
    """TestClient backed by the real app (static dir already exists)."""
    return TestClient(app)


class TestChatEndpoint:
    def test_chat_when_ollama_unconfigured(self, client, monkeypatch):
        monkeypatch.setattr(
            "animus_kernel.server.app.OllamaProvider.is_configured",
            lambda self: False,
        )
        response = client.post("/chat", json={"message": "hello"})
        assert response.status_code == 503
        body = response.json()
        assert "Ollama is not reachable" in body["detail"]

    def test_chat_stream_success(self, client, monkeypatch):
        async def _fake_stream(self, request):
            from animus_kernel.providers.base import StreamChunk

            yield StreamChunk(content="Hello ", model="m", provider="p")
            yield StreamChunk(content="world", model="m", provider="p")

        monkeypatch.setattr(
            "animus_kernel.server.app.OllamaProvider.is_configured",
            lambda self: True,
        )
        monkeypatch.setattr(
            "animus_kernel.server.app.OllamaProvider.initialize",
            lambda self: None,
        )
        monkeypatch.setattr(
            "animus_kernel.providers.ollama_provider.OllamaProvider.complete_stream_async",
            _fake_stream,
        )
        response = client.post("/chat", json={"message": "hello"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        text = response.text
        assert "Hello " in text
        assert "world" in text
        assert "done" in text

    def test_chat_init_failure(self, client, monkeypatch):
        monkeypatch.setattr(
            "animus_kernel.server.app.OllamaProvider.is_configured",
            lambda self: True,
        )
        monkeypatch.setattr(
            "animus_kernel.server.app.OllamaProvider.initialize",
            lambda self: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        response = client.post("/chat", json={"message": "hello"})
        assert response.status_code == 503

    def test_stream_error(self, client, monkeypatch):
        async def _bad_stream(self, request):
            # Async generator that immediately raises (avoids coroutine-not-awaited warning)
            raise RuntimeError("stream broken")
            yield  # noqa: B901

        monkeypatch.setattr(
            "animus_kernel.server.app.OllamaProvider.is_configured",
            lambda self: True,
        )
        monkeypatch.setattr(
            "animus_kernel.server.app.OllamaProvider.initialize",
            lambda self: None,
        )
        monkeypatch.setattr(
            "animus_kernel.providers.ollama_provider.OllamaProvider.complete_stream_async",
            _bad_stream,
        )
        response = client.post("/chat", json={"message": "hello"})
        assert response.status_code == 200
        assert "error" in response.text


class TestBudgetEndpoint:
    def test_get_budget(self, client, monkeypatch):
        # Reset budget to known state
        from animus_kernel.budget.manager import BudgetConfig, BudgetManager

        fresh = BudgetManager(config=BudgetConfig(total_budget=600))
        monkeypatch.setattr("animus_kernel.server.app._budget_manager", fresh)
        response = client.get("/api/budget")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 600
        assert body["used"] == 0
        assert body["remaining"] == 600
        assert body["status"] == "ok"
        assert body["percent"] == 0.0


class TestQueueEndpoint:
    def test_get_queue_empty(self, client):
        response = client.get("/api/queue")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_queue_with_items(self, client, monkeypatch):
        monkeypatch.setattr("animus_kernel.server.app._build_queue", [{"id": "1"}])
        response = client.get("/api/queue")
        assert response.status_code == 200
        assert response.json() == [{"id": "1"}]


class TestExceptionHandlers:
    def test_http_exception(self, client):
        response = client.get("/__not_a_route__")
        assert response.status_code == 404

    def test_validation_error(self, client):
        response = client.post("/chat", json={"bad_key": "value"})
        assert response.status_code == 400
        body = response.json()
        assert "Validation Error" in body["error"]

    def test_generic_exception(self):
        import asyncio

        handler = app.exception_handlers[Exception]
        response = asyncio.run(handler(None, RuntimeError("kaboom")))
        assert response.status_code == 500
        body = json.loads(response.body)
        assert "RuntimeError" in body["error"]
        assert "kaboom" in body["detail"]
