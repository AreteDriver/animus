"""Tests for the tasks dashboard page."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from animus_bootstrap.dashboard.app import app
from animus_bootstrap.intelligence.tools.builtin.task_store import TaskStore


@pytest.fixture()
def store(tmp_path):
    """Return a TaskStore backed by tmp_path."""
    s = TaskStore(tmp_path / "tasks.db")
    yield s
    s.close()


@pytest.fixture()
def client():
    """TestClient for the dashboard app."""
    return TestClient(app)


class TestTasksPage:
    def test_get_tasks_returns_200(self, client):
        resp = client.get("/tasks")
        assert resp.status_code == 200

    def test_get_tasks_contains_heading(self, client):
        resp = client.get("/tasks")
        assert "Task Management" in resp.text

    def test_get_tasks_shows_empty_state(self, client):
        resp = client.get("/tasks")
        assert "No tasks yet" in resp.text

    def test_get_tasks_shows_tasks(self, client, store):
        store.create("test task", priority="high")
        runtime = MagicMock()
        runtime._task_store = store
        app.state.runtime = runtime
        resp = client.get("/tasks")
        assert "test task" in resp.text
        assert "high" in resp.text


class TestTasksCreate:
    def test_create_redirects(self, client, store):
        runtime = MagicMock()
        runtime._task_store = store
        app.state.runtime = runtime
        resp = client.post(
            "/tasks/create",
            data={"name": "new task", "description": "", "priority": "normal", "due_date": ""},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/tasks"

    def test_create_persists(self, client, store):
        runtime = MagicMock()
        runtime._task_store = store
        app.state.runtime = runtime
        client.post(
            "/tasks/create",
            data={"name": "persisted", "description": "desc", "priority": "urgent", "due_date": ""},
            follow_redirects=False,
        )
        tasks = store.list_all()
        assert len(tasks) == 1
        assert tasks[0]["name"] == "persisted"
        assert tasks[0]["priority"] == "urgent"


class TestTasksComplete:
    def test_complete_returns_html(self, client, store):
        runtime = MagicMock()
        runtime._task_store = store
        app.state.runtime = runtime
        task_id = store.create("completable")
        resp = client.post(f"/tasks/{task_id}/complete")
        assert resp.status_code == 200
        assert "Done" in resp.text
        assert store.get(task_id)["status"] == "completed"


class TestTasksDelete:
    def test_delete_returns_html(self, client, store):
        runtime = MagicMock()
        runtime._task_store = store
        app.state.runtime = runtime
        task_id = store.create("deletable")
        resp = client.post(f"/tasks/{task_id}/delete")
        assert resp.status_code == 200
        assert "Deleted" in resp.text
        assert store.get(task_id) is None
