"""Tests for operational control endpoints (Phase 3)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from animus_bootstrap.dashboard.app import app


@pytest.fixture
def client():
    """Return a TestClient with CSRF cookie preset."""
    with TestClient(app) as c:
        # Prime the CSRF cookie by hitting a safe endpoint
        c.get("/")
        yield c


@pytest.fixture(autouse=True)
def reset_app_state():
    """Clear runtime mock between tests."""
    app.state.runtime = None
    yield


def _csrf_headers(client: TestClient) -> dict[str, str]:
    """Build headers with CSRF token from cookie."""
    token = client.cookies.get("animus_csrf", "")
    return {"X-CSRF-Token": token}


# ── Pause / Resume ──────────────────────────────────────────────────────────


def test_runtime_pause_records_event(client: TestClient) -> None:
    """POST /runtime/pause sets paused flag and records an event."""
    runtime = MagicMock()
    runtime.started = True
    runtime.paused = False
    app.state.runtime = runtime

    resp = client.post("/runtime/pause", headers=_csrf_headers(client))
    assert resp.status_code == 200
    assert resp.json()["status"] == "paused"
    runtime.pause.assert_called_once()


def test_runtime_resume_records_event(client: TestClient) -> None:
    """POST /runtime/resume clears paused flag and records an event."""
    runtime = MagicMock()
    runtime.started = True
    runtime.paused = True
    app.state.runtime = runtime

    resp = client.post("/runtime/resume", headers=_csrf_headers(client))
    assert resp.status_code == 200
    assert resp.json()["status"] == "resumed"
    runtime.resume.assert_called_once()


# ── Task Kill ───────────────────────────────────────────────────────────────


def test_task_kill_deletes_and_records_event(client: TestClient) -> None:
    """POST /tasks/{id}/kill deletes the task and records an event."""
    runtime = MagicMock()
    store = MagicMock()
    runtime._task_store = store
    app.state.runtime = runtime

    resp = client.post("/tasks/abc123/kill", headers=_csrf_headers(client))
    assert resp.status_code == 200
    store.delete.assert_called_once_with("abc123")


# ── Memory Clear ────────────────────────────────────────────────────────────


def test_memory_clear_backend_with_clear_method(client: TestClient) -> None:
    """POST /memory/clear calls backend.clear() when available."""
    runtime = MagicMock()
    backend = MagicMock()
    backend.clear = MagicMock(return_value=None)
    runtime.memory_manager._backend = backend
    app.state.runtime = runtime

    resp = client.post("/memory/clear", headers=_csrf_headers(client))
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cleared"
    backend.clear.assert_called_once()


def test_memory_clear_fallback_search_delete(client: TestClient) -> None:
    """POST /memory/clear falls back to search+delete when no clear() method."""
    runtime = MagicMock()
    backend = MagicMock()
    del backend.clear  # no clear method
    backend.search = MagicMock(return_value=[{"id": "m1"}, {"id": "m2"}])
    backend.delete = MagicMock(return_value=True)
    runtime.memory_manager._backend = backend
    app.state.runtime = runtime

    resp = client.post("/memory/clear", headers=_csrf_headers(client))
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cleared"
    assert data["entries"] == 2


# ── Tool Re-run ─────────────────────────────────────────────────────────────


def test_tool_rerun_executes_and_records_event(client: TestClient) -> None:
    """POST /tools/{name}/rerun executes the tool and records an event."""

    runtime = MagicMock()
    executor = MagicMock()
    result = MagicMock()
    result.success = True
    result.output = "ok"
    result.duration_ms = 42.0

    # Make execute a coroutine mock
    async def _execute(*args, **kwargs):
        return result

    executor.execute = _execute
    runtime.tool_executor = executor
    app.state.runtime = runtime

    resp = client.post(
        "/tools/test_tool/rerun",
        data={"arguments": '{"key": "val"}'},
        headers=_csrf_headers(client),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["duration_ms"] == 42.0


def test_tool_rerun_no_executor(client: TestClient) -> None:
    """POST /tools/{name}/rerun returns 503 when executor is missing."""
    runtime = MagicMock()
    runtime.tool_executor = None
    app.state.runtime = runtime

    resp = client.post(
        "/tools/test_tool/rerun",
        data={"arguments": "{}"},
        headers=_csrf_headers(client),
    )
    assert resp.status_code == 503


# ── Events Export ──────────────────────────────────────────────────────────


def test_events_export_json(client: TestClient) -> None:
    """GET /events/export?format=json returns JSON attachment."""
    runtime = MagicMock()
    ledger = MagicMock()
    ledger.query = MagicMock(return_value=[{"type": "test", "source": "s", "payload": {}}])
    runtime.event_ledger = ledger
    app.state.runtime = runtime

    resp = client.get("/events/export?format=json")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    assert "animus_events.json" in resp.headers.get("content-disposition", "")


def test_events_export_csv(client: TestClient) -> None:
    """GET /events/export?format=csv returns CSV attachment."""
    runtime = MagicMock()
    ledger = MagicMock()
    ledger.query = MagicMock(return_value=[{"type": "test", "source": "s", "payload": {}}])
    runtime.event_ledger = ledger
    app.state.runtime = runtime

    resp = client.get("/events/export?format=csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    body = resp.text
    assert "timestamp,type,source,payload" in body


# ── Alert Acknowledge ───────────────────────────────────────────────────────


def test_alert_acknowledge_form_encoded(client: TestClient) -> None:
    """POST /alerts/acknowledge accepts form-encoded alert_type."""
    runtime = MagicMock()
    ledger = MagicMock()
    runtime.event_ledger = ledger
    app.state.runtime = runtime

    resp = client.post(
        "/alerts/acknowledge",
        data={"alert_type": "error_rate"},
        headers=_csrf_headers(client),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "acknowledged"
    assert resp.json()["alert_type"] == "error_rate"


def test_alert_acknowledge_json_body(client: TestClient) -> None:
    """POST /alerts/acknowledge accepts JSON body with alert_type."""
    runtime = MagicMock()
    ledger = MagicMock()
    runtime.event_ledger = ledger
    app.state.runtime = runtime

    headers = _csrf_headers(client)
    headers["Content-Type"] = "application/json"
    resp = client.post(
        "/alerts/acknowledge",
        data=json.dumps({"alert_type": "tool_failure_rate"}),
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["alert_type"] == "tool_failure_rate"


# ── CSRF Protection ─────────────────────────────────────────────────────────


def test_pause_requires_csrf(client: TestClient) -> None:
    """POST /runtime/pause without CSRF token returns 403."""
    resp = client.post("/runtime/pause")
    assert resp.status_code == 403


def test_kill_requires_csrf(client: TestClient) -> None:
    """POST /tasks/{id}/kill without CSRF token returns 403."""
    resp = client.post("/tasks/abc/kill")
    assert resp.status_code == 403


def test_memory_clear_requires_csrf(client: TestClient) -> None:
    """POST /memory/clear without CSRF token returns 403."""
    resp = client.post("/memory/clear")
    assert resp.status_code == 403


def test_tool_rerun_requires_csrf(client: TestClient) -> None:
    """POST /tools/x/rerun without CSRF token returns 403."""
    resp = client.post("/tools/x/rerun")
    assert resp.status_code == 403


def test_alert_acknowledge_requires_csrf(client: TestClient) -> None:
    """POST /alerts/acknowledge without CSRF token returns 403."""
    resp = client.post("/alerts/acknowledge")
    assert resp.status_code == 403
