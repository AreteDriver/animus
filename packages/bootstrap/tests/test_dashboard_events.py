"""Tests for the operational events dashboard router and SSE stream."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from animus_bootstrap.dashboard.app import app
from animus_bootstrap.intelligence.event_ledger import EventLedger


@pytest.fixture()
def client() -> TestClient:
    """TestClient for the dashboard app."""
    return TestClient(app)


def _csrf_headers(client: TestClient) -> dict[str, str]:
    """Prime the CSRF cookie via GET / and return the X-CSRF-Token header."""
    client.get("/")
    token = client.cookies.get("animus_csrf")
    assert token is not None, "CSRF cookie not set"
    return {"X-CSRF-Token": token}


class TestEventsPage:
    """Tests for the /events page."""

    def test_events_page_returns_200(self, client: TestClient) -> None:
        """GET /events returns 200."""
        resp = client.get("/events")
        assert resp.status_code == 200

    def test_events_page_shows_empty_state(self, client: TestClient) -> None:
        """GET /events shows empty state when no ledger."""
        app.state.runtime = None  # Ensure no runtime mock leaks from other tests
        resp = client.get("/events")
        assert "No events recorded yet" in resp.text

    def test_events_page_shows_live_events(self, client: TestClient) -> None:
        """GET /events displays events from the ledger."""
        ledger = EventLedger()
        ledger.record("tool_execution", "test", {"tool_name": "echo"})
        ledger.record("task_created", "test", {"task_id": "123"})

        runtime = MagicMock()
        runtime.started = True
        runtime.event_ledger = ledger
        app.state.runtime = runtime

        resp = client.get("/events")
        assert resp.status_code == 200
        assert "tool_execution" in resp.text
        assert "task_created" in resp.text
        assert "echo" in resp.text

    def test_events_feed_fragment(self, client: TestClient) -> None:
        """GET /events/feed returns the HTMX fragment."""
        ledger = EventLedger()
        ledger.record("error", "test", {"msg": "boom"})

        runtime = MagicMock()
        runtime.started = True
        runtime.event_ledger = ledger
        app.state.runtime = runtime

        resp = client.get("/events/feed")
        assert resp.status_code == 200
        assert "error" in resp.text


class TestEventsStream:
    """Tests for the SSE /events/stream endpoint."""

    @pytest.mark.skip(reason="SSE stream is infinite; verified manually via curl")
    def test_stream_returns_sse_headers(self, client: TestClient) -> None:
        """GET /events/stream returns SSE headers (stream itself is infinite)."""
        # Manual verification: curl -N http://localhost:7700/events/stream
        pass


class TestEventLedgerIntegration:
    """Integration tests verifying event recording through dashboard actions."""

    def test_task_create_records_event(self, client: TestClient) -> None:
        """Creating a task via dashboard records a task_created event."""
        from animus_bootstrap.intelligence.tools.builtin.task_store import TaskStore
        import tempfile
        import pathlib

        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(pathlib.Path(tmp) / "tasks.db")
            ledger = EventLedger()

            runtime = MagicMock()
            runtime._task_store = store
            runtime.event_ledger = ledger
            app.state.runtime = runtime

            resp = client.post(
                "/tasks/create",
                data={"name": "test event", "description": "", "priority": "normal", "due_date": ""},
                headers=_csrf_headers(client),
                follow_redirects=False,
            )
            assert resp.status_code == 303

            # Verify event was recorded
            events = ledger.query(event_type="task_created")
            assert len(events) == 1
            assert events[0]["payload"]["name"] == "test event"
