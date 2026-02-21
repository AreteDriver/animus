"""Tests for the intelligence dashboard routes (tools, automations, activity)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from animus_bootstrap.dashboard.app import app

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """TestClient for the dashboard app."""
    return TestClient(app)


# ------------------------------------------------------------------
# Tools page — GET /tools
# ------------------------------------------------------------------


class TestToolsPage:
    """Tests for the tools management dashboard route."""

    def test_tools_page_returns_200(self, client: TestClient) -> None:
        """GET /tools returns 200."""
        resp = client.get("/tools")
        assert resp.status_code == 200

    def test_tools_page_has_table_structure(self, client: TestClient) -> None:
        """GET /tools response contains the tools section and empty state."""
        resp = client.get("/tools")
        body = resp.text
        assert "Available Tools" in body
        assert "No tools registered yet" in body

    def test_tools_page_has_history_section(self, client: TestClient) -> None:
        """GET /tools response contains the execution history section."""
        resp = client.get("/tools")
        body = resp.text
        assert "Execution History" in body
        assert "No tool executions recorded yet" in body


# ------------------------------------------------------------------
# Automations page — GET /automations
# ------------------------------------------------------------------


class TestAutomationsPage:
    """Tests for the automations management dashboard route."""

    def test_automations_page_returns_200(self, client: TestClient) -> None:
        """GET /automations returns 200."""
        resp = client.get("/automations")
        assert resp.status_code == 200

    def test_automations_page_has_table_structure(self, client: TestClient) -> None:
        """GET /automations response contains the rules section and empty state."""
        resp = client.get("/automations")
        body = resp.text
        assert "Active Rules" in body
        assert "No automation rules configured yet" in body

    def test_automations_page_has_add_button(self, client: TestClient) -> None:
        """GET /automations response contains the Add Rule button."""
        resp = client.get("/automations")
        assert "Add Rule" in resp.text

    def test_automations_page_has_history_section(self, client: TestClient) -> None:
        """GET /automations response contains the execution history section."""
        resp = client.get("/automations")
        body = resp.text
        assert "Execution History" in body
        assert "No automation executions recorded yet" in body


# ------------------------------------------------------------------
# Activity page — GET /activity
# ------------------------------------------------------------------


class TestActivityPage:
    """Tests for the proactive engine activity dashboard route."""

    def test_activity_page_returns_200(self, client: TestClient) -> None:
        """GET /activity returns 200."""
        resp = client.get("/activity")
        assert resp.status_code == 200

    def test_activity_page_has_engine_status(self, client: TestClient) -> None:
        """GET /activity response shows engine status."""
        resp = client.get("/activity")
        body = resp.text
        assert "Engine Status" in body
        assert "Stopped" in body

    def test_activity_page_has_checks_section(self, client: TestClient) -> None:
        """GET /activity response contains the registered checks section."""
        resp = client.get("/activity")
        body = resp.text
        assert "Registered Checks" in body
        assert "No proactive checks registered yet" in body

    def test_activity_page_has_nudge_history(self, client: TestClient) -> None:
        """GET /activity response contains the nudge history section."""
        resp = client.get("/activity")
        body = resp.text
        assert "Nudge History" in body
        assert "No nudges delivered yet" in body

    def test_activity_page_has_manual_triggers(self, client: TestClient) -> None:
        """GET /activity response contains manual trigger buttons."""
        resp = client.get("/activity")
        body = resp.text
        assert "Manual Triggers" in body
        assert "Run All Checks" in body
        assert "Send Test Nudge" in body
