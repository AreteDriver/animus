"""Tests for the self-mod, forge, and timers dashboard routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from animus_bootstrap.dashboard.app import app

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_improvement_log() -> None:
    """Ensure improvement log is clean for each test."""
    from animus_bootstrap.intelligence.tools.builtin.self_improve import (
        clear_improvement_log,
    )

    clear_improvement_log()
    yield
    clear_improvement_log()


@pytest.fixture()
def client() -> TestClient:
    """TestClient for the dashboard app."""
    return TestClient(app)


# ------------------------------------------------------------------
# Self-Mod page — GET /self-mod
# ------------------------------------------------------------------


class TestSelfModPage:
    """Tests for the self-modification dashboard route."""

    def test_self_mod_page_returns_200(self, client: TestClient) -> None:
        """GET /self-mod returns 200."""
        resp = client.get("/self-mod")
        assert resp.status_code == 200

    def test_self_mod_page_has_improvement_section(self, client: TestClient) -> None:
        """GET /self-mod has improvement proposals section."""
        resp = client.get("/self-mod")
        body = resp.text
        assert "Improvement Proposals" in body
        assert "No improvement proposals yet" in body

    def test_self_mod_page_has_code_history_section(self, client: TestClient) -> None:
        """GET /self-mod has code edit history section."""
        resp = client.get("/self-mod")
        body = resp.text
        assert "Code Edit History" in body
        assert "No code edits recorded yet" in body

    def test_self_mod_page_has_tool_references(self, client: TestClient) -> None:
        """GET /self-mod references the correct tool names in empty state."""
        resp = client.get("/self-mod")
        body = resp.text
        # Empty state shows tool references
        assert "propose_improvement" in body
        assert "code_write" in body
        assert "code_patch" in body

    def test_self_mod_page_with_improvements(self, client: TestClient) -> None:
        """GET /self-mod shows improvements when they exist."""
        from animus_bootstrap.intelligence.tools.builtin.self_improve import (
            _improvement_log,
            clear_improvement_log,
        )

        _improvement_log.append(
            {
                "id": 1,
                "area": "tool:web_search",
                "description": "Add retry logic for timeouts",
                "status": "proposed",
                "timestamp": "2026-02-20T12:00:00+00:00",
                "analysis": "Should add exponential backoff",
                "patch": None,
            }
        )
        try:
            resp = client.get("/self-mod")
            body = resp.text
            assert "tool:web_search" in body
            assert "Add retry logic" in body
            assert "Proposed" in body
        finally:
            clear_improvement_log()

    def test_improvement_rows_have_htmx_detail(self, client: TestClient) -> None:
        """Improvement rows include hx-get for detail expansion."""
        from animus_bootstrap.intelligence.tools.builtin.self_improve import (
            _improvement_log,
        )

        _improvement_log.append(
            {
                "id": 5,
                "area": "config",
                "description": "Validate schema",
                "status": "proposed",
                "timestamp": "2026-02-20T12:00:00+00:00",
                "analysis": "Need validation",
                "patch": None,
            }
        )
        resp = client.get("/self-mod")
        body = resp.text
        assert 'hx-get="/self-mod/improvement/5"' in body
        assert 'id="detail-5"' in body


# ------------------------------------------------------------------
# Improvement detail — GET /self-mod/improvement/{id}
# ------------------------------------------------------------------


class TestImprovementDetail:
    """Tests for the improvement proposal detail fragment."""

    def test_detail_not_found(self, client: TestClient) -> None:
        """GET /self-mod/improvement/999 returns not-found message."""
        resp = client.get("/self-mod/improvement/999")
        assert resp.status_code == 200
        assert "Proposal not found" in resp.text

    def test_detail_returns_analysis(self, client: TestClient) -> None:
        """GET /self-mod/improvement/{id} returns analysis text."""
        from animus_bootstrap.intelligence.tools.builtin.self_improve import (
            _improvement_log,
        )

        _improvement_log.append(
            {
                "id": 1,
                "area": "tool:web_search",
                "description": "Add retry logic for timeouts",
                "status": "proposed",
                "timestamp": "2026-02-20T12:00:00+00:00",
                "analysis": "Should add exponential backoff with jitter",
                "patch": None,
            }
        )
        resp = client.get("/self-mod/improvement/1")
        assert resp.status_code == 200
        body = resp.text
        assert "exponential backoff with jitter" in body
        assert "tool:web_search" in body
        assert "Add retry logic" in body

    def test_detail_shows_patch_when_present(self, client: TestClient) -> None:
        """GET /self-mod/improvement/{id} shows patch if available."""
        from animus_bootstrap.intelligence.tools.builtin.self_improve import (
            _improvement_log,
        )

        _improvement_log.append(
            {
                "id": 2,
                "area": "router",
                "description": "Fix routing",
                "status": "applied",
                "timestamp": "2026-02-20T13:00:00+00:00",
                "analysis": "Route logic was wrong",
                "patch": "--- a/router.py\n+++ b/router.py\n@@ -1 +1 @@\n-old\n+new",
            }
        )
        resp = client.get("/self-mod/improvement/2")
        assert resp.status_code == 200
        body = resp.text
        assert "Patch:" in body
        assert "--- a/router.py" in body

    def test_detail_no_patch_section_when_empty(self, client: TestClient) -> None:
        """No patch section when patch is None."""
        from animus_bootstrap.intelligence.tools.builtin.self_improve import (
            _improvement_log,
        )

        _improvement_log.append(
            {
                "id": 3,
                "area": "memory",
                "description": "Optimize queries",
                "status": "proposed",
                "timestamp": "2026-02-20T14:00:00+00:00",
                "analysis": "Some analysis",
                "patch": None,
            }
        )
        resp = client.get("/self-mod/improvement/3")
        assert resp.status_code == 200
        assert "Patch:" not in resp.text


# ------------------------------------------------------------------
# Forge page — GET /forge
# ------------------------------------------------------------------


class TestForgePage:
    """Tests for the Forge status dashboard route."""

    def test_forge_page_returns_200(self, client: TestClient) -> None:
        """GET /forge returns 200."""
        resp = client.get("/forge")
        assert resp.status_code == 200

    def test_forge_page_has_status_section(self, client: TestClient) -> None:
        """GET /forge has connection status section."""
        resp = client.get("/forge")
        body = resp.text
        assert "Connection Status" in body

    def test_forge_page_disabled_by_default(self, client: TestClient) -> None:
        """GET /forge shows disabled when no runtime configured."""
        resp = client.get("/forge")
        body = resp.text
        assert "Disabled" in body

    def test_forge_page_has_title(self, client: TestClient) -> None:
        """GET /forge has correct page title."""
        resp = client.get("/forge")
        assert "Forge Orchestration" in resp.text

    def test_forge_page_no_invoke_form_when_disabled(self, client: TestClient) -> None:
        """GET /forge hides invoke form when Forge is disabled."""
        resp = client.get("/forge")
        body = resp.text
        # Invoke form is gated behind forge_enabled
        assert "Invoke Endpoint" not in body

    def test_forge_page_has_invoke_form_when_enabled(self, client: TestClient) -> None:
        """GET /forge shows invoke form when Forge is enabled."""
        from unittest.mock import MagicMock, patch

        mock_runtime = MagicMock()
        mock_runtime.config.forge.enabled = True
        mock_runtime.config.forge.host = "127.0.0.1"
        mock_runtime.config.forge.port = 8000

        with patch.object(client.app.state, "runtime", mock_runtime, create=True):
            # httpx will fail to connect — forge_status will be "stopped"
            resp = client.get("/forge")

        body = resp.text
        assert "Invoke Endpoint" in body
        assert "forge-endpoint" in body
        assert "forge-method" in body
        assert "forge-body" in body
        assert "forge_invoke" in body


# ------------------------------------------------------------------
# Timers page — GET /timers
# ------------------------------------------------------------------


class TestTimersPage:
    """Tests for the timer management dashboard route."""

    def test_timers_page_returns_200(self, client: TestClient) -> None:
        """GET /timers returns 200."""
        resp = client.get("/timers")
        assert resp.status_code == 200

    def test_timers_page_has_dynamic_section(self, client: TestClient) -> None:
        """GET /timers has dynamic timers section."""
        resp = client.get("/timers")
        body = resp.text
        assert "Dynamic Timers" in body
        assert "No dynamic timers created yet" in body

    def test_timers_page_has_engine_section(self, client: TestClient) -> None:
        """GET /timers has live engine timers section."""
        resp = client.get("/timers")
        body = resp.text
        assert "Live Engine Timers" in body
        assert "No live timers in the ProactiveEngine" in body

    def test_timers_page_has_controls(self, client: TestClient) -> None:
        """GET /timers has create timer form."""
        resp = client.get("/timers")
        body = resp.text
        assert "Create Timer" in body

    def test_timers_page_has_create_form(self, client: TestClient) -> None:
        """GET /timers has the create timer form fields."""
        resp = client.get("/timers")
        body = resp.text
        assert "timer-name" in body
        assert "timer-schedule" in body
        assert "timer-action" in body
        assert "timer_create" in body

    def test_timers_page_with_dynamic_timers(self, client: TestClient) -> None:
        """GET /timers shows timers when they exist."""
        from animus_bootstrap.intelligence.tools.builtin.timer_ctl import (
            _dynamic_timers,
            clear_dynamic_timers,
        )

        _dynamic_timers.append(
            {
                "name": "daily_check",
                "schedule": "0 9 * * *",
                "action": "Run daily health check",
                "channels": ["telegram"],
                "created": "2026-02-20T12:00:00+00:00",
            }
        )
        try:
            resp = client.get("/timers")
            body = resp.text
            assert "daily_check" in body
            assert "0 9 * * *" in body
            assert "Run daily health check" in body
        finally:
            clear_dynamic_timers()


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------


class TestSidebarNavigation:
    """Verify new pages appear in the sidebar."""

    def test_sidebar_has_self_mod_link(self, client: TestClient) -> None:
        """Homepage sidebar includes Self-Mod link."""
        resp = client.get("/")
        assert 'href="/self-mod"' in resp.text

    def test_sidebar_has_forge_link(self, client: TestClient) -> None:
        """Homepage sidebar includes Forge link."""
        resp = client.get("/")
        assert 'href="/forge"' in resp.text

    def test_sidebar_has_timers_link(self, client: TestClient) -> None:
        """Homepage sidebar includes Timers link."""
        resp = client.get("/")
        assert 'href="/timers"' in resp.text
