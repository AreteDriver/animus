"""Targeted tests to close coverage gaps in lowest-coverage modules.

Applies standalone testing techniques: lazy import mocking, SSE generator testing,
threshold edge cases, service lifecycle mocking.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from animus_bootstrap.intelligence.feedback import FeedbackStore


def _run(coro):
    """Run an async coroutine without closing the global event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


# ---------------------------------------------------------------------------
# 1. System tools — shell_exec coverage (38% → target ~80%)
# ---------------------------------------------------------------------------
class TestShellExec:
    """Tests for system.py _shell_exec — covers timeout and OSError paths."""

    def test_shell_exec_basic_command(self):
        from animus_bootstrap.intelligence.tools.builtin.system import _shell_exec

        result = _run(_shell_exec("echo hello"))
        assert "hello" in result

    def test_shell_exec_timeout(self):
        from animus_bootstrap.intelligence.tools.builtin.system import _shell_exec

        result = _run(_shell_exec("sleep 60", timeout=0.1))
        assert "timed out" in result

    def test_shell_exec_invalid_command(self):
        from animus_bootstrap.intelligence.tools.builtin.system import _shell_exec

        result = _run(_shell_exec("/nonexistent_binary_xyz_12345"))
        assert "Failed to execute" in result

    def test_shell_exec_stderr_output(self):
        from animus_bootstrap.intelligence.tools.builtin.system import _shell_exec

        result = _run(_shell_exec("ls /nonexistent_dir_xyz_12345"))
        assert "[stderr]" in result or "No such file" in result or "exit code" in result

    def test_get_system_tools_returns_one(self):
        from animus_bootstrap.intelligence.tools.builtin.system import get_system_tools

        tools = get_system_tools()
        assert len(tools) == 1
        assert tools[0].name == "shell_exec"
        assert tools[0].permission == "approve"


# ---------------------------------------------------------------------------
# 2. Feedback dashboard router (33% → target ~80%)
# ---------------------------------------------------------------------------
class TestFeedbackDashboardRouter:
    """Tests for dashboard/routers/feedback.py — POST and GET endpoints."""

    @pytest.fixture()
    def feedback_app(self, tmp_path):
        from animus_bootstrap.dashboard.routers.feedback import router

        app = FastAPI()
        app.include_router(router)

        from fastapi.templating import Jinja2Templates

        template_dir = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "animus_bootstrap"
            / "dashboard"
            / "templates"
        )
        app.state.templates = Jinja2Templates(directory=str(template_dir))

        store = FeedbackStore(tmp_path / "feedback.db")
        runtime = MagicMock()
        runtime.feedback_store = store
        app.state.runtime = runtime

        return app, store

    def test_record_feedback_positive(self, feedback_app):
        app, store = feedback_app
        client = TestClient(app)
        resp = client.post(
            "/api/feedback",
            data={
                "message_text": "hello",
                "response_text": "hi there",
                "rating": 1,
            },
        )
        assert resp.status_code == 200
        assert "Thanks" in resp.text
        assert len(store.get_recent()) == 1

    def test_record_feedback_negative(self, feedback_app):
        app, store = feedback_app
        client = TestClient(app)
        resp = client.post(
            "/api/feedback",
            data={
                "message_text": "bad query",
                "response_text": "bad answer",
                "rating": -1,
                "comment": "too verbose",
            },
        )
        assert resp.status_code == 200
        recent = store.get_recent()
        assert recent[0]["rating"] == -1

    def test_record_feedback_no_store(self):
        from animus_bootstrap.dashboard.routers.feedback import router

        app = FastAPI()
        app.include_router(router)
        app.state.runtime = None
        client = TestClient(app)
        resp = client.post("/api/feedback", data={"rating": 1})
        assert resp.status_code == 200
        assert "not available" in resp.text

    def test_feedback_page_with_data(self, feedback_app):
        app, store = feedback_app
        store.record("q1", "a1", 1)
        store.record("q2", "a2", -1)
        client = TestClient(app)
        resp = client.get("/feedback")
        assert resp.status_code == 200

    def test_feedback_page_no_store(self):
        from animus_bootstrap.dashboard.routers.feedback import router

        app = FastAPI()
        app.include_router(router)

        from fastapi.templating import Jinja2Templates

        template_dir = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "animus_bootstrap"
            / "dashboard"
            / "templates"
        )
        app.state.templates = Jinja2Templates(directory=str(template_dir))
        app.state.runtime = None
        client = TestClient(app)
        resp = client.get("/feedback")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 3. Web tools — lazy import mocking (61% → target ~80%)
# ---------------------------------------------------------------------------
class TestWebTools:
    """Tests for web.py — DuckDuckGo mock and httpx fetch."""

    def test_web_search_without_duckduckgo(self):
        from animus_bootstrap.intelligence.tools.builtin import web

        original = web.HAS_DUCKDUCKGO
        web.HAS_DUCKDUCKGO = False
        try:
            result = _run(web._web_search("test query"))
            assert "Would search for" in result
            assert "duckduckgo-search" in result
        finally:
            web.HAS_DUCKDUCKGO = original

    def test_web_search_with_mocked_ddgs(self):
        from animus_bootstrap.intelligence.tools.builtin import web

        mock_results = [
            {"title": "Result 1", "href": "https://example.com", "body": "Body text"},
        ]
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text = MagicMock(return_value=mock_results)

        original = web.HAS_DUCKDUCKGO
        web.HAS_DUCKDUCKGO = True
        try:
            with patch.object(web, "DDGS", return_value=mock_ddgs, create=True):
                result = _run(web._web_search("test query"))
                assert "Result 1" in result
                assert "example.com" in result
        finally:
            web.HAS_DUCKDUCKGO = original

    def test_web_search_no_results(self):
        from animus_bootstrap.intelligence.tools.builtin import web

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text = MagicMock(return_value=[])

        original = web.HAS_DUCKDUCKGO
        web.HAS_DUCKDUCKGO = True
        try:
            with patch.object(web, "DDGS", return_value=mock_ddgs, create=True):
                result = _run(web._web_search("obscure query"))
                assert "No results found" in result
        finally:
            web.HAS_DUCKDUCKGO = original

    def test_web_fetch_success(self):
        """Mock httpx at the module level since _web_fetch imports it lazily."""
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>Hello World</body></html>"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            from animus_bootstrap.intelligence.tools.builtin.web import _web_fetch

            result = _run(_web_fetch("https://example.com"))
            assert "Hello World" in result

    def test_get_web_tools(self):
        from animus_bootstrap.intelligence.tools.builtin.web import get_web_tools

        tools = get_web_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "web_search" in names
        assert "web_fetch" in names


# ---------------------------------------------------------------------------
# 4. SSE log streaming (73% → target ~85%)
# ---------------------------------------------------------------------------
class TestLogStreaming:
    """Tests for logs.py — SSE generator and log page."""

    def test_tail_log_no_file(self):
        from animus_bootstrap.dashboard.routers.logs import _tail_log

        with patch("animus_bootstrap.dashboard.routers.logs._LOG_FILE") as mock_path:
            mock_path.is_file.return_value = False

            async def collect():
                events = []
                async for event in _tail_log():
                    events.append(event)
                    break  # just get the first event
                return events

            events = _run(collect())
            assert len(events) == 1
            assert events[0]["data"] == "No logs yet"

    def test_logs_page_renders(self):
        from animus_bootstrap.dashboard.routers.logs import router

        app = FastAPI()
        app.include_router(router)

        from fastapi.templating import Jinja2Templates

        template_dir = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "animus_bootstrap"
            / "dashboard"
            / "templates"
        )
        app.state.templates = Jinja2Templates(directory=str(template_dir))
        client = TestClient(app)
        resp = client.get("/logs")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 5. Linux service lifecycle with mocked subprocess
# ---------------------------------------------------------------------------
class TestLinuxServiceLifecycle:
    """Tests for daemon/platforms/linux.py — service start/stop/restart."""

    def test_generate_unit_file(self):
        from animus_bootstrap.daemon.platforms.linux import LinuxService

        svc = LinuxService()
        unit = svc.generate_systemd_unit("/usr/bin/python3", "animus_bootstrap.daemon")
        assert "[Unit]" in unit
        assert "[Service]" in unit
        assert "[Install]" in unit
        assert "animus_bootstrap.daemon" in unit

    def test_stop_success(self):
        from animus_bootstrap.daemon.platforms.linux import LinuxService

        svc = LinuxService()
        with patch("animus_bootstrap.daemon.platforms.linux.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert svc.stop() is True

    def test_stop_timeout(self):
        import subprocess

        from animus_bootstrap.daemon.platforms.linux import LinuxService

        svc = LinuxService()
        with patch(
            "animus_bootstrap.daemon.platforms.linux.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="systemctl", timeout=30),
        ):
            assert svc.stop() is False

    def test_stop_file_not_found(self):
        from animus_bootstrap.daemon.platforms.linux import LinuxService

        svc = LinuxService()
        with patch(
            "animus_bootstrap.daemon.platforms.linux.subprocess.run",
            side_effect=FileNotFoundError("systemctl not found"),
        ):
            assert svc.stop() is False

    def test_is_running_active(self):
        from animus_bootstrap.daemon.platforms.linux import LinuxService

        svc = LinuxService()
        with patch("animus_bootstrap.daemon.platforms.linux.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=b"active\n",
            )
            assert svc.is_running() is True

    def test_is_running_inactive(self):
        from animus_bootstrap.daemon.platforms.linux import LinuxService

        svc = LinuxService()
        with patch("animus_bootstrap.daemon.platforms.linux.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=3,
                stdout=b"inactive\n",
            )
            assert svc.is_running() is False

    def test_is_running_systemctl_not_found(self):
        from animus_bootstrap.daemon.platforms.linux import LinuxService

        svc = LinuxService()
        with patch(
            "animus_bootstrap.daemon.platforms.linux.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert svc.is_running() is False

    def test_enable_and_start_success(self):
        from animus_bootstrap.daemon.platforms.linux import LinuxService

        svc = LinuxService()
        with patch("animus_bootstrap.daemon.platforms.linux.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert svc.enable_and_start() is True
            assert mock_run.call_count == 2  # enable + start

    def test_enable_and_start_fails_on_enable(self):
        import subprocess

        from animus_bootstrap.daemon.platforms.linux import LinuxService

        svc = LinuxService()
        with patch(
            "animus_bootstrap.daemon.platforms.linux.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "systemctl", stderr=b"failed"),
        ):
            assert svc.enable_and_start() is False
