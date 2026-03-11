"""End-to-end smoke tests for the full AnimusRuntime lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from animus_bootstrap.config.schema import (
    AnimusConfig,
    AnimusSection,
    ApiSection,
    ForgeSection,
    GatewaySection,
    IntelligenceSection,
    PersonasSection,
    ProactiveSection,
    ServicesSection,
)
from animus_bootstrap.runtime import AnimusRuntime


def _make_config(
    *,
    intelligence_enabled: bool = True,
    proactive_enabled: bool = True,
    backend: str = "ollama",
    data_dir: str = "/tmp/animus-test",
    memory_backend: str = "sqlite",
    memory_db_path: str = "/tmp/animus-test/intelligence.db",
) -> AnimusConfig:
    """Build an AnimusConfig with all subsystems enabled."""
    return AnimusConfig(
        animus=AnimusSection(data_dir=data_dir),
        api=ApiSection(anthropic_key=""),
        forge=ForgeSection(enabled=False, host="localhost", port=9999, api_key="fk-test"),
        gateway=GatewaySection(default_backend=backend, system_prompt="You are Animus."),
        intelligence=IntelligenceSection(
            enabled=intelligence_enabled,
            memory_backend=memory_backend,
            memory_db_path=memory_db_path,
        ),
        proactive=ProactiveSection(enabled=proactive_enabled),
        services=ServicesSection(port=7700),
        personas=PersonasSection(enabled=True),
    )


def _start_runtime(rt: AnimusRuntime) -> None:
    """Start runtime, suppressing actual network calls."""
    asyncio.run(rt.start())


def _stop_runtime(rt: AnimusRuntime) -> None:
    """Stop runtime."""
    asyncio.run(rt.stop())


class TestRuntimeE2E:
    """End-to-end smoke tests covering full boot, health, and shutdown."""

    def _full_config(self, tmp_path: Path) -> AnimusConfig:
        """Config with all subsystems enabled, isolated to tmp_path."""
        return _make_config(
            intelligence_enabled=True,
            proactive_enabled=True,
            data_dir=str(tmp_path),
            memory_db_path=str(tmp_path / "intelligence.db"),
        )

    def test_full_boot_all_components_initialized(self, tmp_path: Path) -> None:
        """Boot with all subsystems and verify every component is alive."""
        cfg = self._full_config(tmp_path)
        rt = AnimusRuntime(config=cfg)
        try:
            _start_runtime(rt)

            assert rt.memory_manager is not None
            assert rt.tool_executor is not None
            assert rt.proactive_engine is not None
            assert rt.automation_engine is not None
            assert rt.persona_engine is not None
            assert rt.cognitive_backend is not None
            assert rt.router is not None
            assert rt.session_manager is not None
            assert rt.identity_manager is not None
            assert rt.context_adapter is not None
            assert rt.feedback_store is not None
        finally:
            _stop_runtime(rt)

    def test_health_endpoint_all_true(self, tmp_path: Path) -> None:
        """GET /health returns ok with all component flags true."""
        cfg = self._full_config(tmp_path)
        rt = AnimusRuntime(config=cfg)
        try:
            _start_runtime(rt)

            from animus_bootstrap.dashboard.app import app

            app.state.runtime = rt
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")

            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["components"]["memory"] is True
            assert data["components"]["tools"] is True
            assert data["components"]["proactive"] is True
            assert data["components"]["automations"] is True
        finally:
            _stop_runtime(rt)

    def test_clean_shutdown(self, tmp_path: Path) -> None:
        """Verify stop() cleans up all components and resets started."""
        cfg = self._full_config(tmp_path)
        rt = AnimusRuntime(config=cfg)
        try:
            _start_runtime(rt)
            assert rt.started is True
        finally:
            _stop_runtime(rt)

        assert rt.started is False

    def test_started_flag_lifecycle(self, tmp_path: Path) -> None:
        """started is False -> True after start -> False after stop."""
        cfg = self._full_config(tmp_path)
        rt = AnimusRuntime(config=cfg)

        assert rt.started is False

        try:
            _start_runtime(rt)
            assert rt.started is True
        finally:
            _stop_runtime(rt)

        assert rt.started is False

    def test_double_start_is_idempotent(self, tmp_path: Path) -> None:
        """Calling start() twice does not crash or re-initialize."""
        cfg = self._full_config(tmp_path)
        rt = AnimusRuntime(config=cfg)
        try:
            _start_runtime(rt)
            first_memory = rt.memory_manager

            # Second start should be a no-op
            _start_runtime(rt)
            assert rt.memory_manager is first_memory
            assert rt.started is True
        finally:
            _stop_runtime(rt)

    def test_stop_without_start_is_safe(self, tmp_path: Path) -> None:
        """Calling stop() on a never-started runtime does not crash."""
        cfg = self._full_config(tmp_path)
        rt = AnimusRuntime(config=cfg)
        # Should not raise
        _stop_runtime(rt)
        assert rt.started is False

    def test_double_stop_is_safe(self, tmp_path: Path) -> None:
        """Calling stop() twice does not crash."""
        cfg = self._full_config(tmp_path)
        rt = AnimusRuntime(config=cfg)
        try:
            _start_runtime(rt)
        finally:
            _stop_runtime(rt)

        # Second stop should be a no-op
        _stop_runtime(rt)
        assert rt.started is False

    def test_health_endpoint_degraded_without_runtime(self) -> None:
        """GET /health returns degraded when no runtime is set."""
        from animus_bootstrap.dashboard.app import app

        # Clear any existing runtime
        if hasattr(app.state, "runtime"):
            delattr(app.state, "runtime")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["components"]["memory"] is False
        assert data["components"]["tools"] is False
        assert data["components"]["proactive"] is False
        assert data["components"]["automations"] is False

    def test_tool_executor_has_builtin_tools(self, tmp_path: Path) -> None:
        """Tool executor ships with built-in tools registered."""
        cfg = self._full_config(tmp_path)
        rt = AnimusRuntime(config=cfg)
        try:
            _start_runtime(rt)
            tools = rt.tool_executor.list_tools()
            assert len(tools) > 0
        finally:
            _stop_runtime(rt)

    def test_data_files_created(self, tmp_path: Path) -> None:
        """Runtime creates expected database files in data_dir."""
        cfg = self._full_config(tmp_path)
        rt = AnimusRuntime(config=cfg)
        try:
            _start_runtime(rt)

            # Intelligence subsystem DBs
            assert (tmp_path / "intelligence.db").exists()
            assert (tmp_path / "sessions.db").exists()
            assert (tmp_path / "automations.db").exists()
            assert (tmp_path / "feedback.db").exists()
            assert (tmp_path / "tool_history.db").exists()
        finally:
            _stop_runtime(rt)
