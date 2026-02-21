"""Tests for the central runtime orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from animus_bootstrap.config.schema import (
    AnimusConfig,
    AnimusSection,
    ApiSection,
    ForgeSection,
    GatewaySection,
    IntelligenceSection,
    PersonaProfileConfig,
    PersonasSection,
    ProactiveCheckConfig,
    ProactiveSection,
    ServicesSection,
)
from animus_bootstrap.runtime import (
    AnimusRuntime,
    get_runtime,
    reset_runtime,
    set_runtime,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_config(
    *,
    intelligence_enabled: bool = True,
    proactive_enabled: bool = True,
    backend: str = "ollama",
    anthropic_key: str = "",
    forge_enabled: bool = False,
    data_dir: str = "/tmp/animus-test",
    memory_backend: str = "sqlite",
    memory_db_path: str = "/tmp/animus-test/intelligence.db",
) -> AnimusConfig:
    """Build an AnimusConfig with predictable values."""
    return AnimusConfig(
        animus=AnimusSection(data_dir=data_dir),
        api=ApiSection(anthropic_key=anthropic_key),
        forge=ForgeSection(enabled=forge_enabled, host="localhost", port=9999, api_key="fk-test"),
        gateway=GatewaySection(default_backend=backend, system_prompt="You are Animus."),
        intelligence=IntelligenceSection(
            enabled=intelligence_enabled,
            memory_backend=memory_backend,
            memory_db_path=memory_db_path,
        ),
        proactive=ProactiveSection(enabled=proactive_enabled),
        services=ServicesSection(port=7700),
    )


# ------------------------------------------------------------------
# TestAnimusRuntime
# ------------------------------------------------------------------


class TestAnimusRuntime:
    """Tests for the AnimusRuntime class."""

    def test_config_property(self) -> None:
        """config property returns the config passed at init."""
        cfg = _make_config()
        rt = AnimusRuntime(config=cfg)
        assert rt.config is cfg

    def test_started_property_default(self) -> None:
        """started is False before start()."""
        rt = AnimusRuntime(config=_make_config())
        assert rt.started is False

    def test_init_loads_default_config_when_none(self) -> None:
        """When config is None, runtime loads config from ConfigManager."""
        mock_cfg = _make_config()
        with patch("animus_bootstrap.runtime.ConfigManager") as mock_cm_cls:
            mock_cm_cls.return_value.load.return_value = mock_cfg
            rt = AnimusRuntime(config=None)
            assert rt.config is mock_cfg

    def test_start_creates_data_directory(self, tmp_path: Path) -> None:
        """start() creates the data directory."""
        data_dir = tmp_path / "data"
        cfg = _make_config(
            data_dir=str(data_dir),
            intelligence_enabled=False,
            proactive_enabled=False,
            memory_db_path=str(data_dir / "int.db"),
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert data_dir.exists()
        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_start_creates_session_manager(self, tmp_path: Path) -> None:
        """start() creates a SessionManager."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=False,
            proactive_enabled=False,
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert rt.session_manager is not None
        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_start_sets_started_true(self, tmp_path: Path) -> None:
        """start() sets started to True."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=False,
            proactive_enabled=False,
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert rt.started is True
        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_start_creates_cognitive_backend(self, tmp_path: Path) -> None:
        """start() creates a cognitive backend."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=False,
            proactive_enabled=False,
            backend="ollama",
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert rt.cognitive_backend is not None
        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_start_intelligence_disabled_skips_components(self, tmp_path: Path) -> None:
        """When intelligence.enabled is False, memory/tools/automations are None."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=False,
            proactive_enabled=False,
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert rt.memory_manager is None
        assert rt.tool_executor is None
        assert rt.automation_engine is None
        assert rt.proactive_engine is None
        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_start_intelligence_enabled_creates_memory_manager(self, tmp_path: Path) -> None:
        """Memory manager is created when intelligence is enabled."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=True,
            proactive_enabled=False,
            memory_db_path=str(tmp_path / "int.db"),
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert rt.memory_manager is not None
        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_start_intelligence_enabled_creates_tool_executor(self, tmp_path: Path) -> None:
        """Tool executor is created with built-in tools when intelligence is enabled."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=True,
            proactive_enabled=False,
            memory_db_path=str(tmp_path / "int.db"),
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert rt.tool_executor is not None
        # Built-in tools should be registered
        assert len(rt.tool_executor.list_tools()) > 0
        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_start_intelligence_enabled_creates_automation_engine(self, tmp_path: Path) -> None:
        """Automation engine is created when intelligence is enabled."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=True,
            proactive_enabled=False,
            memory_db_path=str(tmp_path / "int.db"),
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert rt.automation_engine is not None
        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_start_creates_intelligent_router_when_components_available(
        self, tmp_path: Path
    ) -> None:
        """IntelligentRouter is used when intelligence components are available."""
        from animus_bootstrap.intelligence.router import IntelligentRouter

        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=True,
            proactive_enabled=False,
            memory_db_path=str(tmp_path / "int.db"),
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert isinstance(rt.router, IntelligentRouter)
        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_start_creates_basic_router_when_intelligence_disabled(self, tmp_path: Path) -> None:
        """MessageRouter is used when intelligence is disabled."""
        from animus_bootstrap.gateway.router import MessageRouter

        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=False,
            proactive_enabled=False,
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert isinstance(rt.router, MessageRouter)
        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_start_proactive_enabled_creates_engine(self, tmp_path: Path) -> None:
        """Proactive engine is created and started when both flags are True."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=True,
            proactive_enabled=True,
            memory_db_path=str(tmp_path / "int.db"),
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert rt.proactive_engine is not None
        assert rt.proactive_engine.running is True
        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_start_proactive_disabled_skips_engine(self, tmp_path: Path) -> None:
        """Proactive engine is not created when proactive.enabled is False."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=True,
            proactive_enabled=False,
            memory_db_path=str(tmp_path / "int.db"),
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert rt.proactive_engine is None
        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_double_start_warns(self, tmp_path: Path) -> None:
        """Calling start() twice logs a warning and returns early."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=False,
            proactive_enabled=False,
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())

        with patch("animus_bootstrap.runtime.logger") as mock_logger:
            asyncio.get_event_loop().run_until_complete(rt.start())
            mock_logger.warning.assert_called_once_with("Runtime already started")

        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_stop_closes_all_components(self, tmp_path: Path) -> None:
        """stop() closes all initialized components."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=True,
            proactive_enabled=True,
            memory_db_path=str(tmp_path / "int.db"),
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())

        # Verify components are alive
        assert rt.started is True
        assert rt.session_manager is not None
        assert rt.memory_manager is not None
        assert rt.automation_engine is not None
        assert rt.proactive_engine is not None

        asyncio.get_event_loop().run_until_complete(rt.stop())
        assert rt.started is False

    def test_stop_safe_when_not_started(self) -> None:
        """stop() is a no-op when runtime was never started."""
        rt = AnimusRuntime(config=_make_config())
        # Should not raise
        asyncio.get_event_loop().run_until_complete(rt.stop())
        assert rt.started is False

    def test_stop_twice_is_safe(self, tmp_path: Path) -> None:
        """Calling stop() twice does not raise."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=False,
            proactive_enabled=False,
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        asyncio.get_event_loop().run_until_complete(rt.stop())
        # Second stop is a no-op
        asyncio.get_event_loop().run_until_complete(rt.stop())
        assert rt.started is False


# ------------------------------------------------------------------
# TestCognitiveBackendCreation
# ------------------------------------------------------------------


class TestCognitiveBackendCreation:
    """Tests for _create_cognitive_backend."""

    def test_anthropic_backend_with_key(self) -> None:
        """Anthropic backend is created when key is present."""
        from animus_bootstrap.gateway.cognitive import AnthropicBackend

        cfg = _make_config(backend="anthropic", anthropic_key="sk-ant-" + "x" * 90)
        rt = AnimusRuntime(config=cfg)
        result = rt._create_cognitive_backend()
        assert isinstance(result, AnthropicBackend)

    def test_anthropic_backend_without_key_falls_back(self) -> None:
        """Anthropic backend without key falls back to Ollama."""
        from animus_bootstrap.gateway.cognitive import OllamaBackend

        cfg = _make_config(backend="anthropic", anthropic_key="")
        rt = AnimusRuntime(config=cfg)
        result = rt._create_cognitive_backend()
        assert isinstance(result, OllamaBackend)

    def test_ollama_backend(self) -> None:
        """Ollama backend is created when configured."""
        from animus_bootstrap.gateway.cognitive import OllamaBackend

        cfg = _make_config(backend="ollama")
        rt = AnimusRuntime(config=cfg)
        result = rt._create_cognitive_backend()
        assert isinstance(result, OllamaBackend)

    def test_forge_backend_when_enabled(self) -> None:
        """Forge backend is created when forge is enabled."""
        from animus_bootstrap.gateway.cognitive import ForgeBackend

        cfg = _make_config(backend="forge", forge_enabled=True)
        rt = AnimusRuntime(config=cfg)
        result = rt._create_cognitive_backend()
        assert isinstance(result, ForgeBackend)

    def test_forge_backend_disabled_falls_back(self) -> None:
        """Forge backend falls back to Ollama when forge is disabled."""
        from animus_bootstrap.gateway.cognitive import OllamaBackend

        cfg = _make_config(backend="forge", forge_enabled=False)
        rt = AnimusRuntime(config=cfg)
        result = rt._create_cognitive_backend()
        assert isinstance(result, OllamaBackend)

    def test_unknown_backend_falls_back(self) -> None:
        """Unknown backend type falls back to Ollama with warning."""
        from animus_bootstrap.gateway.cognitive import OllamaBackend

        cfg = _make_config(backend="unknown-thing")
        rt = AnimusRuntime(config=cfg)
        with patch("animus_bootstrap.runtime.logger") as mock_logger:
            result = rt._create_cognitive_backend()
            mock_logger.warning.assert_called_once()
        assert isinstance(result, OllamaBackend)


# ------------------------------------------------------------------
# TestMemoryManagerCreation
# ------------------------------------------------------------------


class TestMemoryManagerCreation:
    """Tests for _create_memory_manager."""

    def test_sqlite_backend(self, tmp_path: Path) -> None:
        """SQLite memory backend is created by default."""
        from animus_bootstrap.intelligence.memory import MemoryManager

        cfg = _make_config(
            memory_backend="sqlite",
            memory_db_path=str(tmp_path / "mem.db"),
        )
        rt = AnimusRuntime(config=cfg)
        result = rt._create_memory_manager()
        assert isinstance(result, MemoryManager)
        result.close()

    def test_unknown_backend_defaults_to_sqlite(self, tmp_path: Path) -> None:
        """Unknown memory backend defaults to SQLite."""
        from animus_bootstrap.intelligence.memory import MemoryManager

        cfg = _make_config(
            memory_backend="nonexistent",
            memory_db_path=str(tmp_path / "mem.db"),
        )
        rt = AnimusRuntime(config=cfg)
        result = rt._create_memory_manager()
        assert isinstance(result, MemoryManager)
        result.close()

    def test_chromadb_backend_raises_not_implemented(self) -> None:
        """ChromaDB backend raises NotImplementedError (stub)."""
        cfg = _make_config(memory_backend="chromadb")
        rt = AnimusRuntime(config=cfg)
        # ChromaDB stub raises NotImplementedError (or RuntimeError if not installed)
        with pytest.raises((NotImplementedError, RuntimeError)):
            rt._create_memory_manager()

    def test_animus_backend_raises_not_implemented(self) -> None:
        """Animus backend raises (not implemented or import error)."""
        cfg = _make_config(memory_backend="animus")
        rt = AnimusRuntime(config=cfg)
        with pytest.raises((NotImplementedError, RuntimeError)):
            rt._create_memory_manager()

    def test_memory_db_parent_created(self, tmp_path: Path) -> None:
        """Parent directory for memory DB is created if missing."""
        db_path = tmp_path / "deep" / "nested" / "mem.db"
        cfg = _make_config(
            memory_backend="sqlite",
            memory_db_path=str(db_path),
        )
        rt = AnimusRuntime(config=cfg)
        result = rt._create_memory_manager()
        assert db_path.parent.exists()
        result.close()


# ------------------------------------------------------------------
# TestToolExecutorCreation
# ------------------------------------------------------------------


class TestToolExecutorCreation:
    """Tests for _create_tool_executor."""

    def test_creates_executor_with_builtin_tools(self) -> None:
        """Tool executor is created with built-in tools registered."""
        from animus_bootstrap.intelligence.tools.executor import ToolExecutor

        cfg = _make_config()
        rt = AnimusRuntime(config=cfg)
        result = rt._create_tool_executor()
        assert isinstance(result, ToolExecutor)
        assert len(result.list_tools()) > 0

    def test_executor_respects_max_calls_config(self) -> None:
        """Tool executor uses max_tool_calls_per_turn from config."""
        cfg = _make_config()
        cfg.intelligence.max_tool_calls_per_turn = 10
        rt = AnimusRuntime(config=cfg)
        result = rt._create_tool_executor()
        assert result._max_calls == 10

    def test_executor_respects_timeout_config(self) -> None:
        """Tool executor uses tool_timeout_seconds from config."""
        cfg = _make_config()
        cfg.intelligence.tool_timeout_seconds = 60
        rt = AnimusRuntime(config=cfg)
        result = rt._create_tool_executor()
        assert result._timeout == 60.0


# ------------------------------------------------------------------
# TestRouterCreation
# ------------------------------------------------------------------


class TestRouterCreation:
    """Tests for _create_router."""

    def test_intelligent_router_when_memory_available(self, tmp_path: Path) -> None:
        """IntelligentRouter is created when memory_manager is set."""
        from animus_bootstrap.intelligence.router import IntelligentRouter

        cfg = _make_config(
            data_dir=str(tmp_path),
            memory_db_path=str(tmp_path / "int.db"),
        )
        rt = AnimusRuntime(config=cfg)
        rt.session_manager = MagicMock()
        rt.cognitive_backend = MagicMock()
        rt.memory_manager = MagicMock()
        result = rt._create_router()
        assert isinstance(result, IntelligentRouter)

    def test_intelligent_router_when_tool_executor_available(self, tmp_path: Path) -> None:
        """IntelligentRouter is created when tool_executor is set."""
        from animus_bootstrap.intelligence.router import IntelligentRouter

        cfg = _make_config(data_dir=str(tmp_path))
        rt = AnimusRuntime(config=cfg)
        rt.session_manager = MagicMock()
        rt.cognitive_backend = MagicMock()
        rt.tool_executor = MagicMock()
        result = rt._create_router()
        assert isinstance(result, IntelligentRouter)

    def test_intelligent_router_when_automation_engine_available(self) -> None:
        """IntelligentRouter is created when automation_engine is set."""
        from animus_bootstrap.intelligence.router import IntelligentRouter

        cfg = _make_config()
        rt = AnimusRuntime(config=cfg)
        rt.session_manager = MagicMock()
        rt.cognitive_backend = MagicMock()
        rt.automation_engine = MagicMock()
        result = rt._create_router()
        assert isinstance(result, IntelligentRouter)

    def test_basic_router_when_no_intelligence_components(self) -> None:
        """MessageRouter is created when no intelligence components are set."""
        from animus_bootstrap.gateway.router import MessageRouter

        cfg = _make_config()
        rt = AnimusRuntime(config=cfg)
        rt.session_manager = MagicMock()
        rt.cognitive_backend = MagicMock()
        # All intelligence components are None by default
        result = rt._create_router()
        assert isinstance(result, MessageRouter)


# ------------------------------------------------------------------
# TestProactiveEngineCreation
# ------------------------------------------------------------------


class TestProactiveEngineCreation:
    """Tests for _create_proactive_engine."""

    def test_engine_started(self, tmp_path: Path) -> None:
        """Proactive engine is started after creation."""
        from animus_bootstrap.intelligence.proactive.engine import ProactiveEngine

        cfg = _make_config(data_dir=str(tmp_path))
        rt = AnimusRuntime(config=cfg)
        rt.router = MagicMock()
        engine = asyncio.get_event_loop().run_until_complete(rt._create_proactive_engine())
        assert isinstance(engine, ProactiveEngine)
        assert engine.running is True
        # Clean up
        asyncio.get_event_loop().run_until_complete(engine.stop())
        engine.close()

    def test_engine_registers_builtin_checks(self, tmp_path: Path) -> None:
        """Proactive engine has built-in checks registered."""
        cfg = _make_config(data_dir=str(tmp_path))
        rt = AnimusRuntime(config=cfg)
        rt.router = MagicMock()
        engine = asyncio.get_event_loop().run_until_complete(rt._create_proactive_engine())
        checks = engine.list_checks()
        assert len(checks) >= 3  # morning_brief, task_nudge, calendar
        asyncio.get_event_loop().run_until_complete(engine.stop())
        engine.close()

    def test_engine_applies_check_config_overrides(self, tmp_path: Path) -> None:
        """Check config overrides from proactive.checks are applied."""
        cfg = _make_config(data_dir=str(tmp_path))
        cfg.proactive.checks["morning_brief"] = ProactiveCheckConfig(
            enabled=False,
            schedule="0 8 * * *",
            channels=["telegram"],
        )
        rt = AnimusRuntime(config=cfg)
        rt.router = MagicMock()
        engine = asyncio.get_event_loop().run_until_complete(rt._create_proactive_engine())

        # Find the morning_brief check
        morning = None
        for check in engine.list_checks():
            if check.name == "morning_brief":
                morning = check
                break

        assert morning is not None
        assert morning.enabled is False
        assert morning.schedule == "0 8 * * *"
        assert morning.channels == ["telegram"]

        asyncio.get_event_loop().run_until_complete(engine.stop())
        engine.close()


# ------------------------------------------------------------------
# TestRuntimeSingleton
# ------------------------------------------------------------------


class TestRuntimeSingleton:
    """Tests for module-level singleton functions."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_runtime()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_runtime()

    def test_get_runtime_returns_singleton(self) -> None:
        """get_runtime() returns the same instance on repeated calls."""
        with patch("animus_bootstrap.runtime.ConfigManager") as mock_cm:
            mock_cm.return_value.load.return_value = _make_config()
            rt1 = get_runtime()
            rt2 = get_runtime()
            assert rt1 is rt2

    def test_set_runtime_overrides_singleton(self) -> None:
        """set_runtime() overrides the cached instance."""
        custom = AnimusRuntime(config=_make_config())
        set_runtime(custom)
        assert get_runtime() is custom

    def test_reset_runtime_clears_singleton(self) -> None:
        """reset_runtime() clears the cached instance."""
        with patch("animus_bootstrap.runtime.ConfigManager") as mock_cm:
            mock_cm.return_value.load.return_value = _make_config()
            rt1 = get_runtime()
            reset_runtime()
            rt2 = get_runtime()
            assert rt1 is not rt2

    def test_get_runtime_creates_new_instance(self) -> None:
        """get_runtime() creates a new AnimusRuntime when singleton is None."""
        with patch("animus_bootstrap.runtime.ConfigManager") as mock_cm:
            mock_cm.return_value.load.return_value = _make_config()
            rt = get_runtime()
            assert isinstance(rt, AnimusRuntime)


# ------------------------------------------------------------------
# TestHealthEndpoint
# ------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def setup_method(self) -> None:
        """Reset runtime singleton before each test."""
        reset_runtime()

    def teardown_method(self) -> None:
        """Reset runtime singleton after each test."""
        reset_runtime()

    def test_health_returns_ok_when_started(self, tmp_path: Path) -> None:
        """GET /health returns 'ok' when runtime is started."""
        from animus_bootstrap.dashboard.app import app

        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=True,
            proactive_enabled=False,
            memory_db_path=str(tmp_path / "int.db"),
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        app.state.runtime = rt

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.4.0"
        assert data["components"]["memory"] is True
        assert data["components"]["tools"] is True
        assert data["components"]["automations"] is True
        assert data["components"]["proactive"] is False

        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_health_returns_degraded_when_no_runtime(self) -> None:
        """GET /health returns 'degraded' when no runtime is set."""
        from animus_bootstrap.dashboard.app import app

        # Remove runtime from state if present
        if hasattr(app.state, "runtime"):
            delattr(app.state, "runtime")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["components"]["memory"] is False
        assert data["components"]["tools"] is False

    def test_health_returns_degraded_when_not_started(self) -> None:
        """GET /health returns 'degraded' when runtime exists but not started."""
        from animus_bootstrap.dashboard.app import app

        rt = AnimusRuntime(config=_make_config())
        app.state.runtime = rt

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"


# ------------------------------------------------------------------
# TestDashboardLifespan
# ------------------------------------------------------------------


class TestDashboardLifespan:
    """Tests for the FastAPI lifespan integration."""

    def setup_method(self) -> None:
        """Reset runtime singleton before each test."""
        reset_runtime()

    def teardown_method(self) -> None:
        """Reset runtime singleton after each test."""
        reset_runtime()

    def test_lifespan_starts_and_stops_runtime(self, tmp_path: Path) -> None:
        """The lifespan context manager starts and stops the runtime."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=False,
            proactive_enabled=False,
        )
        rt = AnimusRuntime(config=cfg)
        set_runtime(rt)

        from animus_bootstrap.dashboard.app import app

        with TestClient(app) as _client:
            # During lifespan, runtime should be started
            runtime = getattr(app.state, "runtime", None)
            assert runtime is not None
            assert runtime.started is True

        # After exit, runtime should be stopped
        assert rt.started is False

    def test_lifespan_handles_start_failure(self) -> None:
        """The lifespan gracefully handles runtime.start() failure."""
        rt = MagicMock(spec=AnimusRuntime)
        rt.start = AsyncMock(side_effect=RuntimeError("boom"))
        rt.started = False
        set_runtime(rt)

        from animus_bootstrap.dashboard.app import app

        # Should not raise — dashboard runs in limited mode
        with TestClient(app):
            runtime = getattr(app.state, "runtime", None)
            assert runtime is rt


# ------------------------------------------------------------------
# TestDashboardRoutersWithRuntime
# ------------------------------------------------------------------


class TestDashboardRoutersWithRuntime:
    """Tests verifying routers pull data from runtime when available."""

    def setup_method(self) -> None:
        reset_runtime()

    def teardown_method(self) -> None:
        reset_runtime()

    def test_tools_page_with_runtime_tools(self, tmp_path: Path) -> None:
        """Tools page populates tools from runtime.tool_executor."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=True,
            proactive_enabled=False,
            memory_db_path=str(tmp_path / "int.db"),
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        set_runtime(rt)

        from animus_bootstrap.dashboard.app import app

        app.state.runtime = rt
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/tools")
        assert resp.status_code == 200

        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_automations_page_with_runtime(self, tmp_path: Path) -> None:
        """Automations page pulls rules from runtime.automation_engine."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=True,
            proactive_enabled=False,
            memory_db_path=str(tmp_path / "int.db"),
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        set_runtime(rt)

        from animus_bootstrap.dashboard.app import app

        app.state.runtime = rt
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/automations")
        assert resp.status_code == 200

        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_activity_page_with_runtime(self, tmp_path: Path) -> None:
        """Activity page pulls data from runtime.proactive_engine."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=True,
            proactive_enabled=True,
            memory_db_path=str(tmp_path / "int.db"),
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        set_runtime(rt)

        from animus_bootstrap.dashboard.app import app

        app.state.runtime = rt
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/activity")
        assert resp.status_code == 200

        asyncio.get_event_loop().run_until_complete(rt.stop())


# ------------------------------------------------------------------
# TestHomeRouterRuntime
# ------------------------------------------------------------------


class TestHomeRouterRuntime:
    """Tests for home router runtime integration."""

    def test_count_runtime_components_none(self) -> None:
        """_count_runtime_components returns 0 for None runtime."""
        from animus_bootstrap.dashboard.routers.home import _count_runtime_components

        assert _count_runtime_components(None) == 0

    def test_count_runtime_components_all_set(self) -> None:
        """_count_runtime_components counts all populated components."""
        from animus_bootstrap.dashboard.routers.home import _count_runtime_components

        rt = MagicMock()
        rt.memory_manager = MagicMock()
        rt.tool_executor = MagicMock()
        rt.proactive_engine = MagicMock()
        rt.automation_engine = MagicMock()
        assert _count_runtime_components(rt) == 4

    def test_count_runtime_components_partial(self) -> None:
        """_count_runtime_components counts only non-None components."""
        from animus_bootstrap.dashboard.routers.home import _count_runtime_components

        rt = MagicMock()
        rt.memory_manager = MagicMock()
        rt.tool_executor = None
        rt.proactive_engine = None
        rt.automation_engine = MagicMock()
        assert _count_runtime_components(rt) == 2


# ------------------------------------------------------------------
# TestServeFunction
# ------------------------------------------------------------------


class TestServeFunction:
    """Tests for the serve() function."""

    def test_serve_reads_port_from_config(self) -> None:
        """serve() reads port from config and passes it to uvicorn."""
        cfg = _make_config()
        cfg.services.port = 8080

        with (
            patch("animus_bootstrap.config.ConfigManager") as mock_cm,
            patch("uvicorn.run") as mock_run,
        ):
            mock_cm.return_value.load.return_value = cfg
            from animus_bootstrap.dashboard.app import serve

            serve()
            mock_run.assert_called_once_with(
                "animus_bootstrap.dashboard.app:app",
                host="0.0.0.0",
                port=8080,
                reload=False,
            )


# ------------------------------------------------------------------
# TestPersonaEngineCreation
# ------------------------------------------------------------------


class TestPersonaEngineCreation:
    """Tests for _create_persona_engine."""

    def test_persona_engine_creation(self) -> None:
        """Persona engine is created with default persona."""
        from animus_bootstrap.personas.engine import PersonaEngine

        cfg = _make_config()
        rt = AnimusRuntime(config=cfg)
        engine = rt._create_persona_engine()
        assert isinstance(engine, PersonaEngine)
        assert engine.persona_count >= 1

        # Default persona has the expected name
        default = engine.get_default()
        assert default is not None
        assert default.name == "Animus"
        assert default.is_default is True

    def test_persona_engine_from_config_profiles(self) -> None:
        """Persona engine registers named profiles from config."""
        from animus_bootstrap.personas.engine import PersonaEngine

        cfg = _make_config()
        cfg.personas = PersonasSection(
            default_name="MyBot",
            default_tone="formal",
            default_system_prompt="You are MyBot.",
            profiles={
                "coder": PersonaProfileConfig(
                    name="CodeBot",
                    description="Coding assistant",
                    system_prompt="You help with code.",
                    tone="technical",
                    knowledge_domains=["python", "rust"],
                    channel_bindings={"discord": True},
                ),
                "writer": PersonaProfileConfig(
                    name="WriteBot",
                    tone="creative",
                ),
            },
        )
        rt = AnimusRuntime(config=cfg)
        engine = rt._create_persona_engine()
        assert isinstance(engine, PersonaEngine)
        # Default + 2 profiles = 3 personas
        assert engine.persona_count == 3

        # Default persona
        default = engine.get_default()
        assert default is not None
        assert default.name == "MyBot"

        # Named profiles are registered
        all_names = [p.name for p in engine.list_personas()]
        assert "CodeBot" in all_names
        assert "WriteBot" in all_names

    def test_persona_engine_initialized_on_start(self, tmp_path: Path) -> None:
        """start() creates persona_engine when personas.enabled is True."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=False,
            proactive_enabled=False,
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert rt.persona_engine is not None
        assert rt.context_adapter is not None
        assert rt.persona_engine.persona_count >= 1
        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_persona_engine_disabled(self, tmp_path: Path) -> None:
        """When personas.enabled is False, persona_engine is None."""
        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=False,
            proactive_enabled=False,
        )
        cfg.personas.enabled = False
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert rt.persona_engine is None
        assert rt.context_adapter is None
        asyncio.get_event_loop().run_until_complete(rt.stop())

    def test_persona_engine_passed_to_intelligent_router(self, tmp_path: Path) -> None:
        """IntelligentRouter receives persona_engine when both intelligence and personas enabled."""
        from animus_bootstrap.intelligence.router import IntelligentRouter

        cfg = _make_config(
            data_dir=str(tmp_path),
            intelligence_enabled=True,
            proactive_enabled=False,
            memory_db_path=str(tmp_path / "int.db"),
        )
        rt = AnimusRuntime(config=cfg)
        asyncio.get_event_loop().run_until_complete(rt.start())
        assert isinstance(rt.router, IntelligentRouter)
        assert rt.router._persona_engine is not None
        assert rt.router._context_adapter is not None
        asyncio.get_event_loop().run_until_complete(rt.stop())
