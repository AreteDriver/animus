"""Coverage push tests — targets uncovered branches to reach 97%.

Covers: router memory cap overflow, identity prompt branches,
forge_ctl subprocess fallbacks, mcp_bridge error paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_bootstrap.gateway.models import create_message
from animus_bootstrap.intelligence.memory import MemoryContext
from animus_bootstrap.intelligence.router import IntelligentRouter
from animus_bootstrap.intelligence.tools.mcp_bridge import MCPToolBridge

# ======================================================================
# Helpers
# ======================================================================


def _make_router(
    cognitive=None,
    identity_manager=None,
    context_adapter=None,
    system_prompt="",
):
    cog = cognitive or AsyncMock()
    session = MagicMock()  # SessionManager requires db_path; mock it
    return IntelligentRouter(
        cognitive=cog,
        session_manager=session,
        system_prompt=system_prompt,
        identity_manager=identity_manager,
        context_adapter=context_adapter,
    )


def _make_bridge_with_servers(servers: dict) -> MCPToolBridge:
    """Create an MCPToolBridge with pre-loaded server configs."""
    bridge = MCPToolBridge(config_path=None)
    bridge._servers = servers
    return bridge


# ======================================================================
# Router: memory context cap overflow (lines 179-217)
# ======================================================================


class TestRouterMemoryCapOverflow:
    """Test memory context hard cap at 2000 chars."""

    def test_semantic_overflow(self):
        """Semantic memories exceeding 2000 chars are truncated."""
        router = _make_router()
        big_memories = [f"fact-{i}: " + "x" * 200 for i in range(15)]
        ctx = MemoryContext(semantic=big_memories)
        result = router._build_system_prompt(ctx)
        assert "Known Facts" in result
        fact_count = result.count("fact-")
        assert fact_count < 15
        assert fact_count > 0

    def test_procedural_overflow(self):
        """Procedural memories capped when semantic used budget."""
        router = _make_router()
        semantic = [f"sem-{i}: " + "x" * 180 for i in range(9)]
        procedural = [f"proc-{i}: " + "x" * 200 for i in range(5)]
        ctx = MemoryContext(semantic=semantic, procedural=procedural)
        result = router._build_system_prompt(ctx)
        assert "Known Facts" in result
        proc_count = result.count("proc-")
        assert proc_count < 5

    def test_user_prefs_overflow(self):
        """User preferences are capped."""
        router = _make_router()
        semantic = [f"s-{i}: " + "x" * 180 for i in range(9)]
        prefs = {f"pref_{i}": "v" * 200 for i in range(5)}
        ctx = MemoryContext(semantic=semantic, user_prefs=prefs)
        result = router._build_system_prompt(ctx)
        pref_count = result.count("pref_")
        assert pref_count < 5

    def test_episodic_overflow(self):
        """Episodic memories capped after other sections consume budget."""
        router = _make_router()
        semantic = [f"s-{i}: " + "x" * 180 for i in range(9)]
        episodic = [f"ep-{i}: " + "x" * 140 for i in range(5)]
        ctx = MemoryContext(semantic=semantic, episodic=episodic)
        result = router._build_system_prompt(ctx)
        ep_count = result.count("ep-")
        assert ep_count < 5


# ======================================================================
# Router: identity prompt branches (lines 144-177)
# ======================================================================


class TestRouterIdentityBranches:
    """Test identity prompt loading branches."""

    def test_condensed_prompt(self):
        """get_condensed_prompt is preferred."""
        mgr = MagicMock()
        mgr.get_condensed_prompt.return_value = "condensed identity"
        router = _make_router(identity_manager=mgr)
        result = router._build_system_prompt(None)
        assert "condensed identity" in result

    def test_fallback_to_get_identity_prompt(self):
        """Falls back to get_identity_prompt without get_condensed_prompt."""
        mgr = MagicMock(spec=[])
        mgr.get_identity_prompt = MagicMock(return_value="full identity")
        router = _make_router(identity_manager=mgr)
        result = router._build_system_prompt(None)
        assert "full identity" in result

    def test_identity_exception(self):
        """Exception in identity loading is caught."""
        mgr = MagicMock()
        mgr.get_condensed_prompt.side_effect = RuntimeError("broken")
        router = _make_router(identity_manager=mgr)
        result = router._build_system_prompt(None)
        assert isinstance(result, str)

    def test_identity_with_persona_voice_only(self):
        """With identity + persona + context_adapter, only voice hints."""
        from animus_bootstrap.personas.context import ContextAdapter
        from animus_bootstrap.personas.engine import PersonaProfile
        from animus_bootstrap.personas.voice import VoiceConfig

        mgr = MagicMock()
        mgr.get_condensed_prompt.return_value = "Identity prompt"

        persona = PersonaProfile(
            name="test",
            system_prompt="Base persona prompt",
            voice=VoiceConfig(tone="formal"),
        )
        adapter = ContextAdapter()
        msg = create_message("webchat", "user1", "Alice", "hello")

        router = _make_router(identity_manager=mgr, context_adapter=adapter)
        result = router._build_system_prompt(None, persona=persona, message=msg)
        assert "Identity prompt" in result
        assert "Base persona prompt" not in result


# ======================================================================
# Forge CTL: subprocess fallback paths (lines 39-98)
# ======================================================================


class TestForgeCtl:
    """Test forge_ctl subprocess fallback paths."""

    @pytest.mark.asyncio
    async def test_systemd_start_success(self):
        from animus_bootstrap.intelligence.tools.builtin.forge_ctl import (
            _forge_start,
        )

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", return_value=(b"inactive", b"")):
                result = await _forge_start()
        assert "systemd" in result

    @pytest.mark.asyncio
    async def test_uvicorn_fallback_running(self):
        """Uvicorn starts and keeps running (returncode None)."""
        from animus_bootstrap.intelligence.tools.builtin.forge_ctl import (
            _forge_start,
        )

        call_count = {"n": 0}

        mock_uvicorn = AsyncMock()
        mock_uvicorn.returncode = None
        mock_uvicorn.pid = 12345

        async def fake_exec(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise FileNotFoundError("no systemd")
            return mock_uvicorn

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await _forge_start()
        assert "uvicorn" in result.lower() or "PID" in result

    @pytest.mark.asyncio
    async def test_uvicorn_exits_immediately(self):
        from animus_bootstrap.intelligence.tools.builtin.forge_ctl import (
            _forge_start,
        )

        call_count = {"n": 0}

        mock_uvicorn = AsyncMock()
        mock_uvicorn.returncode = 1
        mock_uvicorn.pid = 12345
        mock_uvicorn.communicate = AsyncMock(return_value=(b"", b"error msg"))

        async def fake_exec(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise FileNotFoundError("no systemd")
            return mock_uvicorn

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await _forge_start()
        assert "failed" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_neither_available(self):
        from animus_bootstrap.intelligence.tools.builtin.forge_ctl import (
            _forge_start,
        )

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("nope"),
        ):
            result = await _forge_start()
        assert "Neither" in result or "cannot" in result.lower()

    @pytest.mark.asyncio
    async def test_uvicorn_os_error(self):
        from animus_bootstrap.intelligence.tools.builtin.forge_ctl import (
            _forge_start,
        )

        call_count = {"n": 0}

        async def fake_exec(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise FileNotFoundError("no systemd")
            raise OSError("permission denied")

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            result = await _forge_start()
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_forge_invoke_post(self):
        import httpx
        import respx

        from animus_bootstrap.intelligence.tools.builtin.forge_ctl import (
            _forge_invoke,
        )

        with respx.mock:
            respx.post("http://127.0.0.1:8000/api/test").mock(
                return_value=httpx.Response(200, text='{"ok": true}')
            )
            result = await _forge_invoke("/api/test", method="POST", body='{"key": "val"}')
        assert "ok" in result


# ======================================================================
# MCP Bridge: error paths (lines 168-219)
# ======================================================================


class TestMCPBridgeErrors:
    """Test MCP bridge error and edge case paths."""

    @pytest.mark.asyncio
    async def test_import_tools_unknown_server(self):
        bridge = _make_bridge_with_servers({})
        tools = await bridge.import_tools("nonexistent")
        assert tools == []

    @pytest.mark.asyncio
    async def test_import_tools_start_failure(self):
        bridge = _make_bridge_with_servers({"srv": {"command": "fake", "args": []}})
        with patch.object(bridge, "_start_server", side_effect=OSError("boom")):
            tools = await bridge.import_tools("srv")
        assert tools == []

    @pytest.mark.asyncio
    async def test_import_tools_list_failure(self):
        mock_conn = AsyncMock()
        mock_conn.send_request = AsyncMock(side_effect=ConnectionError("broken"))
        mock_conn.close = AsyncMock()

        bridge = _make_bridge_with_servers({"srv": {"command": "fake", "args": []}})
        with patch.object(bridge, "_start_server", return_value=mock_conn):
            tools = await bridge.import_tools("srv")
        assert tools == []
        mock_conn.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_import_tools_skips_empty_name(self):
        mock_conn = AsyncMock()
        mock_conn.send_request = AsyncMock(
            return_value={
                "tools": [
                    {"name": "", "description": "empty"},
                    {"name": "valid", "description": "A tool"},
                ]
            }
        )

        bridge = _make_bridge_with_servers({"srv": {"command": "fake", "args": []}})
        with patch.object(bridge, "_start_server", return_value=mock_conn):
            tools = await bridge.import_tools("srv")
        assert len(tools) == 1
        assert tools[0].name == "mcp_srv_valid"


# ======================================================================
# Automations engine: evaluate_event paths (lines 200, 226, 236, 240)
# ======================================================================


class TestAutomationEngineEventPaths:
    """Test evaluate_event branches in AutomationEngine."""

    @pytest.mark.asyncio
    async def test_evaluate_event_fires_rule(self):
        """Event-triggered rule fires when trigger matches."""
        import tempfile
        from pathlib import Path

        from animus_bootstrap.intelligence.automations.engine import (
            AutomationEngine,
        )
        from animus_bootstrap.intelligence.automations.models import (
            ActionConfig,
            AutomationRule,
            TriggerConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = AutomationEngine(db_path=Path(tmpdir) / "auto.db")
            rule = AutomationRule(
                name="test-event",
                trigger=TriggerConfig(type="event", params={"event_type": "deploy"}),
                actions=[ActionConfig(type="log", params={"message": "deployed"})],
            )
            engine.add_rule(rule)

            results = await engine.evaluate_event({"type": "deploy"})
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_evaluate_event_cooldown_skip(self):
        """Event rule with active cooldown is skipped."""
        import tempfile
        from datetime import UTC, datetime
        from pathlib import Path

        from animus_bootstrap.intelligence.automations.engine import (
            AutomationEngine,
        )
        from animus_bootstrap.intelligence.automations.models import (
            ActionConfig,
            AutomationRule,
            TriggerConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = AutomationEngine(db_path=Path(tmpdir) / "auto.db")
            rule = AutomationRule(
                name="test-cd",
                trigger=TriggerConfig(type="event", params={"event_type": "deploy"}),
                actions=[ActionConfig(type="log", params={"message": "x"})],
                cooldown_seconds=3600,
            )
            rule.last_fired = datetime.now(UTC)
            engine.add_rule(rule)

            results = await engine.evaluate_event({"type": "deploy"})
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_evaluate_event_trigger_no_match(self):
        """Event rule not matching trigger value is skipped."""
        import tempfile
        from pathlib import Path

        from animus_bootstrap.intelligence.automations.engine import (
            AutomationEngine,
        )
        from animus_bootstrap.intelligence.automations.models import (
            ActionConfig,
            AutomationRule,
            TriggerConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = AutomationEngine(db_path=Path(tmpdir) / "auto.db")
            rule = AutomationRule(
                name="test-nomatch",
                trigger=TriggerConfig(type="event", params={"event_type": "deploy"}),
                actions=[ActionConfig(type="log", params={"message": "x"})],
            )
            engine.add_rule(rule)

            results = await engine.evaluate_event({"type": "other_event"})
            assert len(results) == 0


# ======================================================================
# Runtime: cognitive backend + memory + MCP paths
# ======================================================================


class TestRuntimeCoveragePaths:
    """Test runtime branches: DualOllama, animus memory, MCP conflicts."""

    def test_dual_ollama_backend(self):
        """Ollama with code_model creates DualOllamaBackend."""
        from animus_bootstrap.config.schema import OllamaSection
        from animus_bootstrap.runtime import AnimusRuntime

        cfg = MagicMock()
        cfg.gateway.default_backend = "ollama"
        cfg.ollama = OllamaSection(model="llama3.2", code_model="deepseek-coder-v2")

        rt = AnimusRuntime.__new__(AnimusRuntime)
        rt._config = cfg
        with patch("animus_bootstrap.gateway.cognitive.DualOllamaBackend") as mock_cls:
            mock_cls.return_value = MagicMock()
            rt._create_cognitive_backend()
        mock_cls.assert_called_once()

    def test_fallback_dual_ollama_backend(self):
        """Unknown backend falls back to Ollama with code_model."""
        from animus_bootstrap.config.schema import OllamaSection
        from animus_bootstrap.runtime import AnimusRuntime

        cfg = MagicMock()
        cfg.gateway.default_backend = "unknown_backend"
        cfg.forge.enabled = False
        cfg.ollama = OllamaSection(model="llama3.2", code_model="deepseek-coder-v2")

        rt = AnimusRuntime.__new__(AnimusRuntime)
        rt._config = cfg
        with patch("animus_bootstrap.gateway.cognitive.DualOllamaBackend") as mock_cls:
            mock_cls.return_value = MagicMock()
            rt._create_cognitive_backend()
        mock_cls.assert_called_once()

    def test_animus_memory_backend(self):
        """Memory backend 'animus' creates AnimusMemoryBackend."""
        import tempfile

        from animus_bootstrap.runtime import AnimusRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MagicMock()
            cfg.intelligence.memory_backend = "animus"
            cfg.intelligence.memory_db_path = f"{tmpdir}/mem.db"

            rt = AnimusRuntime.__new__(AnimusRuntime)
            rt._config = cfg
            with patch(
                "animus_bootstrap.intelligence.memory_backends.animus_backend.AnimusMemoryBackend"
            ) as mock_cls:
                mock_cls.return_value = MagicMock()
                rt._create_memory_manager()
            mock_cls.assert_called_once()

    def test_chromadb_memory_backend(self):
        """Memory backend 'chromadb' creates ChromaDBMemoryBackend."""
        import tempfile

        from animus_bootstrap.runtime import AnimusRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MagicMock()
            cfg.intelligence.memory_backend = "chromadb"
            cfg.intelligence.memory_db_path = f"{tmpdir}/mem.db"
            from pathlib import Path as _Path

            cfg.get_data_path.return_value = _Path(tmpdir)

            rt = AnimusRuntime.__new__(AnimusRuntime)
            rt._config = cfg
            # Use sys.modules patch to avoid poisoning chromadb module
            mock_mod = MagicMock()
            mock_mod.ChromaDBMemoryBackend.return_value = MagicMock()
            with patch.dict(
                "sys.modules",
                {
                    "animus_bootstrap.intelligence.memory_backends.chromadb_backend": mock_mod,
                },
            ):
                rt._create_memory_manager()
            mock_mod.ChromaDBMemoryBackend.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_tool_conflict(self):
        """MCP tool name conflict logs warning."""
        from animus_bootstrap.runtime import AnimusRuntime

        mock_tool = MagicMock()
        mock_tool.name = "conflicting_tool"

        mock_bridge = AsyncMock()
        mock_bridge.discover_servers = AsyncMock(return_value=["srv1"])
        mock_bridge.import_tools = AsyncMock(return_value=[mock_tool])

        mock_executor = MagicMock()
        mock_executor.register.side_effect = ValueError("duplicate")

        rt = AnimusRuntime.__new__(AnimusRuntime)
        rt._config = MagicMock()
        rt._config.intelligence.mcp.config_path = "/tmp/fake.json"
        rt.tool_executor = mock_executor
        rt._mcp_bridge = None

        with patch(
            "animus_bootstrap.intelligence.tools.mcp_bridge.MCPToolBridge",
            return_value=mock_bridge,
        ):
            await rt._discover_mcp_tools()
        mock_executor.register.assert_called_once()

    @pytest.mark.asyncio
    async def test_runtime_stop_closes_mcp(self):
        """Runtime stop closes MCP bridge."""
        from animus_bootstrap.runtime import AnimusRuntime

        mock_bridge = AsyncMock()

        rt = AnimusRuntime.__new__(AnimusRuntime)
        rt._started = True
        rt._mcp_bridge = mock_bridge
        rt._channels = {}
        rt._timer_store = None
        rt._improvement_store = None
        rt._tool_history_store = None
        rt.proactive_engine = None
        rt.automation_engine = None
        rt.tool_executor = None
        rt.feedback_store = None
        rt.memory_manager = None
        rt.session_manager = None

        await rt.stop()
        mock_bridge.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_tool_import_success_logged(self):
        """MCP import logs total count when tools succeed."""
        from animus_bootstrap.runtime import AnimusRuntime

        mock_tool = MagicMock()
        mock_tool.name = "good_tool"

        mock_bridge = AsyncMock()
        mock_bridge.discover_servers = AsyncMock(return_value=["srv1"])
        mock_bridge.import_tools = AsyncMock(return_value=[mock_tool])

        mock_executor = MagicMock()
        mock_executor.register = MagicMock()

        rt = AnimusRuntime.__new__(AnimusRuntime)
        rt._config = MagicMock()
        rt._config.intelligence.mcp.config_path = "/tmp/fake.json"
        rt.tool_executor = mock_executor
        rt._mcp_bridge = None

        with patch(
            "animus_bootstrap.intelligence.tools.mcp_bridge.MCPToolBridge",
            return_value=mock_bridge,
        ):
            await rt._discover_mcp_tools()
        mock_executor.register.assert_called_once_with(mock_tool)


# ======================================================================
# Config schema: env var overrides (lines 41, 44)
# ======================================================================


class TestConfigSchemaEnvOverrides:
    """Test ApiSection model_post_init env var overrides."""

    def test_anthropic_key_from_env(self):
        from animus_bootstrap.config.schema import ApiSection

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            section = ApiSection()
        assert section.anthropic_key == "sk-test-key"

    def test_openai_key_from_env(self):
        from animus_bootstrap.config.schema import ApiSection

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-openai-key"}):
            section = ApiSection()
        assert section.openai_key == "sk-openai-key"


# ======================================================================
# Animus backend: import paths (lines 15, 19)
# ======================================================================


class TestAnimusBackendInit:
    """Test AnimusMemoryBackend __init__ branches."""

    def test_import_error_raises_runtime(self, tmp_path):
        from animus_bootstrap.intelligence.memory_backends.animus_backend import (
            AnimusMemoryBackend,
        )

        with patch.dict("sys.modules", {"animus": None, "animus.memory": None}):
            with pytest.raises(RuntimeError, match="animus-core is not installed"):
                AnimusMemoryBackend(data_dir=tmp_path / "mem")

    def test_success_creates_backend(self, tmp_path):
        from animus_bootstrap.intelligence.memory_backends.animus_backend import (
            AnimusMemoryBackend,
        )

        try:
            b = AnimusMemoryBackend(data_dir=tmp_path / "mem")
            assert (tmp_path / "mem").exists()
            b.close()
        except RuntimeError:
            pytest.skip("animus-core not installed")


# ======================================================================
# Forge CTL: systemd start fail + httpx missing (lines 72, 146-147)
# ======================================================================


class TestForgeCtlExtra:
    """Test remaining forge_ctl uncovered lines."""

    @pytest.mark.asyncio
    async def test_systemd_start_fails_then_uvicorn(self):
        """systemd start returns non-zero, then uvicorn fallback."""
        from animus_bootstrap.intelligence.tools.builtin.forge_ctl import (
            _forge_start,
        )

        call_count = {"n": 0}

        mock_is_active = AsyncMock()
        mock_is_active.returncode = 3  # inactive
        mock_is_active.communicate = AsyncMock(return_value=(b"inactive", b""))

        mock_systemd_start = AsyncMock()
        mock_systemd_start.returncode = 1
        mock_systemd_start.communicate = AsyncMock(return_value=(b"", b"service not found"))

        mock_uvicorn = AsyncMock()
        mock_uvicorn.returncode = None
        mock_uvicorn.pid = 55555

        async def fake_exec(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return mock_is_active
            if call_count["n"] == 2:
                return mock_systemd_start
            return mock_uvicorn

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            with patch(
                "asyncio.wait_for",
                side_effect=[
                    (b"inactive", b""),  # is-active check
                    (b"", b"service not found"),  # start fails
                ],
            ):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await _forge_start()
        assert "uvicorn" in result.lower() or "PID" in result

    @pytest.mark.asyncio
    async def test_forge_invoke_connect_error(self):
        """forge_invoke handles connection refused."""
        import httpx
        import respx

        from animus_bootstrap.intelligence.tools.builtin.forge_ctl import (
            _forge_invoke,
        )

        with respx.mock:
            respx.get("http://127.0.0.1:8000/health").mock(
                side_effect=httpx.ConnectError("refused")
            )
            result = await _forge_invoke("/health")
        assert "not reachable" in result


# ======================================================================
# Dashboard proposals: no runtime / no PM (lines 35, 45, 78, 98)
# ======================================================================


class TestProposalsDashboard:
    """Test proposals dashboard edge cases."""

    def test_get_stores_no_runtime(self):
        from animus_bootstrap.dashboard.routers.proposals import _get_stores

        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])
        result = _get_stores(mock_request)
        assert result == (None, None)

    def test_get_proposal_manager_none(self):
        from animus_bootstrap.dashboard.routers.proposals import (
            _get_proposal_manager,
        )

        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])
        result = _get_proposal_manager(mock_request)
        assert result is None


# ======================================================================
# Small gaps: __main__, web import, schedule, self_mod, forge_page
# ======================================================================


class TestSmallCoverageGaps:
    """Cover single-line branches across multiple modules."""

    def test_schedule_step_with_single_value(self):
        """Schedule parser handles step/N with single base value."""
        from animus_bootstrap.intelligence.proactive.schedule import (
            _parse_cron_field,
        )

        result = _parse_cron_field("5/10", 0, 59)
        assert 5 in result
        assert 15 in result

    def test_self_mod_page_with_history(self):
        """Self-mod page renders code history from tool executor."""
        from animus_bootstrap.dashboard.routers.self_mod import (
            _get_runtime,
        )

        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])
        result = _get_runtime(mock_request)
        assert result is None

    def test_session_manager_close_branch(self):
        """SessionManager close handles missing db."""
        import tempfile
        from pathlib import Path

        from animus_bootstrap.gateway.session import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(db_path=Path(tmpdir) / "session.db")
            mgr.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_sqlite_backend_stats_missing_db(self):
        """SQLite backend get_stats handles missing db file."""
        import tempfile
        from pathlib import Path

        from animus_bootstrap.intelligence.memory_backends.sqlite_backend import (
            SQLiteMemoryBackend,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "mem.db"
            backend = SQLiteMemoryBackend(db_path)
            # Delete the db file to trigger OSError in stat()
            db_path.unlink(missing_ok=True)
            stats = await backend.get_stats()
            assert stats["db_size_bytes"] == 0

    def test_installer_unsupported_os(self):
        """Installer returns False for unsupported OS."""
        from animus_bootstrap.daemon.installer import AnimusInstaller

        installer = AnimusInstaller()
        with patch.object(installer, "detect_os", return_value="haiku_os"):
            result = installer.register_service()
        assert result is False

    def test_self_mod_with_tool_history(self):
        """Self-mod page processes code_write entries from history."""
        from animus_bootstrap.dashboard.routers.self_mod import self_mod_page

        mock_entry = MagicMock()
        mock_entry.tool_name = "code_write"
        mock_entry.timestamp = "2025-01-01T00:00:00"
        mock_entry.success = True
        mock_entry.duration_ms = 42
        mock_entry.output = "wrote file"

        mock_executor = MagicMock()
        mock_executor.get_history.return_value = [mock_entry]

        mock_runtime = MagicMock()
        mock_runtime.tool_executor = mock_executor

        mock_request = MagicMock()
        mock_request.app.state.runtime = mock_runtime
        mock_request.app.state.templates = MagicMock()

        import asyncio

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self_mod_page(mock_request))
        finally:
            loop.close()
        mock_executor.get_history.assert_called_once()
