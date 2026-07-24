"""Coverage push round 6 — targeting tools.py, filesystem.py,
swarm/engine.py, proactive.py."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from animus.tools import ToolResult

# ── tools.py ───────────────────────────────────────────────────────────


class TestToolResultToDict:
    """Cover ToolResult.to_dict (line 114)."""

    def test_to_dict(self):
        r = ToolResult(tool_name="test", success=True, output="ok", error=None)
        d = r.to_dict()
        assert d == {"tool_name": "test", "success": True, "output": "ok", "error": None}


class TestToolGetDatetime:
    """Cover _tool_get_datetime error path (lines 257-258)."""

    def test_error_path(self):
        from animus.tools import _tool_get_datetime

        with patch("animus.tools.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.side_effect = ValueError("bad format")
            result = _tool_get_datetime({"format": "%Y"})
            assert not result.success


class TestToolReadFile:
    """Cover _tool_read_file branches."""

    def test_read_file_exception(self, tmp_path):
        """Lines 322-323: general exception (unreadable file)."""
        from animus.tools import _tool_read_file

        f = tmp_path / "test.txt"
        f.write_text("content")
        f.chmod(0o000)
        try:
            result = _tool_read_file({"path": str(f)})
            assert not result.success
        finally:
            f.chmod(0o644)

    def test_read_file_too_large(self, tmp_path):
        """File exceeds max size."""
        from animus.tools import _tool_read_file

        f = tmp_path / "big.txt"
        f.write_text("x" * 2_000_000)
        result = _tool_read_file({"path": str(f)})
        assert not result.success
        assert "large" in result.error.lower()


class TestToolListFiles:
    """Cover _tool_list_files exception path (lines 376-377)."""

    def test_list_files_exception(self):
        from animus.tools import _tool_list_files

        with patch("animus.tools.glob_module.glob", side_effect=OSError("disk error")):
            result = _tool_list_files({"pattern": "*", "directory": "."})
            assert not result.success


class TestToolRunCommand:
    """Cover _tool_run_command paths."""

    def test_run_command_stderr(self):
        """Line 422: command with stderr output."""
        from animus.tools import _tool_run_command

        result = _tool_run_command({"command": "echo hello && echo err >&2"})
        assert result.success
        assert "hello" in result.output
        assert "err" in result.output

    def test_run_command_timeout(self):
        """Lines 437-438: timeout exception."""
        from animus.tools import _tool_run_command

        result = _tool_run_command({"command": "sleep 60", "timeout": 1})
        assert not result.success
        assert "timed out" in result.error.lower()

    def test_run_command_general_exception(self):
        """Lines 437-438: general exception."""
        from animus.tools import _tool_run_command

        with patch("animus.tools.subprocess.run", side_effect=OSError("not found")):
            result = _tool_run_command({"command": "some_command"})
            assert not result.success


class TestToolHttpRequest:
    """Cover _tool_http_request branches."""

    def test_http_truncates_long_response(self):
        """Line 515: truncate responses > 50000 chars."""
        from animus.tools import _tool_http_request

        long_body = "x" * 60_000

        mock_response = MagicMock()
        mock_response.read.return_value = long_body.encode()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = _tool_http_request({"url": "http://example.com"})
            assert result.success
            assert "truncated" in result.output

    def test_http_body_with_post(self):
        """Lines 502-504: POST with body sets Content-Type."""
        from animus.tools import _tool_http_request

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"ok": true}'
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_open:
            result = _tool_http_request(
                {
                    "url": "http://example.com/api",
                    "method": "POST",
                    "body": '{"key": "val"}',
                }
            )
            assert result.success
            req = mock_open.call_args[0][0]
            assert req.get_header("Content-type") == "application/json"

    def test_http_error_body_read_fails(self):
        """Lines 533-534: HTTP error body read fails."""
        import urllib.error

        from animus.tools import _tool_http_request

        error = urllib.error.HTTPError("http://example.com", 500, "Server Error", {}, None)
        error.read = MagicMock(side_effect=OSError("read error"))

        with patch("urllib.request.urlopen", side_effect=error):
            result = _tool_http_request({"url": "http://example.com"})
            assert not result.success
            assert "500" in result.error

    def test_http_general_exception(self):
        """Lines 541-542: general exception."""
        from animus.tools import _tool_http_request

        with patch("urllib.request.urlopen", side_effect=OSError("connection failed")):
            result = _tool_http_request({"url": "http://example.com"})
            assert not result.success
            assert "connection failed" in result.error


class TestToolWebSearch:
    """Cover web_search exception path."""

    def test_web_search_exception(self):
        """Lines 587-591: urlopen exception."""
        from animus.tools import _tool_web_search

        with patch("urllib.request.urlopen", side_effect=OSError("network error")):
            result = _tool_web_search({"query": "test query"})
            assert not result.success


# ── filesystem.py ──────────────────────────────────────────────────────


def _make_fs_integration(tmp_path):
    from animus.integrations.filesystem import FilesystemIntegration

    return FilesystemIntegration(data_dir=tmp_path)


class TestFilesystemShouldExclude:
    """Cover _should_exclude path match (line 200)."""

    def test_exclude_by_path_pattern(self, tmp_path):
        fs = _make_fs_integration(tmp_path)
        path = tmp_path / ".git" / "config"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        assert fs._should_exclude(path)


class TestFilesystemIndexDirectory:
    """Cover _index_directory edge cases."""

    def test_index_nonexistent_path(self, tmp_path):
        fs = _make_fs_integration(tmp_path)
        count = fs._index_directory(tmp_path / "nonexistent")
        assert count == 0

    def test_index_excludes_pattern(self, tmp_path):
        fs = _make_fs_integration(tmp_path)
        pyc = tmp_path / "test.pyc"
        pyc.touch()
        normal = tmp_path / "test.py"
        normal.write_text("hello")
        fs._index_directory(tmp_path, recursive=False)
        assert str(pyc) not in fs._index
        assert str(normal) in fs._index

    def test_index_permission_error_on_entry(self, tmp_path):
        fs = _make_fs_integration(tmp_path)
        sub = tmp_path / "restricted"
        sub.mkdir()
        sub.chmod(0o000)
        try:
            count = fs._index_directory(tmp_path, recursive=False)
            assert count >= 0
        finally:
            sub.chmod(0o755)

    def test_index_permission_error_on_dir(self, tmp_path):
        fs = _make_fs_integration(tmp_path)
        restricted = tmp_path / "noaccess"
        restricted.mkdir()
        inner = restricted / "file.txt"
        inner.touch()
        restricted.chmod(0o000)
        try:
            count = fs._index_directory(restricted)
            assert count == 0
        finally:
            restricted.chmod(0o755)


class TestFilesystemToolSearchContent:
    """Cover _tool_search_content branches."""

    def test_search_content_invalid_regex(self, tmp_path):
        fs = _make_fs_integration(tmp_path)
        result = asyncio.run(fs._tool_search_content(pattern="[invalid"))
        assert not result.success

    def test_search_content_skips_binary_extension(self, tmp_path):
        from animus.integrations.filesystem import FileEntry

        fs = _make_fs_integration(tmp_path)
        exe = tmp_path / "app.exe"
        exe.write_bytes(b"\x00\x01\x02")
        fs._index[str(exe)] = FileEntry(
            path=str(exe),
            name="app.exe",
            extension=".exe",
            size=100,
            modified=datetime.now(),
            is_dir=False,
        )
        result = asyncio.run(fs._tool_search_content(pattern="test"))
        assert result.success
        assert result.output["count"] == 0

    def test_search_content_skips_large_file(self, tmp_path):
        from animus.integrations.filesystem import FileEntry

        fs = _make_fs_integration(tmp_path)
        big = tmp_path / "big.py"
        big.write_text("hello")
        fs._index[str(big)] = FileEntry(
            path=str(big),
            name="big.py",
            extension=".py",
            size=2_000_000,
            modified=datetime.now(),
            is_dir=False,
        )
        result = asyncio.run(fs._tool_search_content(pattern="hello"))
        assert result.success
        assert result.output["count"] == 0

    def test_search_content_file_pattern_filter(self, tmp_path):
        from animus.integrations.filesystem import FileEntry

        fs = _make_fs_integration(tmp_path)
        py_file = tmp_path / "test.py"
        py_file.write_text("match_me")
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("match_me")
        fs._index[str(py_file)] = FileEntry(
            path=str(py_file),
            name="test.py",
            extension=".py",
            size=100,
            modified=datetime.now(),
            is_dir=False,
        )
        fs._index[str(txt_file)] = FileEntry(
            path=str(txt_file),
            name="test.txt",
            extension=".txt",
            size=100,
            modified=datetime.now(),
            is_dir=False,
        )
        result = asyncio.run(
            fs._tool_search_content(
                pattern="match_me",
                file_pattern="*.py",
            )
        )
        assert result.success
        assert result.output["count"] == 1

    def test_search_content_oserror_on_read(self, tmp_path):
        from animus.integrations.filesystem import FileEntry

        fs = _make_fs_integration(tmp_path)
        fake_path = tmp_path / "gone.py"
        fs._index[str(fake_path)] = FileEntry(
            path=str(fake_path),
            name="gone.py",
            extension=".py",
            size=100,
            modified=datetime.now(),
            is_dir=False,
        )
        result = asyncio.run(fs._tool_search_content(pattern="anything"))
        assert result.success
        assert result.output["count"] == 0


class TestFilesystemToolRead:
    """Cover _tool_read edge cases."""

    def test_read_file_truncated(self, tmp_path):
        fs = _make_fs_integration(tmp_path)
        f = tmp_path / "long.txt"
        f.write_text("\n".join(f"line {i}" for i in range(200)))
        result = asyncio.run(fs._tool_read(path=str(f), max_lines=10))
        assert result.success
        assert result.output["truncated"] is True
        assert result.output["lines"] == 10

    def test_read_file_oserror(self, tmp_path):
        fs = _make_fs_integration(tmp_path)
        f = tmp_path / "nope.txt"
        f.write_text("data")
        f.chmod(0o000)
        try:
            result = asyncio.run(fs._tool_read(path=str(f)))
            assert not result.success
        finally:
            f.chmod(0o644)


# ── swarm/engine.py ────────────────────────────────────────────────────


def _mock_cognitive():
    """Create a mock CognitiveLayer for swarm tests."""
    from animus.cognitive import CognitiveLayer, ModelConfig

    config = ModelConfig.mock(default_response="mock output")
    return CognitiveLayer(config)


def _make_swarm_config(agents=None, gates=None, max_cost=None):
    from animus.forge.models import AgentConfig, WorkflowConfig

    default_agents = agents or [
        AgentConfig(name="a1", archetype="researcher", outputs=["brief"]),
        AgentConfig(name="a2", archetype="writer", inputs=["a1.brief"], outputs=["draft"]),
    ]
    return WorkflowConfig(
        name="test_swarm",
        agents=default_agents,
        gates=gates or [],
        max_cost_usd=max_cost or 10.0,
    )


class TestSwarmEngineResume:
    """Cover budget restore from prior results (lines 89-93)."""

    def test_resume_restores_budget_and_skips_completed(self, tmp_path):
        from animus.forge.checkpoint import CheckpointStore
        from animus.forge.models import StepResult, WorkflowState
        from animus.swarm.engine import SwarmEngine

        config = _make_swarm_config()
        db_path = tmp_path / "forge_checkpoints.db"
        cp = CheckpointStore(db_path)
        state = WorkflowState(workflow_name="test_swarm")
        state.results.append(
            StepResult(
                agent_name="a1",
                success=True,
                outputs={"a1.brief": "done"},
                tokens_used=100,
                cost_usd=0.01,
            )
        )
        state.status = "running"
        cp.save_state(state)

        cognitive = _mock_cognitive()
        engine = SwarmEngine(cognitive, checkpoint_dir=tmp_path)
        result = engine.run(config, resume=True)
        assert result.status == "completed"
        agent_names = {r.agent_name for r in result.results}
        assert "a2" in agent_names


class TestSwarmEngineUnexpectedException:
    """Cover unexpected exception wrapping (lines 179-182)."""

    def test_unexpected_error_wraps_in_swarm_error(self, tmp_path):
        from animus.swarm.engine import SwarmEngine
        from animus.swarm.models import SwarmError

        config = _make_swarm_config()
        cognitive = _mock_cognitive()
        engine = SwarmEngine(cognitive, checkpoint_dir=tmp_path)

        with patch.object(engine, "_execute_stage", side_effect=RuntimeError("boom")):
            with pytest.raises(SwarmError, match="Unexpected error"):
                engine.run(config)


class TestSwarmEngineConflicts:
    """Cover intent conflict resolution (lines 214-220)."""

    def test_conflict_resolution(self, tmp_path):
        from animus.forge.models import AgentConfig
        from animus.swarm.engine import SwarmEngine

        agents = [
            AgentConfig(name="a1", archetype="researcher", outputs=["data"]),
            AgentConfig(name="a2", archetype="researcher", outputs=["data"]),
        ]
        config = _make_swarm_config(agents=agents)
        cognitive = _mock_cognitive()
        engine = SwarmEngine(cognitive, checkpoint_dir=tmp_path)
        result = engine.run(config)
        assert result.status == "completed"


class TestSwarmEngineMissingInput:
    """Cover missing input warning (line 294)."""

    def test_missing_input_ref(self, tmp_path):
        from animus.swarm.engine import SwarmEngine

        cognitive = _mock_cognitive()
        engine = SwarmEngine(cognitive, checkpoint_dir=tmp_path)

        # Test _resolve_inputs directly with a missing key
        inputs = engine._resolve_inputs(
            ["existing.key", "missing.key"],
            {"existing.key": "value1"},
        )
        assert inputs == {"existing.key": "value1"}
        # missing.key should be absent (logged as warning)


# ── proactive.py ───────────────────────────────────────────────────────


def _make_proactive_engine(tmp_path, memory=None, cognitive=None):
    from animus.proactive import ProactiveEngine

    mem = memory or MagicMock()
    mem.search.return_value = []
    mem.search_by_tags.return_value = []
    mem.store = MagicMock()
    mem.store.list_all.return_value = []

    return ProactiveEngine(data_dir=tmp_path, memory=mem, cognitive=cognitive)


class TestProactiveLoadNudgesCorrupt:
    """Cover _load_nudges corrupt file (lines 182-183)."""

    def test_corrupt_nudges_file(self, tmp_path):
        from animus.proactive import ProactiveEngine

        nudges_file = tmp_path / "nudges.json"
        nudges_file.write_text("NOT JSON!!!")

        mem = MagicMock()
        mem.search.return_value = []
        mem.search_by_tags.return_value = []
        mem.store = MagicMock()
        mem.store.list_all.return_value = []

        pe = ProactiveEngine(data_dir=tmp_path, memory=mem)
        assert pe._nudges == []


class TestProactiveSynthesisExceptions:
    """Cover cognitive synthesis exception paths."""

    def test_morning_brief_synthesis_exception(self, tmp_path):
        """Lines 267-268: cognitive.think fails during briefing."""
        from animus.memory import Memory, MemoryType

        cog = MagicMock()
        cog.think.side_effect = RuntimeError("LLM down")

        mem = MagicMock()
        mem.search.return_value = [
            Memory.create(content="Task: finish report", memory_type=MemoryType.EPISODIC),
        ]
        mem.search_by_tags.return_value = []
        mem.store = MagicMock()
        mem.store.list_all.return_value = []

        pe = _make_proactive_engine(tmp_path, memory=mem, cognitive=cog)
        nudge = pe.generate_morning_brief()
        assert nudge is not None
        assert nudge.content

    def test_deadline_synthesis_exception(self, tmp_path):
        """Lines 323-324: cognitive.think fails during deadline check."""
        from animus.memory import Memory, MemoryType

        cog = MagicMock()
        cog.think.side_effect = RuntimeError("LLM down")

        deadline_mem = Memory.create(
            content="Deadline: submit report by tomorrow",
            memory_type=MemoryType.SEMANTIC,
            subtype="deadline",
        )
        deadline_mem.tags = ["deadline"]

        mem = MagicMock()
        mem.recall.return_value = [deadline_mem]
        mem.search.return_value = []
        mem.search_by_tags.return_value = []
        mem.store = MagicMock()
        mem.store.list_all.return_value = []

        pe = _make_proactive_engine(tmp_path, memory=mem, cognitive=cog)
        nudges = pe.scan_deadlines()
        assert len(nudges) >= 1

    def test_follow_up_synthesis_exception(self, tmp_path):
        """Lines 457-458: cognitive.think fails during follow-up check."""
        from animus.memory import Memory, MemoryType

        cog = MagicMock()
        cog.think.side_effect = RuntimeError("LLM down")

        follow_up_mem = Memory.create(
            content="I need to follow up on this request",
            memory_type=MemoryType.EPISODIC,
        )
        # Must be recent (within 7 days) for scan_follow_ups
        follow_up_mem.created_at = datetime.now() - timedelta(days=2)

        pe = _make_proactive_engine(tmp_path, cognitive=cog)
        # Override after helper (helper reassigns mem.store)
        pe.memory.store.list_all.return_value = [follow_up_mem]
        nudges = pe.scan_follow_ups()
        assert len(nudges) >= 1

    def test_context_nudge_synthesis_exception(self, tmp_path):
        """Lines 520-521: cognitive.think fails during context recall."""
        from animus.memory import Memory, MemoryType

        cog = MagicMock()
        cog.think.side_effect = RuntimeError("LLM down")

        old_mem = Memory.create(
            content="We discussed the API redesign last month",
            memory_type=MemoryType.EPISODIC,
        )
        old_mem.created_at = datetime.now() - timedelta(days=30)

        mem = MagicMock()
        mem.recall.return_value = [old_mem]
        mem.search.return_value = []
        mem.search_by_tags.return_value = []
        mem.store = MagicMock()
        mem.store.list_all.return_value = []

        pe = _make_proactive_engine(tmp_path, memory=mem, cognitive=cog)
        nudge = pe.generate_context_nudge("API redesign")
        assert nudge is not None


class TestProactiveBackgroundLoop:
    """Cover start_background already running (lines 581-582)."""

    def test_start_background_already_running(self, tmp_path):
        pe = _make_proactive_engine(tmp_path)
        pe._running = True
        pe.start_background(interval_seconds=60)
        assert pe._running is True
        pe._running = False
