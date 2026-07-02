"""Tests for Animus Head core components.

Covers checkpoint persistence, tool orchestration, session bootstrap,
and REPL lifecycle without requiring a live Ollama instance.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from animus_kernel.head.checkpoint import HeadCheckpoint, HeadCheckpointStore
from animus_kernel.head.session_bootstrap import SessionBootstrap
from animus_kernel.head.tool_orchestrator import HeadToolOrchestrator
from animus_kernel.head.tool_validator import HeadToolValidator, RetryableToolExecutor


# ------------------------------------------------------------------
# Checkpoint tests
# ------------------------------------------------------------------

class TestHeadCheckpointStore:
    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_head.db"
            store = HeadCheckpointStore(db_path=db_path)

            cp = HeadCheckpoint(
                session_id="test-123",
                started_at=datetime.now(UTC),
                last_active_at=datetime.now(UTC),
                project_root="/fake/project",
                messages=[{"role": "user", "content": "hello"}],
                summary="test session",
                total_tokens=42,
                turns=1,
            )
            store.save(cp)

            loaded = store.load("test-123")
            assert loaded is not None
            assert loaded.session_id == "test-123"
            assert loaded.project_root == "/fake/project"
            assert loaded.messages == [{"role": "user", "content": "hello"}]
            assert loaded.total_tokens == 42
            assert loaded.turns == 1

    def test_list_recent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_head.db"
            store = HeadCheckpointStore(db_path=db_path)

            for i in range(3):
                cp = HeadCheckpoint(
                    session_id=f"sess-{i}",
                    started_at=datetime.now(UTC),
                    last_active_at=datetime.now(UTC),
                )
                store.save(cp)

            recent = store.list_recent(limit=2)
            assert len(recent) == 2
            # Most recent first (sess-2, then sess-1)
            assert recent[0].session_id == "sess-2"
            assert recent[1].session_id == "sess-1"

    def test_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_head.db"
            store = HeadCheckpointStore(db_path=db_path)

            cp = HeadCheckpoint(
                session_id="to-delete",
                started_at=datetime.now(UTC),
                last_active_at=datetime.now(UTC),
            )
            store.save(cp)
            assert store.load("to-delete") is not None

            assert store.delete("to-delete") is True
            assert store.load("to-delete") is None
            assert store.delete("to-delete") is False


# ------------------------------------------------------------------
# Tool orchestrator tests
# ------------------------------------------------------------------

class TestHeadToolOrchestrator:
    def test_list_tools_includes_filesystem_and_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HeadToolOrchestrator(
                project_root=tmpdir,
                memory_dir=Path(tmpdir) / "memory",
                enable_shell=True,
            )
            tools = orchestrator.list_tools()
            names = [t["function"]["name"] for t in tools]

            # Forge filesystem tools
            assert "read_file" in names
            assert "list_files" in names
            assert "search_code" in names
            assert "get_project_structure" in names
            assert "write_file" in names
            assert "edit_file" in names

            # Head custom tools
            assert "run_shell" in names
            assert "remember" in names
            assert "recall" in names
            assert "list_tasks" in names
            assert "create_task" in names

    def test_run_shell_allowed_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HeadToolOrchestrator(
                project_root=tmpdir,
                memory_dir=Path(tmpdir) / "memory",
                enable_shell=True,
            )
            result = orchestrator.execute("run_shell", {"command": "pwd"})
            assert "STDOUT:" in result
            assert "EXIT CODE: 0" in result

    def test_run_shell_blocked_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HeadToolOrchestrator(
                project_root=tmpdir,
                memory_dir=Path(tmpdir) / "memory",
                enable_shell=True,
            )
            result = orchestrator.execute("run_shell", {"command": "rm -rf /"})
            assert "not in allowed list" in result

    def test_remember_and_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HeadToolOrchestrator(
                project_root=tmpdir,
                memory_dir=Path(tmpdir) / "memory",
            )
            store_result = orchestrator.execute(
                "remember",
                {"content": "SQLite WAL deadlock under async load", "tags": ["sqlite", "bug"]}
            )
            assert "Stored memory" in store_result

            # LocalMemoryStore.search uses substring matching on the full query
            recall_result = orchestrator.execute("recall", {"query": "WAL deadlock"})
            assert "Found" in recall_result
            assert "WAL deadlock" in recall_result

    def test_read_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "hello.txt"
            test_file.write_text("Hello world")

            orchestrator = HeadToolOrchestrator(
                project_root=tmpdir,
                memory_dir=Path(tmpdir) / "memory",
            )
            result = orchestrator.execute("read_file", {"path": "hello.txt"})
            assert "Hello world" in result

    def test_unknown_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HeadToolOrchestrator(
                project_root=tmpdir,
                memory_dir=Path(tmpdir) / "memory",
            )
            result = orchestrator.execute("nonexistent_tool", {})
            assert "Unknown tool" in result


# ------------------------------------------------------------------
# Session bootstrap tests
# ------------------------------------------------------------------

class TestSessionBootstrap:
    def test_bootstrap_collects_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "my_project"
            project.mkdir()

            # Create a TODO.md
            todo = project / "TODO.md"
            todo.write_text("- [ ] Fix auth bug\n- [x] Deploy infra\n- [ ] Write tests\n")

            bootstrap = SessionBootstrap(project_root=project)
            ctx = bootstrap.bootstrap()

            assert ctx["project_name"] == "my_project"
            assert ctx["project_root"] == str(project)
            assert len(ctx["active_tasks"]) == 2
            assert "Fix auth bug" in ctx["active_tasks"][0]

    def test_build_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "my_project"
            project.mkdir()

            # Create TODO so active tasks section appears
            todo = project / "TODO.md"
            todo.write_text("- [ ] Fix auth\n")

            bootstrap = SessionBootstrap(project_root=project)
            ctx = bootstrap.bootstrap()
            prompt = bootstrap.build_system_prompt(ctx)

            assert "Animus Head" in prompt
            assert "my_project" in prompt
            assert "ACTIVE TASKS" in prompt

    def test_previous_session_not_loaded_when_old(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sessions.db"
            store = HeadCheckpointStore(db_path=db_path)

            # Save a very old checkpoint
            old = HeadCheckpoint(
                session_id="old-session",
                started_at=datetime(2020, 1, 1, tzinfo=UTC),
                last_active_at=datetime(2020, 1, 1, tzinfo=UTC),
            )
            store.save(old)

            bootstrap = SessionBootstrap(
                project_root=tmpdir,
                checkpoint_store=store,
            )
            ctx = bootstrap.bootstrap()
            assert ctx["previous_session"] is None


# ------------------------------------------------------------------
# Integration: full flow (mock provider)
# ------------------------------------------------------------------

class TestHeadREPLLifecycle:
    def test_repl_initializes_with_context(self) -> None:
        """Verify REPL can be created and bootstraps context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "test_project"
            project.mkdir()

            # Skip if Ollama is not available — this test validates structure only
            try:
                from animus_kernel.head.repl import HeadREPL

                repl = HeadREPL(
                    model="llama3.1:8b",
                    project_root=project,
                    memory_dir=Path(tmpdir) / "memory",
                )
                # If we get here, provider init succeeded (Ollama is running)
                repl.bootstrap()
                assert repl.project_root == project
                assert repl.model == "llama3.1:8b"
                assert len(repl.messages) >= 1
                assert repl.messages[0]["role"] == "system"
            except RuntimeError as exc:
                if "Ollama is not running" in str(exc):
                    pytest.skip("Ollama not available")
                raise


# ------------------------------------------------------------------
# Phase 2: Tool validation and retry
# ------------------------------------------------------------------

class TestHeadToolValidator:
    def test_validate_known_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HeadToolOrchestrator(
                project_root=tmpdir,
                memory_dir=Path(tmpdir) / "memory",
            )
            validator = HeadToolValidator(registry=orchestrator._forge)
            result = validator.validate("read_file", {"path": "test.py"})
            assert result.valid is True
            assert result.error == ""

    def test_validate_unknown_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HeadToolOrchestrator(
                project_root=tmpdir,
                memory_dir=Path(tmpdir) / "memory",
            )
            validator = HeadToolValidator(registry=orchestrator._forge)
            result = validator.validate("nonexistent_tool", {})
            assert result.valid is False
            assert "Unknown tool" in result.error

    def test_validate_missing_required_arg(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HeadToolOrchestrator(
                project_root=tmpdir,
                memory_dir=Path(tmpdir) / "memory",
            )
            validator = HeadToolValidator(registry=orchestrator._forge)
            result = validator.validate("edit_file", {"path": "test.py"})
            assert result.valid is False
            assert "old_string" in result.error

    def test_extract_json_tool_call(self) -> None:
        validator = HeadToolValidator()
        results = validator.extract_tool_calls(
            '{"name": "read_file", "arguments": {"path": "main.py"}}'
        )
        assert len(results) == 1
        assert results[0].tool_name == "read_file"
        assert results[0].arguments == {"path": "main.py"}

    def test_extract_hermes_xml_tool_call(self) -> None:
        validator = HeadToolValidator()
        results = validator.extract_tool_calls(
            '<tool_call><name>read_file</name>'
            '<arguments>{"path": "main.py"}</arguments></tool_call>'
        )
        assert len(results) == 1
        assert results[0].tool_name == "read_file"
        assert results[0].arguments == {"path": "main.py"}

    def test_build_retry_prompt(self) -> None:
        validator = HeadToolValidator()
        invalid = [
            validator.validate("nonexistent", {}),
        ]
        prompt = validator.build_retry_prompt(invalid)
        assert "invalid" in prompt.lower()
        assert "nonexistent" in prompt


class TestRetryableToolExecutor:
    def test_execute_with_valid_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HeadToolOrchestrator(
                project_root=tmpdir,
                memory_dir=Path(tmpdir) / "memory",
            )
            executor = RetryableToolExecutor(
                orchestrator=orchestrator,
                registry=orchestrator._forge,
                max_retries=2,
            )

            # Mock model callback that returns a valid tool call
            call_count = 0
            def mock_callback():
                nonlocal call_count
                call_count += 1
                from animus_kernel.providers.base import CompletionResponse, ToolCall
                return CompletionResponse(
                    content="",
                    model="test",
                    provider="test",
                    tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "test.py"})],
                )

            # Since read_file doesn't exist as a file, it will error but the validation passes
            result = executor.execute_with_retry(
                "read_file",
                {"path": "test.py"},
                messages=[],
                model_callback=mock_callback,
            )
            # Result is from tool execution, not validation failure
            assert "File:" in result or "ERROR" in result
            assert call_count == 0  # No retry needed for valid tool

    def test_execute_with_invalid_tool_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HeadToolOrchestrator(
                project_root=tmpdir,
                memory_dir=Path(tmpdir) / "memory",
            )
            executor = RetryableToolExecutor(
                orchestrator=orchestrator,
                registry=orchestrator._forge,
                max_retries=2,
            )

            call_count = 0
            def mock_callback():
                nonlocal call_count
                call_count += 1
                from animus_kernel.providers.base import CompletionResponse
                # Return no tool_calls (simulates model not responding with fix)
                return CompletionResponse(
                    content="I don't know",
                    model="test",
                    provider="test",
                )

            result = executor.execute_with_retry(
                "nonexistent_tool",
                {},
                messages=[],
                model_callback=mock_callback,
            )
            # After validation fails and model didn't provide fix, early return
            assert "Model did not provide" in result
            assert call_count == 1  # Callback called once, then early return


# ------------------------------------------------------------------
# Phase 2: MCP tool discovery (graceful degradation)
# ------------------------------------------------------------------

class TestMCPToolDiscovery:
    def test_mcp_discovery_gracefully_degrades(self) -> None:
        """MCP discovery should not crash when MCP infra is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HeadToolOrchestrator(
                project_root=tmpdir,
                memory_dir=Path(tmpdir) / "memory",
                enable_mcp=True,  # Request MCP but infra missing
            )
            # Should still have core tools
            tools = orchestrator.list_tools()
            names = [t["function"]["name"] for t in tools]
            assert "read_file" in names
            assert "remember" in names
            # MCP tools won't be present since client module doesn't exist
            mcp_tools = [n for n in names if n.startswith("mcp_")]
            assert len(mcp_tools) == 0


# ------------------------------------------------------------------
# Phase 2: Autonomy benchmark
# ------------------------------------------------------------------

class TestAutonomyBenchmark:
    def test_benchmark_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HeadToolOrchestrator(
                project_root=tmpdir,
                memory_dir=Path(tmpdir) / "memory",
            )
            from animus_kernel.head.benchmarks.autonomy_suite import AutonomyBenchmark

            benchmark = AutonomyBenchmark(orchestrator)
            report = benchmark.run()

            assert report["total"] == 20
            assert report["accuracy"] >= 80.0  # All expected tools should be available
            assert report["passed"] >= 16

    def test_benchmark_report_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = HeadToolOrchestrator(
                project_root=tmpdir,
                memory_dir=Path(tmpdir) / "memory",
            )
            from animus_kernel.head.benchmarks.autonomy_suite import AutonomyBenchmark

            benchmark = AutonomyBenchmark(orchestrator)
            report = benchmark.run()

            assert "total" in report
            assert "passed" in report
            assert "failed" in report
            assert "accuracy" in report
            assert "results" in report
            assert len(report["results"]) == 20
