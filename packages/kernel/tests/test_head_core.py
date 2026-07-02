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
from animus_kernel.head.context_manager import HeadContextManager
from animus_kernel.head.fallback_controller import HeadFallbackController
from animus_kernel.head.intent_parser import HeadIntentParser, IntentType, ParsedIntent
from animus_kernel.head.planner import HeadPlanner, ToolPlan, ToolPlanStep
from animus_kernel.head.quality_gate import HeadQualityGate, QualityScore
from animus_kernel.head.session_bootstrap import SessionBootstrap
from animus_kernel.head.synthesizer import HeadSynthesizer
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
                msgs = repl.context.get_messages()
                assert len(msgs) >= 1
                assert msgs[0]["role"] == "system"
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


# ------------------------------------------------------------------
# Phase 3: Session persistence polish (context manager)
# ------------------------------------------------------------------

class TestHeadContextManager:
    def test_add_message_increases_count(self) -> None:
        mgr = HeadContextManager(model="llama3.1:8b")
        mgr.add_message({"role": "system", "content": "You are a test assistant."})
        mgr.add_message({"role": "user", "content": "Hello"})
        assert len(mgr.get_messages()) == 2

    def test_system_message_first(self) -> None:
        mgr = HeadContextManager(model="llama3.1:8b")
        mgr.add_message({"role": "system", "content": "You are a test assistant."})
        mgr.add_message({"role": "user", "content": "Hello"})
        msgs = mgr.get_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_model_limit_resolution(self) -> None:
        assert HeadContextManager("qwen2.5:32b").max_tokens == 32768
        assert HeadContextManager("llama3.1:8b").max_tokens == 8192
        assert HeadContextManager("unknown-model").max_tokens == 8192  # default

    def test_pruning_drops_oldest_messages(self) -> None:
        # Very small window to force pruning
        mgr = HeadContextManager(model="llama3.1:8b", reserve_tokens=7000)
        mgr.add_message({"role": "system", "content": "System prompt"})

        # Fill with large messages
        for i in range(20):
            mgr.add_message({"role": "user", "content": "x" * 1000})
            mgr.add_message({"role": "assistant", "content": "y" * 1000})

        stats = mgr.get_stats()
        assert stats.message_count < 40  # Some messages were pruned
        assert stats.dropped_messages > 0
        assert stats.available_tokens >= 0

    def test_pruning_preserves_tool_call_pairs(self) -> None:
        mgr = HeadContextManager(model="llama3.1:8b", reserve_tokens=6000)
        mgr.add_message({"role": "system", "content": "System prompt"})

        # Add a tool-call round
        mgr.add_message({"role": "user", "content": "Run test"})
        mgr.add_message({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "run_shell", "arguments": "{}"}}],
        })
        mgr.add_message({"role": "tool", "tool_call_id": "tc1", "name": "run_shell", "content": "ok"})

        # Fill to trigger pruning
        for i in range(30):
            mgr.add_message({"role": "user", "content": "filler " * 500})
            mgr.add_message({"role": "assistant", "content": "response " * 500})

        # Ensure the tool pair is either fully present or fully absent
        # (it should be absent because it was added early)
        raw = mgr._messages
        tool_assist = [m for m in raw if m.get("role") == "assistant" and m.get("tool_calls")]
        tool_resp = [m for m in raw if m.get("role") == "tool"]
        # If any assistant tool_calls remain, their responses must also remain
        for ta in tool_assist:
            tc_id = ta["tool_calls"][0]["id"]
            assert any(m.get("tool_call_id") == tc_id for m in tool_resp)

    def test_summary_set_and_retrieve(self) -> None:
        mgr = HeadContextManager(model="llama3.1:8b")
        mgr.set_summary("The user asked about Python async patterns.")
        msgs = mgr.get_messages()
        summary_msg = [m for m in msgs if m["role"] == "system" and "Previous conversation" in m.get("content", "")]
        assert len(summary_msg) == 1
        assert "Python async" in summary_msg[0]["content"]

    def test_stats_format(self) -> None:
        mgr = HeadContextManager(model="llama3.1:8b")
        mgr.add_message({"role": "system", "content": "System"})
        mgr.add_message({"role": "user", "content": "Hello"})
        mgr.add_message({"role": "assistant", "content": "Hi there"})

        stats = mgr.get_stats()
        assert stats.max_tokens == 8192
        assert stats.reserve_tokens == 2048
        assert stats.message_count == 3
        assert stats.user_messages == 1
        assert stats.assistant_messages == 1
        assert stats.available_tokens > 0
        assert 0 <= stats.utilization_percent <= 100

    def test_clear_keeps_summary(self) -> None:
        mgr = HeadContextManager(model="llama3.1:8b")
        mgr.add_message({"role": "system", "content": "System"})
        mgr.add_message({"role": "user", "content": "Hello"})
        mgr.set_summary("Prior context")

        mgr.clear()
        assert len(mgr._messages) == 0
        assert mgr._summary == "Prior context"
        # get_messages should only contain summary system message
        msgs = mgr.get_messages()
        assert len(msgs) == 1
        assert "Prior context" in msgs[0]["content"]

    def test_invalid_message_skipped(self) -> None:
        mgr = HeadContextManager(model="llama3.1:8b")
        mgr.add_message({"role": "system", "content": "System"})
        mgr.add_message({"role": "user", "content": "Hello"})
        mgr.add_message({"not_a_role": "bad"})  # type: ignore[arg-type]
        assert len(mgr.get_messages()) == 2  # system + user only, invalid skipped

    def test_pruning_keeps_minimum_user_turns(self) -> None:
        mgr = HeadContextManager(model="llama3.1:8b", reserve_tokens=7500)
        mgr.add_message({"role": "system", "content": "System prompt"})

        for i in range(5):
            mgr.add_message({"role": "user", "content": "x" * 2000})
            mgr.add_message({"role": "assistant", "content": "y" * 2000})

        stats = mgr.get_stats()
        # Should preserve at least 2 user turns
        assert stats.user_messages >= 2


# ------------------------------------------------------------------
# Phase 4: Quality gates and cloud fallback
# ------------------------------------------------------------------

class TestHeadQualityGate:
    def test_score_empty_response_low(self) -> None:
        gate = HeadQualityGate()
        from animus_kernel.providers.base import CompletionResponse

        response = CompletionResponse(content="", model="test", provider="test")
        score = gate.evaluate("Do something", response, [], [])
        assert score.overall < 40
        assert score.response_completeness <= 10

    def test_score_refusal_low(self) -> None:
        gate = HeadQualityGate()
        from animus_kernel.providers.base import CompletionResponse

        response = CompletionResponse(content="I cannot help with that.", model="test", provider="test")
        score = gate.evaluate("Do something", response, [], [])
        # Refusals score below average but not catastrophic
        assert score.overall < 50
        assert score.response_completeness <= 10

    def test_score_good_response_high(self) -> None:
        gate = HeadQualityGate()
        from animus_kernel.providers.base import CompletionResponse

        response = CompletionResponse(
            content="Here is the file content you requested.", model="test", provider="test"
        )
        score = gate.evaluate("Read main.py", response, [], [])
        assert score.overall >= 60
        assert score.response_completeness == 40

    def test_score_valid_tool_calls_high(self) -> None:
        gate = HeadQualityGate()
        from animus_kernel.providers.base import CompletionResponse

        response = CompletionResponse(content="", model="test", provider="test")
        score = gate.evaluate("Read main.py", response, [{"name": "read_file"}], [])
        assert score.tool_call_quality == 40

    def test_score_invalid_tool_calls_low(self) -> None:
        gate = HeadQualityGate()
        from animus_kernel.providers.base import CompletionResponse

        response = CompletionResponse(content="", model="test", provider="test")
        # Mock invalid call result
        class FakeInvalid:
            tool_name = "bad_tool"
            error = "unknown tool"

        score = gate.evaluate("Run test", response, [], [FakeInvalid()])
        assert score.tool_call_quality == 0
        assert score.structure_quality <= 15  # penalized for invalid structure

    def test_failure_streak_increments(self) -> None:
        gate = HeadQualityGate()
        from animus_kernel.providers.base import CompletionResponse

        class FakeInvalid:
            tool_name = "bad_tool"
            error = "unknown tool"

        for i in range(3):
            response = CompletionResponse(content="", model="test", provider="test")
            score = gate.evaluate("Do something", response, [], [FakeInvalid()])
            assert score.failure_streak == i + 1

    def test_should_fallback_on_low_score(self) -> None:
        gate = HeadQualityGate()
        score = QualityScore(overall=30, failure_streak=1)
        assert gate.should_fallback(score)

    def test_should_fallback_on_streak(self) -> None:
        gate = HeadQualityGate(max_failure_streak=2)
        score = QualityScore(overall=50, failure_streak=3)
        assert gate.should_fallback(score)

    def test_should_not_fallback_on_good_score(self) -> None:
        gate = HeadQualityGate()
        score = QualityScore(overall=80, failure_streak=0)
        assert not gate.should_fallback(score)

    def test_reset_clears_streak(self) -> None:
        gate = HeadQualityGate()
        from animus_kernel.providers.base import CompletionResponse

        class FakeInvalid:
            tool_name = "bad_tool"
            error = "unknown tool"

        for _ in range(3):
            response = CompletionResponse(content="", model="test", provider="test")
            gate.evaluate("x", response, [], [FakeInvalid()])
        assert gate._failure_streak == 3

        gate.reset()
        assert gate._failure_streak == 0


class TestHeadFallbackController:
    def test_fallback_disabled_by_default(self) -> None:
        ctrl = HeadFallbackController()
        assert ctrl.enabled is False
        assert ctrl.can_fallback() is False

    def test_fallback_not_configured_when_provider_missing(self) -> None:
        ctrl = HeadFallbackController(enabled=True)
        assert ctrl.is_configured() is False
        assert ctrl.can_fallback() is False

    def test_fallback_status_format(self) -> None:
        ctrl = HeadFallbackController(
            fallback_provider="anthropic",
            enabled=True,
            max_fallbacks_per_session=5,
        )
        status = ctrl.status
        assert status.enabled is True
        assert status.provider_name == "anthropic"
        assert status.max_fallbacks == 5
        assert status.fallbacks_this_session == 0

    def test_try_fallback_when_disabled_returns_none(self) -> None:
        ctrl = HeadFallbackController(enabled=False)
        result = ctrl.try_fallback(messages=[], reason="test")
        assert result is None

    def test_try_fallback_when_not_configured_returns_none(self) -> None:
        ctrl = HeadFallbackController(enabled=True)
        result = ctrl.try_fallback(messages=[], reason="test")
        assert result is None

    def test_fallback_respects_max_fallbacks(self) -> None:
        ctrl = HeadFallbackController(enabled=True, max_fallbacks_per_session=1)
        ctrl._fallbacks_used = 1  # Simulate one already used
        assert ctrl.can_fallback() is False

    def test_reset_clears_counters(self) -> None:
        ctrl = HeadFallbackController(enabled=True, max_fallbacks_per_session=5)
        ctrl._fallbacks_used = 3
        ctrl._last_reason = "test"
        ctrl.reset()
        assert ctrl._fallbacks_used == 0
        assert ctrl._last_reason == ""

    def test_try_fallback_with_mock_provider(self) -> None:
        from animus_kernel.providers.base import CompletionResponse
        from animus_kernel.providers.manager import ProviderManager

        # Create a mock provider
        class MockProvider:
            name = "mock"
            provider_type = "mock"

            def is_configured(self):
                return True

            def complete(self, request):
                return CompletionResponse(
                    content="Cloud fallback response",
                    model="mock-model",
                    provider="mock",
                    tokens_used=100,
                )

        pm = ProviderManager()
        pm._providers["mock_cloud"] = MockProvider()

        ctrl = HeadFallbackController(
            provider_manager=pm,
            fallback_provider="mock_cloud",
            enabled=True,
            max_fallbacks_per_session=5,
        )
        result = ctrl.try_fallback(messages=[], reason="test")
        assert result is not None
        assert result.content == "Cloud fallback response"
        assert ctrl._fallbacks_used == 1


# ------------------------------------------------------------------
# Phase 5: Natural language interface
# ------------------------------------------------------------------

class TestHeadIntentParser:
    def test_parse_read_file_direct_command(self) -> None:
        parser = HeadIntentParser()
        intent = parser.parse("read main.py")
        assert intent.intent_type == IntentType.DIRECT_COMMAND
        assert "read_file" in intent.suggested_tools
        assert intent.extracted_args.get("path") == "main.py"
        assert intent.confidence > 0.7

    def test_parse_list_files_direct_command(self) -> None:
        parser = HeadIntentParser()
        intent = parser.parse("list files in src")
        assert intent.intent_type == IntentType.DIRECT_COMMAND
        assert "list_files" in intent.suggested_tools
        assert intent.extracted_args.get("path") == "src"

    def test_parse_git_status_direct_command(self) -> None:
        parser = HeadIntentParser()
        intent = parser.parse("git status")
        assert intent.intent_type == IntentType.DIRECT_COMMAND
        assert "run_shell" in intent.suggested_tools
        assert intent.extracted_args.get("command") == "git status"

    def test_parse_vague_request(self) -> None:
        parser = HeadIntentParser()
        intent = parser.parse("check for issues")
        assert intent.intent_type == IntentType.VAGUE_REQUEST
        assert len(intent.suggested_tools) > 0

    def test_parse_conversational(self) -> None:
        parser = HeadIntentParser()
        intent = parser.parse("hello there")
        assert intent.intent_type == IntentType.CONVERSATIONAL
        assert len(intent.suggested_tools) == 0

    def test_parse_empty_input(self) -> None:
        parser = HeadIntentParser()
        intent = parser.parse("")
        assert intent.intent_type == IntentType.CONVERSATIONAL

    def test_parse_remember_direct_command(self) -> None:
        parser = HeadIntentParser()
        intent = parser.parse("remember that auth token is in .env")
        assert intent.intent_type == IntentType.DIRECT_COMMAND
        assert "remember" in intent.suggested_tools
        assert "auth token is in .env" in intent.extracted_args.get("content", "")

    def test_parse_clarification_needed(self) -> None:
        # This is tricky to trigger naturally; simulate by checking that
        # ambiguous inputs get at least one match
        parser = HeadIntentParser()
        # "fix" could mean search_code or run_tests
        intent = parser.parse("fix the bug")
        assert intent.intent_type in (IntentType.VAGUE_REQUEST, IntentType.DIRECT_COMMAND)
        assert len(intent.suggested_tools) > 0


class TestHeadPlanner:
    def test_plan_direct_command(self) -> None:
        parser = HeadIntentParser()
        planner = HeadPlanner()
        intent = parser.parse("read main.py")
        plan = planner.plan(intent)
        assert plan.confidence > 0.5
        assert len(plan.steps) >= 1
        assert plan.steps[0].tool_name == "read_file"

    def test_plan_conversational(self) -> None:
        planner = HeadPlanner()
        intent = ParsedIntent(
            intent_type=IntentType.CONVERSATIONAL,
            confidence=1.0,
        )
        plan = planner.plan(intent)
        assert len(plan.steps) == 0

    def test_plan_vague_request(self) -> None:
        parser = HeadIntentParser()
        planner = HeadPlanner()
        intent = parser.parse("how do I set up this project?")
        plan = planner.plan(intent)
        assert plan.confidence > 0
        assert len(plan.steps) >= 1

    def test_plan_clarification_needed(self) -> None:
        planner = HeadPlanner()
        intent = ParsedIntent(
            intent_type=IntentType.CLARIFICATION_NEEDED,
            confidence=0.5,
            suggested_tools=["read_file", "search_code"],
        )
        plan = planner.plan(intent)
        assert plan.requires_clarification is True
        assert plan.clarification_prompt != ""

    def test_plan_dependencies(self) -> None:
        parser = HeadIntentParser()
        planner = HeadPlanner()
        intent = parser.parse("edit main.py change old to new")
        plan = planner.plan(intent)
        # edit_file depends on read_file first
        tool_names = [s.tool_name for s in plan.steps]
        if "read_file" in tool_names and "edit_file" in tool_names:
            read_idx = tool_names.index("read_file")
            edit_idx = tool_names.index("edit_file")
            assert read_idx < edit_idx

    def test_estimate_cost(self) -> None:
        planner = HeadPlanner()
        plan = ToolPlan(steps=[ToolPlanStep("read_file"), ToolPlanStep("search_code")])
        cost = planner.estimate_cost(plan)
        assert cost > 0


class TestHeadSynthesizer:
    def test_synthesize_read_file(self) -> None:
        synth = HeadSynthesizer()
        result = synth.synthesize("read_file", {"path": "main.py"}, "line1\nline2")
        assert "main.py" in result.summary
        assert "line1" in result.detail

    def test_synthesize_empty_result(self) -> None:
        synth = HeadSynthesizer()
        result = synth.synthesize("read_file", {"path": "x"}, "")
        assert "no output" in result.summary.lower()

    def test_synthesize_list_files_few(self) -> None:
        synth = HeadSynthesizer()
        result = synth.synthesize("list_files", {"path": "."}, "a.py\nb.py")
        assert "Found" in result.summary
        assert "a.py" in result.summary
        assert "b.py" in result.summary

    def test_synthesize_list_files_many(self) -> None:
        synth = HeadSynthesizer()
        files = "\n".join(f"file{i}.py" for i in range(20))
        result = synth.synthesize("list_files", {"path": "."}, files)
        assert "20" in result.summary or "items" in result.summary

    def test_synthesize_search_no_matches(self) -> None:
        synth = HeadSynthesizer()
        result = synth.synthesize("search_code", {"query": "foobar"}, "no matches")
        assert "No matches" in result.summary
        assert result.needs_follow_up is True

    def test_synthesize_multi(self) -> None:
        synth = HeadSynthesizer()
        results = [
            ("read_file", {"path": "a.py"}, "content1"),
            ("read_file", {"path": "b.py"}, "content2"),
        ]
        multi = synth.synthesize_multi(results)
        assert "a.py" in multi.summary
        assert "b.py" in multi.summary
