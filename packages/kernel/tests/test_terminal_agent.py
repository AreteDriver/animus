"""Tests for TerminalAgent iterative build loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_kernel.builder.command_runner import CommandResult
from animus_kernel.builder.terminal_agent import (
    BuildCheckpoint,
    BuildResult,
    TerminalAgent,
)
from animus_kernel.tools.filesystem import FilesystemTools
from animus_kernel.tools.safety import PathValidator


@pytest.fixture
def mock_fs(tmp_path):
    """FilesystemTools-ish mock with a real project root."""
    fs = MagicMock()
    fs.project_root = tmp_path
    fs.glob_files.return_value = ["main.py"]
    fs.get_structure.return_value = {"dirs": [], "files": ["main.py"]}
    fs.read_file.return_value = "print('hello')"
    search_result = MagicMock()
    search_result.matches = [MagicMock(path="main.py")]
    fs.search_code.return_value = search_result
    return fs


@pytest.fixture
def mock_budget():
    """BudgetManager-ish mock."""
    bm = MagicMock()
    bm.config.model_multipliers = {}
    bm.can_allocate.return_value = True
    bm.allocate.return_value = True
    record = MagicMock()
    record.tokens = 1000
    record.input_tokens = 500
    record.output_tokens = 500
    record.cache_read_tokens = 0
    record.metadata = {}
    bm.record_usage.return_value = record
    return bm


@pytest.fixture
def mock_rollback():
    """RollbackManager-ish mock."""
    rm = MagicMock()
    snapshot = MagicMock()
    snapshot.id = "snap-123"
    rm.create_snapshot.return_value = snapshot
    rm.rollback.return_value = True
    return rm


@pytest.fixture
def mock_supervisor():
    """Async-capable supervisor mock."""
    return AsyncMock()


@pytest.fixture
def patched_run_command(monkeypatch):
    """Patch command_runner.run to avoid real subprocesses."""
    def _fake_run(cmd, cwd, timeout=None, env=None):
        return CommandResult(
            exit_code=0,
            stdout="1 passed",
            stderr="",
            duration_ms=10.0,
            timeout=False,
            truncated=False,
        )

    monkeypatch.setattr(
        "animus_kernel.builder.terminal_agent.run_command",
        _fake_run,
    )
    return _fake_run


class TestTerminalAgentBuildSuccess:
    async def test_build_passes_first_iteration(
        self,
        tmp_path,
        mock_fs,
        mock_budget,
        mock_rollback,
        mock_supervisor,
        patched_run_command,
    ):
        (tmp_path / "main.py").write_text("print('hello')")
        agent = TerminalAgent(
            filesystem_tools=mock_fs,
            budget_manager=mock_budget,
            rollback_manager=mock_rollback,
            supervisor=mock_supervisor,
            test_command="pytest",
            max_iterations=3,
            budget_tokens_per_iteration=1000,
        )
        result = await agent.build("add feature", tmp_path)
        assert isinstance(result, BuildResult)
        assert result.success is True
        assert result.tests_passed is True
        assert result.iterations_used == 1
        assert "main.py" in result.files_changed or result.files_changed == []
        mock_supervisor.process_message.assert_awaited_once()
        mock_rollback.create_snapshot.assert_called_once()
        mock_budget.can_allocate.assert_called()
        mock_budget.allocate.assert_called()
        mock_budget.release.assert_called()

    async def test_build_no_supervisor(
        self,
        tmp_path,
        mock_fs,
        mock_budget,
        mock_rollback,
        patched_run_command,
    ):
        (tmp_path / "main.py").write_text("print('hello')")
        agent = TerminalAgent(
            filesystem_tools=mock_fs,
            budget_manager=mock_budget,
            rollback_manager=mock_rollback,
            supervisor=None,
            test_command="pytest",
            max_iterations=3,
        )
        result = await agent.build("add feature", tmp_path)
        assert result.success is True
        assert result.tests_passed is True

    async def test_build_detects_file_changes(
        self,
        tmp_path,
        mock_fs,
        mock_budget,
        mock_rollback,
        mock_supervisor,
        patched_run_command,
    ):
        (tmp_path / "main.py").write_text("print('hello')")

        # Mutate file during supervisor processing so _hash_files detects it
        async def _mutate(*a, **kw):
            (tmp_path / "main.py").write_text("print('changed')")

        mock_supervisor.process_message.side_effect = _mutate

        agent = TerminalAgent(
            filesystem_tools=mock_fs,
            budget_manager=mock_budget,
            rollback_manager=mock_rollback,
            supervisor=mock_supervisor,
            test_command="pytest",
            max_iterations=3,
        )
        result = await agent.build("add feature", tmp_path)
        assert result.success is True
        assert "main.py" in result.files_changed

    async def test_budget_gate_closed(
        self,
        tmp_path,
        mock_fs,
        mock_budget,
        mock_rollback,
        patched_run_command,
    ):
        (tmp_path / "main.py").write_text("print('hello')")
        mock_budget.can_allocate.return_value = False
        agent = TerminalAgent(
            filesystem_tools=mock_fs,
            budget_manager=mock_budget,
            rollback_manager=mock_rollback,
            supervisor=None,
            test_command="pytest",
            max_iterations=3,
        )
        result = await agent.build("add feature", tmp_path)
        assert result.success is False
        assert result.tests_passed is False
        assert result.iterations_used == 0

    async def test_build_fail_then_rollback(
        self,
        tmp_path,
        mock_fs,
        mock_budget,
        mock_rollback,
        mock_supervisor,
    ):
        (tmp_path / "main.py").write_text("print('hello')")
        fail_result = CommandResult(
            exit_code=1,
            stdout="",
            stderr="assert 0",
            duration_ms=10.0,
            timeout=False,
            truncated=False,
        )

        with patch(
            "animus_kernel.builder.terminal_agent.run_command",
            return_value=fail_result,
        ):
            agent = TerminalAgent(
                filesystem_tools=mock_fs,
                budget_manager=mock_budget,
                rollback_manager=mock_rollback,
                supervisor=mock_supervisor,
                test_command="pytest",
                max_iterations=2,
            )
            result = await agent.build("add feature", tmp_path)

        assert result.success is False
        assert result.tests_passed is False
        assert result.iterations_used == 2
        mock_rollback.rollback.assert_called_once()

    async def test_build_exception_then_rollback(
        self,
        tmp_path,
        mock_fs,
        mock_budget,
        mock_rollback,
    ):
        (tmp_path / "main.py").write_text("print('hello')")
        mock_fs.glob_files.side_effect = RuntimeError("boom")
        agent = TerminalAgent(
            filesystem_tools=mock_fs,
            budget_manager=mock_budget,
            rollback_manager=mock_rollback,
            supervisor=None,
            test_command="pytest",
            max_iterations=3,
        )
        result = await agent.build("add feature", tmp_path)
        assert result.success is False
        assert result.tests_passed is False

    async def test_checkpoint_preserved(
        self,
        tmp_path,
        mock_fs,
        mock_budget,
        mock_rollback,
        mock_supervisor,
        patched_run_command,
    ):
        (tmp_path / "main.py").write_text("print('hello')")
        agent = TerminalAgent(
            filesystem_tools=mock_fs,
            budget_manager=mock_budget,
            rollback_manager=mock_rollback,
            supervisor=mock_supervisor,
            test_command="pytest",
            max_iterations=3,
        )
        await agent.build("add feature", tmp_path)
        checkpoints = agent.get_checkpoints()
        assert len(checkpoints) >= 1
        cp = checkpoints[0]
        assert isinstance(cp, BuildCheckpoint)
        assert cp.iteration_count == 1
        assert cp.test_results.get("passed") is True

    async def test_et_consumed_non_negative(
        self,
        tmp_path,
        mock_fs,
        mock_budget,
        mock_rollback,
        mock_supervisor,
        patched_run_command,
    ):
        (tmp_path / "main.py").write_text("print('hello')")
        agent = TerminalAgent(
            filesystem_tools=mock_fs,
            budget_manager=mock_budget,
            rollback_manager=mock_rollback,
            supervisor=mock_supervisor,
            test_command="pytest",
            max_iterations=3,
        )
        result = await agent.build("add feature", tmp_path)
        assert result.et_consumed >= 0

    def test_real_fs_requires_pathvalidator(self, tmp_path):
        """Smoke test that real FilesystemTools can be wired up."""
        validator = PathValidator(project_path=str(tmp_path))
        fs = FilesystemTools(validator=validator)
        assert fs.project_root == tmp_path
