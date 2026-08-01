"""SEC-03 — execution-plane shell-elimination regression tests for animus kernel.

Verifies that agent-reachable shell execution uses ``subprocess.run(...,
shell=False)`` with an argv list, and that shell metacharacters, command
chaining, pipes, redirects, command substitution, newline injection, quoted
payloads, interpreter code-execution flags, and PATH shadowing are rejected.

All subprocess monkeypatching captures the exact arguments passed so no real
command is executed for the security-negative cases.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus_kernel.executor.executor_integrations import IntegrationHandlersMixin
from animus_kernel.head.tool_orchestrator import HeadToolOrchestrator
from animus_kernel.tools.registry import ForgeToolRegistry

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _fake_subprocess_run_factory(captured: dict):
    """Return a fake subprocess.run that records shell flag, argv and timeout."""

    def _fake(cmd, *, shell=False, timeout=None, **kwargs):
        captured["cmd"] = cmd
        captured["shell"] = shell
        captured["timeout"] = timeout
        for key, value in kwargs.items():
            captured[key] = value
        return MagicMock(returncode=0, stdout="safe-output", stderr="")

    return _fake


# ═══════════════════════════════════════════════════════════════════
# SEC-03 — ForgeToolRegistry shell elimination
# ═══════════════════════════════════════════════════════════════════


class TestForgeToolRegistryShellElimination:
    def test_run_command_uses_shell_false_with_argv(self):
        registry = ForgeToolRegistry(
            project_root=Path.cwd(),
            enable_shell=True,
            allowed_commands=["python"],
        )

        captured: dict = {}
        fake = _fake_subprocess_run_factory(captured)

        with patch("subprocess.run", side_effect=fake):
            registry.execute(
                "run_command",
                {"command": "python --version", "timeout": 30},
                agent_id="test-agent",
            )

        assert captured.get("shell") is False, f"captured={captured}"
        assert captured.get("cmd") == ["python", "--version"]

    @pytest.mark.parametrize(
        "payload",
        [
            "python --version; whoami",
            "python --version && whoami",
            "python --version || whoami",
            "cat /etc/passwd | wc -l",
            "echo hi > /tmp/pwned",
            "echo hi < /etc/passwd",
            "echo $(whoami)",
            "echo `whoami`",
            "python --version\nwhoami",
            "python --version\r\nwhoami",
            'python -c "print(1)"',
            "node -e 'console.log(1)'",
        ],
    )
    def test_rejects_shell_metacharacters(self, payload: str):
        registry = ForgeToolRegistry(
            project_root=Path.cwd(),
            enable_shell=True,
            allowed_commands=["python", "node", "echo", "cat"],
        )

        result = registry.execute(
            "run_command",
            {"command": payload, "timeout": 30},
            agent_id="test-agent",
        )

        assert "forbidden shell metacharacter" in result or "code-execution flag" in result

    def test_rejects_python_code_execution_flag(self):
        registry = ForgeToolRegistry(
            project_root=Path.cwd(),
            enable_shell=True,
            allowed_commands=["python"],
        )

        result = registry.execute(
            "run_command",
            {"command": "python -c print(1)", "timeout": 30},
            agent_id="test-agent",
        )

        assert "code-execution flag is not allowed" in result

    def test_rejects_node_code_execution_flag(self):
        registry = ForgeToolRegistry(
            project_root=Path.cwd(),
            enable_shell=True,
            allowed_commands=["node"],
        )

        result = registry.execute(
            "run_command",
            {"command": "node -e console.log(1)", "timeout": 30},
            agent_id="test-agent",
        )

        assert "code-execution flag is not allowed" in result

    @pytest.mark.parametrize(
        "payload",
        [
            "/usr/bin/python --version",
            "../bin/python --version",
            "./python --version",
        ],
    )
    def test_rejects_path_arguments(self, payload: str):
        registry = ForgeToolRegistry(
            project_root=Path.cwd(),
            enable_shell=True,
            allowed_commands=["python"],
        )

        result = registry.execute(
            "run_command",
            {"command": payload, "timeout": 30},
            agent_id="test-agent",
        )

        assert "must be a bare name, not a path" in result

    def test_rejects_timeout_too_large(self):
        registry = ForgeToolRegistry(
            project_root=Path.cwd(),
            enable_shell=True,
            allowed_commands=["echo"],
        )

        captured: dict = {}
        fake = _fake_subprocess_run_factory(captured)

        with patch("subprocess.run", side_effect=fake):
            registry.execute(
                "run_command",
                {"command": "echo hi", "timeout": 9999},
                agent_id="test-agent",
            )

        assert captured.get("timeout") == 300

    def test_allows_simple_command(self):
        registry = ForgeToolRegistry(
            project_root=Path.cwd(),
            enable_shell=True,
            allowed_commands=["echo"],
        )

        captured: dict = {}
        fake = _fake_subprocess_run_factory(captured)

        with patch("subprocess.run", side_effect=fake):
            result = registry.execute(
                "run_command",
                {"command": "echo hello world", "timeout": 30},
                agent_id="test-agent",
            )

        assert captured.get("shell") is False
        assert captured.get("cmd") == ["echo", "hello", "world"]
        assert "safe-output" in result

    def test_allows_command_with_safe_arguments(self):
        registry = ForgeToolRegistry(
            project_root=Path.cwd(),
            enable_shell=True,
            allowed_commands=["git"],
        )

        captured: dict = {}
        fake = _fake_subprocess_run_factory(captured)

        with patch("subprocess.run", side_effect=fake):
            registry.execute(
                "run_command",
                {"command": "git status --short", "timeout": 30},
                agent_id="test-agent",
            )

        assert captured.get("shell") is False
        assert captured.get("cmd") == ["git", "status", "--short"]

    def test_cwd_is_project_root(self):
        project_root = Path("/tmp/fake-project")
        registry = ForgeToolRegistry(
            project_root=project_root,
            enable_shell=True,
            allowed_commands=["pwd"],
        )

        captured: dict = {}
        fake = _fake_subprocess_run_factory(captured)

        with patch("subprocess.run", side_effect=fake):
            registry.execute(
                "run_command",
                {"command": "pwd", "timeout": 30},
                agent_id="test-agent",
            )

        assert captured.get("cwd") == str(project_root)


# ═══════════════════════════════════════════════════════════════════
# SEC-03 — HeadToolOrchestrator shell elimination
# ═══════════════════════════════════════════════════════════════════


class TestHeadToolOrchestratorShellElimination:
    def test_run_shell_uses_shell_false_with_argv(self, tmp_path: Path):
        orchestrator = HeadToolOrchestrator(
            project_root=tmp_path,
            memory_dir=tmp_path / "memory",
            enable_shell=True,
            allowed_commands=["python"],
        )

        captured: dict = {}
        fake = _fake_subprocess_run_factory(captured)

        with patch("subprocess.run", side_effect=fake):
            orchestrator.execute(
                "run_shell",
                {"command": "python --version", "cwd": str(tmp_path)},
            )

        assert captured.get("shell") is False, f"captured={captured}"
        assert captured.get("cmd") == ["python", "--version"]

    @pytest.mark.parametrize(
        "payload",
        [
            "python --version; whoami",
            "python --version && whoami",
            "python --version || whoami",
            "cat /etc/passwd | wc -l",
            "echo hi > /tmp/pwned",
            "echo $(whoami)",
            "echo `whoami`",
            "python --version\nwhoami",
            'python -c "print(1)"',
        ],
    )
    def test_rejects_shell_metacharacters(self, payload: str, tmp_path: Path):
        orchestrator = HeadToolOrchestrator(
            project_root=tmp_path,
            memory_dir=tmp_path / "memory",
            enable_shell=True,
            allowed_commands=["python", "echo", "cat", "node"],
        )

        result = orchestrator.execute(
            "run_shell",
            {"command": payload, "cwd": str(tmp_path)},
        )

        assert "forbidden shell metacharacter" in result or "code-execution flag" in result

    def test_rejects_python_c_flag(self, tmp_path: Path):
        orchestrator = HeadToolOrchestrator(
            project_root=tmp_path,
            memory_dir=tmp_path / "memory",
            enable_shell=True,
            allowed_commands=["python"],
        )

        result = orchestrator.execute(
            "run_shell",
            {"command": "python -c print(1)", "cwd": str(tmp_path)},
        )

        assert "code-execution flag is not allowed" in result

    @pytest.mark.parametrize(
        "payload",
        [
            "/usr/bin/python --version",
            "../bin/python --version",
        ],
    )
    def test_rejects_path_arguments(self, payload: str, tmp_path: Path):
        orchestrator = HeadToolOrchestrator(
            project_root=tmp_path,
            memory_dir=tmp_path / "memory",
            enable_shell=True,
            allowed_commands=["python"],
        )

        result = orchestrator.execute("run_shell", {"command": payload})

        assert "must be a bare name, not a path" in result

    def test_rejects_cwd_outside_project_root(self, tmp_path: Path):
        orchestrator = HeadToolOrchestrator(
            project_root=tmp_path,
            memory_dir=tmp_path / "memory",
            enable_shell=True,
            allowed_commands=["pwd"],
        )

        result = orchestrator.execute(
            "run_shell",
            {"command": "pwd", "cwd": "/tmp"},
        )

        assert "outside project root" in result

    def test_rejects_timeout_too_large(self, tmp_path: Path):
        orchestrator = HeadToolOrchestrator(
            project_root=tmp_path,
            memory_dir=tmp_path / "memory",
            enable_shell=True,
            allowed_commands=["echo"],
        )

        captured: dict = {}
        fake = _fake_subprocess_run_factory(captured)

        with patch("subprocess.run", side_effect=fake):
            orchestrator.execute(
                "run_shell",
                {"command": "echo hi", "timeout": 9999, "cwd": str(tmp_path)},
            )

        assert captured.get("timeout") == 300

    def test_allows_simple_command(self, tmp_path: Path):
        orchestrator = HeadToolOrchestrator(
            project_root=tmp_path,
            memory_dir=tmp_path / "memory",
            enable_shell=True,
            allowed_commands=["echo"],
        )

        captured: dict = {}
        fake = _fake_subprocess_run_factory(captured)

        with patch("subprocess.run", side_effect=fake):
            result = orchestrator.execute(
                "run_shell",
                {"command": "echo hello world", "cwd": str(tmp_path)},
            )

        assert captured.get("shell") is False
        assert captured.get("cmd") == ["echo", "hello", "world"]
        assert "safe-output" in result


# ═══════════════════════════════════════════════════════════════════
# SEC-03 — Workflow executor shell elimination
# ═══════════════════════════════════════════════════════════════════


class _StubExecutor(IntegrationHandlersMixin):
    """Minimal executor stub for exercising the shell-step mixin."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self._context: dict = {}
        self.fallback_callbacks: dict = {}


class TestExecutorShellStepElimination:
    def test_shell_step_uses_shell_false_with_argv(self):
        from animus_kernel.executor.loader import StepConfig

        executor = _StubExecutor()
        step = StepConfig(
            id="s1",
            type="shell",
            params={"command": "echo hello world"},
        )

        captured: dict = {}
        fake = _fake_subprocess_run_factory(captured)

        with patch("subprocess.run", side_effect=fake):
            executor._execute_shell(step, {})

        assert captured.get("shell") is False, f"captured={captured}"
        assert captured.get("cmd") == ["echo", "hello", "world"]

    @pytest.mark.parametrize(
        "payload",
        [
            "echo hi; whoami",
            "echo hi && whoami",
            "echo hi || whoami",
            "cat /etc/passwd | wc -l",
            "echo hi > /tmp/pwned",
            "echo $(whoami)",
            "echo `whoami`",
            "echo hi\nwhoami",
            'python -c "print(1)"',
        ],
    )
    def test_rejects_shell_metacharacters_after_substitution(self, payload: str):
        from animus_kernel.executor.loader import StepConfig

        executor = _StubExecutor()
        step = StepConfig(
            id="s2",
            type="shell",
            params={"command": payload},
        )

        with pytest.raises(ValueError, match="forbidden shell metacharacter|code-execution flag"):
            executor._execute_shell(step, {})

    def test_rejects_absolute_path_in_shell_step(self):
        from animus_kernel.executor.loader import StepConfig

        executor = _StubExecutor()
        step = StepConfig(
            id="s3",
            type="shell",
            params={"command": "/usr/bin/echo hello"},
        )

        with pytest.raises(ValueError, match="bare command name"):
            executor._execute_shell(step, {})

    def test_allows_substituted_command_with_safe_arguments(self):
        from animus_kernel.executor.loader import StepConfig

        executor = _StubExecutor()
        step = StepConfig(
            id="s4",
            type="shell",
            params={"command": "echo ${greeting}"},
        )

        captured: dict = {}
        fake = _fake_subprocess_run_factory(captured)

        with patch("subprocess.run", side_effect=fake):
            executor._execute_shell(step, {"greeting": "hello world"})

        assert captured.get("shell") is False
        assert captured.get("cmd") == ["echo", "hello world"]
