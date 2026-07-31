"""SEC-00 — execution-plane security regression tests for animus kernel.

Reproduces defects SEC-04 and SEC-05 from
``security/SEC-00-threat-model.md``:

- ForgeToolRegistry validates only the first command token and uses
  ``subprocess.run(..., shell=True)``.
- HeadToolOrchestrator repeats the same shell=True pattern.

All proofs monkeypatch ``subprocess.run`` so no real command is executed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus_kernel.head.tool_orchestrator import HeadToolOrchestrator
from animus_kernel.tools.registry import ForgeToolRegistry


# ═══════════════════════════════════════════════════════════════════
# SEC-04 — ForgeToolRegistry token-only validation + shell=True
# ═══════════════════════════════════════════════════════════════════


class TestForgeToolRegistryShellInjection:
    def test_run_command_uses_shell_true(self):
        """_handle_run_command passes shell=True to subprocess.run even though only
        the first token was validated against the allowlist."""
        registry = ForgeToolRegistry(
            project_root=Path.cwd(),
            enable_shell=True,
            allowed_commands=["python"],
        )

        captured: dict = {}

        def _fake_subprocess_run(cmd, *, shell=False, **kwargs):
            captured["cmd"] = cmd
            captured["shell"] = shell
            return MagicMock(returncode=0, stdout="safe-output", stderr="")

        with patch("subprocess.run", side_effect=_fake_subprocess_run):
            result = registry.execute(
                "run_command",
                {"command": "python -c \"print('injected body')\"", "timeout": 30},
                agent_id="test-agent",
            )

        assert result is not None
        assert captured.get("shell") is True, (
            "Expected shell=True in subprocess.run; "
            f"captured={captured}"
        )
        # The full string is handed to the shell; only the first token was checked.
        assert captured.get("cmd") == "python -c \"print('injected body')\""

    def test_only_first_token_validated(self):
        """The allowlist check uses ``Path(cmd_parts[0]).name``; everything after the
        first whitespace token is unreviewed by the registry."""
        registry = ForgeToolRegistry(
            project_root=Path.cwd(),
            enable_shell=True,
            allowed_commands=["echo"],
        )

        captured: dict = {}

        def _fake_subprocess_run(cmd, *, shell=False, **kwargs):
            captured["cmd"] = cmd
            captured["shell"] = shell
            return MagicMock(returncode=0, stdout="ok", stderr="")

        with patch("subprocess.run", side_effect=_fake_subprocess_run):
            registry.execute(
                "run_command",
                {"command": "echo hello; not-a-real-token-but-passes-first-check", "timeout": 30},
                agent_id="test-agent",
            )

        assert captured.get("shell") is True
        # Pre-fix: the entire command string reaches the shell.
        assert "not-a-real-token-but-passes-first-check" in captured.get("cmd", "")


# ═══════════════════════════════════════════════════════════════════
# SEC-05 — HeadToolOrchestrator repeats shell=True
# ═══════════════════════════════════════════════════════════════════


class TestHeadToolOrchestratorShellInjection:
    def test_run_shell_uses_shell_true(self, tmp_path: Path):
        """HeadToolOrchestrator._handle_run_shell also uses shell=True with only a
        base-command allowlist check."""
        orchestrator = HeadToolOrchestrator(
            project_root=tmp_path,
            memory_dir=tmp_path / "memory",
            enable_shell=True,
            allowed_commands=["python"],
        )

        captured: dict = {}

        def _fake_subprocess_run(cmd, *, shell=False, capture_output=False, text=False, cwd=None, timeout=None):
            captured["cmd"] = cmd
            captured["shell"] = shell
            captured["cwd"] = cwd
            captured["timeout"] = timeout
            return MagicMock(returncode=0, stdout="head-ok", stderr="")

        with patch("subprocess.run", side_effect=_fake_subprocess_run):
            result = orchestrator.execute(
                "run_shell",
                {"command": "python -c \"print('head injected body')\"", "cwd": str(tmp_path)},
            )

        assert captured.get("shell") is True, (
            "Expected HeadToolOrchestrator to use shell=True; "
            f"captured={captured}"
        )
        assert captured.get("cmd") == "python -c \"print('head injected body')\""
        assert "head-ok" in result
