"""Tests for animus-kernel CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from animus_kernel.cli import app

runner = CliRunner()


class TestCLIAnalyze:
    def test_analyze_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["analyze", "--path", tmpdir])
            assert result.exit_code == 0
            assert "0 files" in result.output

    def test_analyze_detects_issues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text("def hello():\n    pass\n")

            result = runner.invoke(app, ["analyze", "--path", tmpdir])
            assert result.exit_code == 0
            assert (
                "Missing docstring" in result.output or "Missing module docstring" in result.output
            )

    def test_analyze_unknown_category(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["analyze", "--path", tmpdir, "--category", "nonexistent"])
            assert result.exit_code == 1
            assert "Unknown category" in result.output


class TestCLISelfImprove:
    def test_self_improve_static_only_no_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["self-improve", "--path", tmpdir])
            assert result.exit_code == 0
            assert "No AI provider available" in result.output or "Yes" in result.output

    def test_self_improve_auto_approve_blocked_without_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["self-improve", "--path", tmpdir, "--auto-approve"])
            assert result.exit_code == 1
            assert "Blocked" in result.output
