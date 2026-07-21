"""Tests for sandbox isolation and validation."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from animus_kernel.sandbox.sandbox import Sandbox, SandboxResult, SandboxStatus


# ═══════════════════════════════════════════════════════════════════
# Sandbox creation and lifecycle tests
# ═══════════════════════════════════════════════════════════════════


class TestSandboxLifecycle:
    def test_create_makes_temp_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "main.py").write_text("print('hello')")

            sandbox = Sandbox(source)
            path = sandbox.create()

            assert path.exists()
            # sandbox.create() returns <temp_dir>/workspace
            assert path.name == "workspace"
            assert path.parent.name.startswith("animus_kernel_sandbox_")
            assert (path / "main.py").exists()
            assert (path / "main.py").read_text() == "print('hello')"

            sandbox.cleanup()
            assert not path.exists()

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "file.txt").write_text("content")

            with Sandbox(source) as sandbox:
                assert sandbox.sandbox_path is not None
                assert (sandbox.sandbox_path / "file.txt").exists()

            assert sandbox.sandbox_path is None or not sandbox.sandbox_path.exists()

    def test_create_returns_same_path_on_reuse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()

            sandbox = Sandbox(source)
            p1 = sandbox.create()
            p2 = sandbox.create()
            assert p1 == p2
            sandbox.cleanup()

    def test_ignores_patterns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "main.py").write_text("x")
            (source / "__pycache__").mkdir()
            (source / ".venv").mkdir()
            (source / "node_modules").mkdir()
            (source / "dist").mkdir()

            with Sandbox(source) as sandbox:
                assert (sandbox.sandbox_path / "main.py").exists()
                assert not (sandbox.sandbox_path / "__pycache__").exists()
                assert not (sandbox.sandbox_path / ".venv").exists()
                assert not (sandbox.sandbox_path / "node_modules").exists()
                assert not (sandbox.sandbox_path / "dist").exists()

    def test_no_cleanup_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()

            sandbox = Sandbox(source, cleanup_on_exit=False)
            with sandbox:
                path = sandbox.sandbox_path

            assert path.exists()
            sandbox.cleanup()
            assert not path.exists()


# ═══════════════════════════════════════════════════════════════════
# Sandbox apply_changes tests
# ═══════════════════════════════════════════════════════════════════


class TestSandboxApplyChanges:
    @pytest.mark.asyncio
    async def test_apply_changes_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()

            with Sandbox(source) as sandbox:
                result = await sandbox.apply_changes(
                    {"src/main.py": "print('new')", "README.md": "# Hello"}
                )
                assert result is True
                assert (sandbox.sandbox_path / "src" / "main.py").read_text() == "print('new')"
                assert (sandbox.sandbox_path / "README.md").read_text() == "# Hello"

    @pytest.mark.asyncio
    async def test_apply_changes_fails_when_not_created(self):
        source = Path(tempfile.gettempdir()) / "dummy_source"
        sandbox = Sandbox(source)
        with pytest.raises(RuntimeError, match="Sandbox not created"):
            await sandbox.apply_changes({"a.py": "b"})


# ═══════════════════════════════════════════════════════════════════
# Sandbox command execution tests
# ═══════════════════════════════════════════════════════════════════


class TestSandboxCommands:
    @pytest.mark.asyncio
    async def test_run_command_success(self):
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()

            with Sandbox(source) as sandbox:
                result = await sandbox._run_command(
                    [sys.executable, "-c", "print('hello')"]
                )
                assert result.returncode == 0
                assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_run_command_timeout(self):
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()

            sandbox = Sandbox(source, timeout=1)
            sandbox.create()

            with pytest.raises(TimeoutError):
                await sandbox._run_command(
                    [sys.executable, "-c", "import time; time.sleep(10)"]
                )

            sandbox.cleanup()

    def test_sanitize_env_filters_secrets(self):
        import os

        original = os.environ.copy()
        try:
            os.environ["SAFE_VAR"] = "value"
            os.environ["API_KEY"] = "secret"
            os.environ["MY_TOKEN"] = "token"

            clean = Sandbox._sanitize_env()
            assert clean.get("SAFE_VAR") == "value"
            assert "API_KEY" not in clean
            assert "MY_TOKEN" not in clean
        finally:
            os.environ.clear()
            os.environ.update(original)


# ═══════════════════════════════════════════════════════════════════
# Sandbox test/lint execution tests
# ═══════════════════════════════════════════════════════════════════


class TestSandboxValidation:
    @pytest.mark.asyncio
    async def test_run_tests_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "test_dummy.py").write_text(
                "def test_ok():\n    assert 1 + 1 == 2\n"
            )

            with Sandbox(source) as sandbox:
                result = await sandbox.run_tests()
                assert result.status == SandboxStatus.SUCCESS
                assert result.tests_passed is True
                assert "test_dummy.py" in result.test_output

    @pytest.mark.asyncio
    async def test_run_tests_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "test_broken.py").write_text(
                "def test_fail():\n    assert 1 + 1 == 3\n"
            )

            with Sandbox(source) as sandbox:
                result = await sandbox.run_tests()
                assert result.status == SandboxStatus.FAILED
                assert result.tests_passed is False

    @pytest.mark.asyncio
    async def test_run_lint_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "main.py").write_text("print('hello')\n")

            with Sandbox(source) as sandbox:
                result = await sandbox.run_lint()
                # ruff may find no issues on trivial code
                assert result.lint_passed is True

    @pytest.mark.asyncio
    async def test_validate_changes_detects_preexisting_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            # Pre-existing broken test in source
            (source / "test_broken.py").write_text(
                "def test_fail():\n    assert 1 + 1 == 3\n"
            )

            with Sandbox(source) as sandbox:
                result = await sandbox.validate_changes()
                # Should treat pre-existing failures as clean
                assert result.tests_passed is True
                assert result.lint_passed is True

    def test_count_failures_parsing(self):
        assert Sandbox._count_failures("1 failed, 2 passed") == 1
        assert Sandbox._count_failures("10 failed, 5 passed") == 10
        assert Sandbox._count_failures("all good") == 0
        assert Sandbox._count_failures(None) == 0

    def test_count_lint_errors_parsing(self):
        assert Sandbox._count_lint_errors("Found 3 errors") == 3
        assert Sandbox._count_lint_errors("Found 0 errors") == 0
        assert Sandbox._count_lint_errors("all good") == 0


# ═══════════════════════════════════════════════════════════════════
# SandboxResult dataclass tests
# ═══════════════════════════════════════════════════════════════════


class TestSandboxResult:
    def test_default_values(self):
        result = SandboxResult(status=SandboxStatus.CREATED)
        assert result.exit_code == 0
        assert result.tests_passed is False
        assert result.lint_passed is False
        assert result.error is None

    def test_custom_values(self):
        result = SandboxResult(
            status=SandboxStatus.SUCCESS,
            exit_code=0,
            tests_passed=True,
            lint_passed=True,
            duration_seconds=5.5,
        )
        assert result.tests_passed is True
        assert result.lint_passed is True
        assert result.duration_seconds == 5.5

    def test_performance_fields_default(self):
        result = SandboxResult(status=SandboxStatus.CREATED)
        assert result.performance_regression is False
        assert result.benchmark_before == {}
        assert result.benchmark_after == {}


# ═══════════════════════════════════════════════════════════════════
# Benchmark tests
# ═══════════════════════════════════════════════════════════════════


class TestSandboxBenchmarks:
    @pytest.mark.asyncio
    async def test_run_benchmarks_skipped_when_not_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()

            with Sandbox(source) as sandbox:
                result = await sandbox.run_benchmarks()
                assert result.status == SandboxStatus.SUCCESS
                assert result.metadata.get("benchmark_skipped") is True

    @pytest.mark.asyncio
    async def test_validate_changes_includes_benchmark_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "test_dummy.py").write_text(
                "def test_ok():\n    assert 1 + 1 == 2\n"
            )

            with Sandbox(source) as sandbox:
                result = await sandbox.validate_changes()
                assert result.performance_regression is False
                assert result.benchmark_before == {}
                assert result.benchmark_after == {}

    @pytest.mark.asyncio
    async def test_validate_changes_detects_performance_regression(self):
        import subprocess
        from unittest.mock import AsyncMock

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "test_dummy.py").write_text(
                "def test_ok():\n    assert 1 + 1 == 2\n"
            )

            with Sandbox(source) as sandbox:
                # Mock _run_command to simulate benchmark data
                calls = []

                async def fake_run(cmd, cwd=None):
                    # cmd is a list like [sys.executable, "-m", "pytest", ...]
                    stdout = ""
                    stderr = ""
                    returncode = 0

                    if "--collect-only" in cmd:
                        stdout = "collected 1 item\n<BenchmarkFixture test_bench>"
                    elif "--benchmark-only" in cmd and cwd == str(source):
                        # Baseline (before changes) — fast
                        stdout = "test_bench\n    1 loop, best of 5: 10.0 ms per call\n"
                    elif "--benchmark-only" in cmd:
                        # After changes — slow (>10% regression)
                        stdout = "test_bench\n    1 loop, best of 5: 12.0 ms per call\n"
                    elif "pytest" in cmd and "-v" in cmd:
                        stdout = "1 passed\n"
                    elif "ruff" in cmd:
                        stdout = "Found 0 errors\n"

                    calls.append((cmd, cwd))
                    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

                sandbox._run_command = fake_run  # type: ignore[method-assign]

                result = await sandbox.validate_changes()
                assert result.performance_regression is True
                assert result.benchmark_before == {"bench": {"value": 10.0, "unit": "ms"}}
                assert result.benchmark_after == {"bench": {"value": 12.0, "unit": "ms"}}
                assert result.status == SandboxStatus.FAILED
