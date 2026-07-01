"""Sandbox for isolated execution of self-improvement changes."""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SandboxStatus(str, Enum):
    """Status of sandbox execution."""

    CREATED = "created"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SandboxResult:
    """Result of sandbox execution."""

    status: SandboxStatus
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0
    tests_passed: bool = False
    test_output: str = ""
    lint_passed: bool = False
    lint_output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Sandbox:
    """Isolated environment for testing changes."""

    def __init__(
        self,
        source_path: Path | str,
        timeout: int = 300,
        cleanup_on_exit: bool = True,
    ):
        """Initialize sandbox.

        Args:
            source_path: Path to source repository.
            timeout: Timeout for operations in seconds.
            cleanup_on_exit: Whether to clean up temp directory on exit.
        """
        self.source_path = Path(source_path)
        self.timeout = timeout
        self.cleanup_on_exit = cleanup_on_exit
        self._temp_dir: Path | None = None
        self._sandbox_path: Path | None = None
        self._status = SandboxStatus.CREATED

    @property
    def sandbox_path(self) -> Path | None:
        """Get path to sandbox directory."""
        return self._sandbox_path

    @property
    def status(self) -> SandboxStatus:
        """Get current status."""
        return self._status

    def create(self) -> Path:
        """Create the sandbox environment.

        Returns:
            Path to sandbox directory.
        """
        if self._sandbox_path:
            return self._sandbox_path

        # Create temp directory
        self._temp_dir = Path(tempfile.mkdtemp(prefix="gorgon_sandbox_"))
        self._sandbox_path = self._temp_dir / "workspace"

        # Copy source to sandbox
        logger.info(f"Creating sandbox at {self._sandbox_path}")
        shutil.copytree(
            self.source_path,
            self._sandbox_path,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                "__pycache__",
                "*.pyc",
                ".mypy_cache",
                ".pytest_cache",
                "node_modules",
                "dist",
                "build",
            ),
        )

        return self._sandbox_path

    def cleanup(self) -> None:
        """Clean up sandbox directory."""
        if self._temp_dir and self._temp_dir.exists():
            logger.info(f"Cleaning up sandbox at {self._temp_dir}")
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
            self._sandbox_path = None

    def __enter__(self) -> Sandbox:
        """Context manager entry."""
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self.cleanup_on_exit:
            self.cleanup()

    async def apply_changes(self, changes: dict[str, str]) -> bool:
        """Apply file changes to sandbox.

        Args:
            changes: Dict mapping file paths to new content.

        Returns:
            True if changes applied successfully.
        """
        if not self._sandbox_path:
            raise RuntimeError("Sandbox not created")

        try:
            for file_path, content in changes.items():
                full_path = self._sandbox_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                logger.debug(f"Applied changes to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to apply changes: {e}")
            return False

    async def run_tests(self) -> SandboxResult:
        """Run tests in sandbox.

        Returns:
            Result of test execution.
        """
        if not self._sandbox_path:
            return SandboxResult(
                status=SandboxStatus.FAILED,
                error="Sandbox not created",
            )

        self._status = SandboxStatus.RUNNING
        start_time = datetime.now()

        try:
            # Run pytest using the current Python interpreter (preserves venv)
            result = await self._run_command(
                [sys.executable, "-m", "pytest", "-v", "--tb=short"],
            )

            duration = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                self._status = SandboxStatus.SUCCESS
                return SandboxResult(
                    status=SandboxStatus.SUCCESS,
                    exit_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    duration_seconds=duration,
                    tests_passed=True,
                    test_output=result.stdout,
                )
            else:
                self._status = SandboxStatus.FAILED
                return SandboxResult(
                    status=SandboxStatus.FAILED,
                    exit_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    duration_seconds=duration,
                    tests_passed=False,
                    test_output=result.stdout + result.stderr,
                )

        except TimeoutError:
            self._status = SandboxStatus.TIMEOUT
            return SandboxResult(
                status=SandboxStatus.TIMEOUT,
                error=f"Test execution timed out after {self.timeout}s",
            )
        except Exception as e:
            self._status = SandboxStatus.FAILED
            return SandboxResult(
                status=SandboxStatus.FAILED,
                error=str(e),
            )

    async def run_lint(self) -> SandboxResult:
        """Run linting in sandbox.

        Returns:
            Result of lint execution.
        """
        if not self._sandbox_path:
            return SandboxResult(
                status=SandboxStatus.FAILED,
                error="Sandbox not created",
            )

        try:
            result = await self._run_command(
                [sys.executable, "-m", "ruff", "check", "."],
            )

            return SandboxResult(
                status=SandboxStatus.SUCCESS if result.returncode == 0 else SandboxStatus.FAILED,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                lint_passed=result.returncode == 0,
                lint_output=result.stdout + result.stderr,
            )

        except Exception as e:
            return SandboxResult(
                status=SandboxStatus.FAILED,
                error=str(e),
            )

    @staticmethod
    def _count_failures(output: str | None) -> int:
        """Count test failures from pytest output."""
        if not output:
            return 0
        import re

        match = re.search(r"(\d+) failed", output)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _count_lint_errors(output: str | None) -> int:
        """Count lint errors from ruff output."""
        if not output:
            return 0
        import re

        match = re.search(r"Found (\d+) error", output)
        return int(match.group(1)) if match else 0

    async def validate_changes(self) -> SandboxResult:
        """Validate all changes by running tests and lint.

        Compares against baseline: if the same number of failures exist before
        and after changes, the changes are considered clean (pre-existing failures).

        Returns:
            Combined result.
        """
        # Run baseline (before changes) to detect pre-existing failures
        # We do this by checking only modified files' tests if possible
        lint_result = await self.run_lint()
        test_result = await self.run_tests()

        # Check if failures are pre-existing by comparing against source codebase
        tests_clean = test_result.tests_passed
        lint_clean = lint_result.lint_passed

        if not tests_clean and self.source_path:
            # Count failures — compare against baseline on original codebase
            post_failures = self._count_failures(test_result.test_output)
            # Run only the failing test files on original to check if pre-existing
            import re

            failing_files = set(re.findall(r"FAILED ([\w/]+\.py)::", test_result.test_output or ""))
            if failing_files:
                baseline_cmd = [sys.executable, "-m", "pytest", "-v", "--tb=line"] + list(
                    failing_files
                )
                baseline = await self._run_command(baseline_cmd, cwd=str(self.source_path))
                pre_failures = self._count_failures(baseline.stdout)
                if post_failures <= pre_failures:
                    logger.info(
                        f"Test failures are pre-existing ({pre_failures} baseline, "
                        f"{post_failures} with changes) — treating as clean"
                    )
                    tests_clean = True

        if not lint_clean and self.source_path:
            post_errors = self._count_lint_errors(lint_result.lint_output)
            baseline_lint = await self._run_command(
                [sys.executable, "-m", "ruff", "check", "."],
                cwd=str(self.source_path),
            )
            pre_errors = self._count_lint_errors(baseline_lint.stdout + baseline_lint.stderr)
            if post_errors <= pre_errors:
                logger.info(
                    f"Lint errors are pre-existing ({pre_errors} baseline, "
                    f"{post_errors} with changes) — treating as clean"
                )
                lint_clean = True

        all_passed = tests_clean and lint_clean
        status = SandboxStatus.SUCCESS if all_passed else SandboxStatus.FAILED

        return SandboxResult(
            status=status,
            tests_passed=tests_clean,
            test_output=test_result.test_output,
            lint_passed=lint_clean,
            lint_output=lint_result.lint_output,
            duration_seconds=test_result.duration_seconds,
        )

    @staticmethod
    def _sanitize_env() -> dict[str, str]:
        """Build a clean environment dict for subprocess execution.

        Keeps only safe variables and strips anything that looks like
        a secret (keys, tokens, passwords).

        Returns:
            Sanitized environment dictionary.
        """
        import os
        import re

        secret_pattern = re.compile(r"_(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)S?$", re.IGNORECASE)
        safe_vars = {"PATH", "HOME", "LANG", "TERM", "PYTHONPATH", "LC_ALL", "LC_CTYPE", "USER"}

        clean_env: dict[str, str] = {}
        for key, value in os.environ.items():
            if key in safe_vars:
                clean_env[key] = value
            elif not secret_pattern.search(key):
                # Allow non-secret vars that aren't in safe_vars
                # but still filter anything that looks sensitive
                clean_env[key] = value

        return clean_env

    async def _run_command(
        self,
        cmd: list[str],
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a command in the sandbox or a specified directory.

        Args:
            cmd: Command and arguments.
            cwd: Working directory override. Defaults to sandbox path.

        Returns:
            Completed process result.
        """
        work_dir = cwd or (str(self._sandbox_path) if self._sandbox_path else None)
        if not work_dir:
            raise RuntimeError("Sandbox not created and no cwd specified")

        clean_env = self._sanitize_env()

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=clean_env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout,
            )
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode or 0,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
