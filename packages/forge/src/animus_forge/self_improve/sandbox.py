"""Sandbox for isolated execution of self-improvement changes."""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
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
            # Run pytest
            result = await self._run_command(
                ["python", "-m", "pytest", "-v", "--tb=short"],
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
                ["python", "-m", "ruff", "check", "."],
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

    async def validate_changes(self) -> SandboxResult:
        """Validate all changes by running tests and lint.

        Returns:
            Combined result.
        """
        lint_result = await self.run_lint()
        test_result = await self.run_tests()

        # Combine results
        all_passed = lint_result.lint_passed and test_result.tests_passed
        status = SandboxStatus.SUCCESS if all_passed else SandboxStatus.FAILED

        return SandboxResult(
            status=status,
            tests_passed=test_result.tests_passed,
            test_output=test_result.test_output,
            lint_passed=lint_result.lint_passed,
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
    ) -> subprocess.CompletedProcess:
        """Run a command in the sandbox.

        Args:
            cmd: Command and arguments.

        Returns:
            Completed process result.
        """
        if not self._sandbox_path:
            raise RuntimeError("Sandbox not created")

        clean_env = self._sanitize_env()

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self._sandbox_path),
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
