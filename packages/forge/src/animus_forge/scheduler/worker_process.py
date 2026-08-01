"""Explicit subprocess worker wrapper with graceful terminate and hard kill.

This module replaces the implicit process management of ``ProcessPoolExecutor``.
Each ``WorkerProcess`` owns a real OS subprocess (via ``asyncio``), exposes its
PID, and can terminate the entire process group on Linux.  It is designed for
short-lived citizen tasks where the parent must be able to kill a hung or
runaway worker reliably.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    """Outcome of a worker subprocess."""

    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    killed: bool = False
    timed_out: bool = False
    returncode: int | None = None


@dataclass
class WorkerProcess:
    """A single citizen worker subprocess.

    The worker runs ``python -m animus_forge.scheduler.worker_main`` and
    communicates via stdin/stdout JSON.
    """

    task_id: str
    mission_id: str
    citizen_role: str
    context_json: dict[str, Any]
    description: str = ""
    timeout_seconds: float = 300.0
    grace_period_seconds: float = 5.0
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    pid: int | None = field(default=None)
    cancelled: bool = field(default=False)

    async def start(self) -> bool:
        """Spawn the worker subprocess.

        Returns:
            ``True`` if the process started.
        """
        import json

        payload = {
            "task_id": self.task_id,
            "mission_id": self.mission_id,
            "citizen_role": self.citizen_role,
            "description": self.description,
            "context": self.context_json,
        }

        cmd = [
            sys.executable,
            "-m",
            "animus_forge.scheduler.worker_main",
        ]

        try:
            # Start in a new process group so we can kill the whole tree later.
            # start_new_session=True is POSIX; on Windows this has different
            # semantics (no process group kill), so we gate the tree-kill logic.
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            self.pid = self.process.pid

            # Write the payload and close stdin so the worker can proceed.
            stdin = self.process.stdin
            assert stdin is not None
            stdin.write(json.dumps(payload).encode())
            await stdin.drain()
            stdin.close()
            await stdin.wait_closed()

            logger.info("Worker started for task %s (pid=%s)", self.task_id, self.pid)
            return True
        except Exception as exc:
            logger.error("Failed to start worker for task %s: %s", self.task_id, exc)
            self.cancelled = True
            return False

    async def wait(self) -> WorkerResult:
        """Wait for the worker to finish, respecting its timeout.

        If the timeout is exceeded, the worker is terminated and a timeout
        result is returned.
        """
        if self.process is None or self.process.returncode is not None:
            return WorkerResult(ok=False, error="worker not running")

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                self.process.communicate(),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "Worker for task %s (pid=%s) timed out after %.1fs",
                self.task_id,
                self.pid,
                self.timeout_seconds,
            )
            await self.terminate()
            return WorkerResult(
                ok=False,
                error=f"Worker timed out after {self.timeout_seconds}s",
                killed=True,
                timed_out=True,
                returncode=self.process.returncode,
            )

        returncode = self.process.returncode
        stderr = stderr_b.decode("utf-8", errors="replace").strip()
        if returncode != 0:
            logger.error(
                "Worker for task %s exited with code %s: %s",
                self.task_id,
                returncode,
                stderr,
            )
            return WorkerResult(
                ok=False,
                error=f"Worker exited with code {returncode}: {stderr}",
                killed=self.cancelled,
                returncode=returncode,
            )

        import json

        try:
            stdout = stdout_b.decode("utf-8", errors="replace").strip()
            if not stdout:
                raise ValueError("empty worker output")
            # Last non-empty line is the JSON result.
            lines = [line for line in stdout.splitlines() if line.strip()]
            data = json.loads(lines[-1])
            return WorkerResult(ok=True, data=data, returncode=returncode)
        except Exception as exc:
            logger.error(
                "Failed to parse worker output for task %s: %s (stderr: %s)",
                self.task_id,
                exc,
                stderr,
            )
            return WorkerResult(
                ok=False,
                error=f"Failed to parse worker output: {exc}",
                returncode=returncode,
            )

    async def terminate(self) -> None:
        """Gracefully terminate the worker, then hard-kill if necessary.

        On Linux this kills the process group so any children spawned by the
        worker are also terminated.
        """
        if self.process is None or self.process.returncode is not None:
            return

        self.cancelled = True
        pid = self.pid
        if pid is None:
            return

        # Try graceful process-group termination on POSIX.
        if sys.platform != "win32" and os.getpgid(pid) == pid:
            try:
                os.killpg(pid, signal.SIGTERM)
                logger.info("Sent SIGTERM to process group %s for task %s", pid, self.task_id)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.warning("SIGTERM process group failed for task %s: %s", self.task_id, exc)
        else:
            try:
                self.process.terminate()
                logger.info("Sent terminate to worker for task %s", self.task_id)
            except ProcessLookupError:
                pass

        # Wait briefly for graceful exit.
        try:
            await asyncio.wait_for(self.process.wait(), timeout=self.grace_period_seconds)
            logger.info("Worker for task %s exited gracefully", self.task_id)
            return
        except TimeoutError:
            logger.warning(
                "Worker for task %s did not exit within %.1fs; forcing kill",
                self.task_id,
                self.grace_period_seconds,
            )

        # Hard kill.
        if sys.platform != "win32" and os.getpgid(pid) == pid:
            try:
                os.killpg(pid, signal.SIGKILL)
                logger.info("Sent SIGKILL to process group %s for task %s", pid, self.task_id)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.warning("SIGKILL process group failed for task %s: %s", self.task_id, exc)
        else:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass

        # Reap.
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5.0)
        except TimeoutError:
            logger.error("Worker for task %s could not be reaped", self.task_id)
