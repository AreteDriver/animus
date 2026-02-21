"""Process supervisor with automatic restart on crash."""

from __future__ import annotations

import logging
import subprocess
import time

logger = logging.getLogger(__name__)

_SIGTERM_TIMEOUT = 10
_SUBPROCESS_TIMEOUT = 30


class AnimusSupervisor:
    """Supervises a child process, restarting it on crash up to a configurable limit.

    Args:
        target: The command to run (e.g. "python -m animus_bootstrap.daemon").
        max_restarts: Maximum consecutive restarts before giving up.
        restart_delay: Seconds to wait between restart attempts.
    """

    def __init__(
        self,
        target: str,
        max_restarts: int = 5,
        restart_delay: float = 2.0,
    ) -> None:
        self.target = target
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self._process: subprocess.Popen[bytes] | None = None
        self._restart_count: int = 0

    @property
    def pid(self) -> int | None:
        """Return the PID of the supervised process, or None if not running."""
        if self._process is not None and self._process.poll() is None:
            return self._process.pid
        return None

    def start(self) -> None:
        """Start the target process.

        Raises:
            OSError: If the process cannot be started.
        """
        if self.is_running():
            logger.warning("Process already running (PID %s)", self.pid)
            return

        logger.info("Starting supervised process: %s", self.target)
        try:
            self._process = subprocess.Popen(
                self.target.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info("Process started with PID %s", self._process.pid)
        except OSError:
            logger.exception("Failed to start process: %s", self.target)
            raise

    def stop(self) -> None:
        """Stop the supervised process gracefully.

        Sends SIGTERM first, then SIGKILL after timeout if the process
        does not exit.
        """
        if self._process is None or self._process.poll() is not None:
            logger.debug("No running process to stop")
            self._process = None
            return

        pid = self._process.pid
        logger.info("Stopping process (PID %s) — sending SIGTERM", pid)

        try:
            self._process.terminate()
            self._process.wait(timeout=_SIGTERM_TIMEOUT)
            logger.info("Process (PID %s) terminated gracefully", pid)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Process (PID %s) did not exit after %ds — sending SIGKILL",
                pid,
                _SIGTERM_TIMEOUT,
            )
            self._process.kill()
            try:
                self._process.wait(timeout=_SUBPROCESS_TIMEOUT)
            except subprocess.TimeoutExpired:
                logger.error("Process (PID %s) did not respond to SIGKILL", pid)
        except OSError as exc:
            logger.error("Error stopping process (PID %s): %s", pid, exc)
        finally:
            self._process = None

    def is_running(self) -> bool:
        """Check if the supervised process is alive.

        Returns:
            True if the process exists and has not exited.
        """
        if self._process is None:
            return False
        return self._process.poll() is None

    def run_supervised(self) -> None:
        """Main supervision loop.

        Starts the process, monitors it, and restarts on crash up to
        ``max_restarts`` consecutive times. The restart counter resets
        when the process runs successfully (exits with code 0).

        The loop exits when:
        - The process exits cleanly (code 0).
        - The restart limit is exhausted.
        - A KeyboardInterrupt is received.
        """
        logger.info(
            "Supervisor starting — target=%s, max_restarts=%d, delay=%.1fs",
            self.target,
            self.max_restarts,
            self.restart_delay,
        )
        self._restart_count = 0

        while True:
            try:
                self.start()
            except OSError:
                self._restart_count += 1
                if self._restart_count > self.max_restarts:
                    logger.error(
                        "Max restarts (%d) exceeded — supervisor giving up",
                        self.max_restarts,
                    )
                    return
                logger.info(
                    "Retry %d/%d in %.1fs...",
                    self._restart_count,
                    self.max_restarts,
                    self.restart_delay,
                )
                time.sleep(self.restart_delay)
                continue

            # Wait for the process to exit
            try:
                assert self._process is not None  # guaranteed by start()
                return_code = self._process.wait()
            except KeyboardInterrupt:
                logger.info("Supervisor received interrupt — shutting down")
                self.stop()
                return

            if return_code == 0:
                logger.info("Process exited cleanly (code 0)")
                self._process = None
                return

            # Process crashed
            self._restart_count += 1
            logger.warning(
                "Process exited with code %d (restart %d/%d)",
                return_code,
                self._restart_count,
                self.max_restarts,
            )

            if self._restart_count > self.max_restarts:
                logger.error(
                    "Max restarts (%d) exceeded — supervisor giving up",
                    self.max_restarts,
                )
                self._process = None
                return

            logger.info("Restarting in %.1fs...", self.restart_delay)
            try:
                time.sleep(self.restart_delay)
            except KeyboardInterrupt:
                logger.info("Supervisor received interrupt during delay — shutting down")
                self._process = None
                return
