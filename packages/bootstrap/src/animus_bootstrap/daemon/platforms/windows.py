"""Windows service management via sc.exe.

Uses the Windows Service Control Manager (sc.exe) to manage the Animus
Bootstrap daemon as a native Windows service. No pywin32 dependency required.
"""

from __future__ import annotations

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

_SERVICE_NAME = "AnimusBootstrap"
_DISPLAY_NAME = "Animus Bootstrap Daemon"
_SUBPROCESS_TIMEOUT = 30


def _build_binpath() -> str:
    """Build the service binary path using pythonw.exe.

    Returns:
        Command string for sc.exe binPath= argument.
    """
    python_dir = sys.exec_prefix
    pythonw = f"{python_dir}\\pythonw.exe"
    return f'"{pythonw}" -m animus_bootstrap.daemon'


def _run_sc(*args: str) -> subprocess.CompletedProcess[str]:
    """Run an sc.exe command with standard options.

    Args:
        *args: Arguments to pass after ``sc.exe``.

    Returns:
        Completed process result.

    Raises:
        FileNotFoundError: If sc.exe is not found on PATH.
        subprocess.CalledProcessError: If sc.exe exits non-zero.
    """
    cmd = ["sc.exe", *args]
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
        check=True,
    )


class WindowsService:
    """Manages Animus Bootstrap as a Windows service via sc.exe."""

    def install_service(self) -> bool:
        """Install Animus Bootstrap as a Windows service.

        Registers the service with the Windows Service Control Manager
        using ``sc.exe create``. The service is configured to run
        ``pythonw.exe -m animus_bootstrap.daemon``.

        Returns:
            True if the service was created successfully, False otherwise.
        """
        binpath = _build_binpath()
        try:
            _run_sc(
                "create",
                _SERVICE_NAME,
                f"binPath= {binpath}",
                f"DisplayName= {_DISPLAY_NAME}",
                "start= demand",
            )
            logger.info(
                "Installed Windows service '%s' with binPath: %s",
                _SERVICE_NAME,
                binpath,
            )
            return True
        except FileNotFoundError:
            logger.error("sc.exe not found — is this a Windows system?")
            return False
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Failed to create service '%s': %s",
                _SERVICE_NAME,
                exc.stderr.strip() if exc.stderr else str(exc),
            )
            return False
        except subprocess.TimeoutExpired:
            logger.error("sc.exe create timed out after %ds", _SUBPROCESS_TIMEOUT)
            return False

    def start(self) -> bool:
        """Start the Animus Bootstrap Windows service.

        Returns:
            True if the service was started successfully, False otherwise.
        """
        try:
            _run_sc("start", _SERVICE_NAME)
            logger.info("Started Windows service '%s'", _SERVICE_NAME)
            return True
        except FileNotFoundError:
            logger.error("sc.exe not found — is this a Windows system?")
            return False
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Failed to start service '%s': %s",
                _SERVICE_NAME,
                exc.stderr.strip() if exc.stderr else str(exc),
            )
            return False
        except subprocess.TimeoutExpired:
            logger.error("sc.exe start timed out after %ds", _SUBPROCESS_TIMEOUT)
            return False

    def stop(self) -> bool:
        """Stop the Animus Bootstrap Windows service.

        Returns:
            True if the service was stopped successfully, False otherwise.
        """
        try:
            _run_sc("stop", _SERVICE_NAME)
            logger.info("Stopped Windows service '%s'", _SERVICE_NAME)
            return True
        except FileNotFoundError:
            logger.error("sc.exe not found — is this a Windows system?")
            return False
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Failed to stop service '%s': %s",
                _SERVICE_NAME,
                exc.stderr.strip() if exc.stderr else str(exc),
            )
            return False
        except subprocess.TimeoutExpired:
            logger.error("sc.exe stop timed out after %ds", _SUBPROCESS_TIMEOUT)
            return False

    def is_running(self) -> bool:
        """Check if the Animus Bootstrap Windows service is running.

        Queries the service state via ``sc.exe query`` and parses the
        output for ``STATE`` containing ``RUNNING``.

        Returns:
            True if the service is in the RUNNING state, False otherwise.
        """
        try:
            result = subprocess.run(
                ["sc.exe", "query", _SERVICE_NAME],
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )
            running = "RUNNING" in result.stdout
            logger.debug(
                "Service '%s': %s",
                _SERVICE_NAME,
                "running" if running else "not running",
            )
            return running
        except FileNotFoundError:
            logger.error("sc.exe not found — is this a Windows system?")
            return False
        except subprocess.TimeoutExpired:
            logger.error("sc.exe query timed out after %ds", _SUBPROCESS_TIMEOUT)
            return False

    def uninstall_service(self) -> bool:
        """Uninstall the Animus Bootstrap Windows service.

        Attempts to stop the service first (errors are ignored), then
        deletes the service registration via ``sc.exe delete``.

        Returns:
            True if the service was deleted successfully, False otherwise.
        """
        # Best-effort stop before delete
        if self.is_running():
            logger.info("Stopping service before uninstall...")
            self.stop()

        try:
            _run_sc("delete", _SERVICE_NAME)
            logger.info("Uninstalled Windows service '%s'", _SERVICE_NAME)
            return True
        except FileNotFoundError:
            logger.error("sc.exe not found — is this a Windows system?")
            return False
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Failed to delete service '%s': %s",
                _SERVICE_NAME,
                exc.stderr.strip() if exc.stderr else str(exc),
            )
            return False
        except subprocess.TimeoutExpired:
            logger.error("sc.exe delete timed out after %ds", _SUBPROCESS_TIMEOUT)
            return False
