"""Animus dependency installer and service manager."""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT = 30

# Minimum Python version required
_MIN_PYTHON = (3, 11)

# Required dependencies — must be present for Animus to function
_REQUIRED_DEPS = ("python3", "pip")

# Optional dependencies — enhance functionality but not strictly required
_OPTIONAL_DEPS = ("ollama", "ffmpeg")


class AnimusInstaller:
    """Detects environment, installs dependencies, and manages the Animus service."""

    def detect_os(self) -> str:
        """Detect the current operating system.

        Returns:
            One of "linux", "macos", or "windows".
        """
        system = platform.system().lower()
        mapping = {
            "linux": "linux",
            "darwin": "macos",
            "windows": "windows",
        }
        result = mapping.get(system, "linux")
        logger.info("Detected OS: %s", result)
        return result

    def detect_package_manager(self) -> str:
        """Detect the system package manager.

        Returns:
            One of "apt", "dnf", "brew", "winget", or "unknown".
        """
        # Check in priority order
        managers = [
            ("apt", "apt"),
            ("dnf", "dnf"),
            ("brew", "brew"),
            ("winget", "winget"),
        ]
        for name, binary in managers:
            if shutil.which(binary):
                logger.info("Detected package manager: %s", name)
                return name

        logger.warning("No known package manager detected")
        return "unknown"

    def check_dependencies(self) -> dict[str, bool]:
        """Check which dependencies are available on the system.

        Checks required deps (python3.11+, pip) and optional deps (ollama, ffmpeg).

        Returns:
            Dict mapping dependency name to whether it is available.
        """
        results: dict[str, bool] = {}

        # Python version check — need 3.11+
        python_ok = sys.version_info >= _MIN_PYTHON
        results["python3"] = python_ok
        if python_ok:
            logger.info(
                "Python %s.%s detected (>= %s.%s required)",
                sys.version_info.major,
                sys.version_info.minor,
                *_MIN_PYTHON,
            )
        else:
            logger.warning(
                "Python %s.%s detected, but >= %s.%s is required",
                sys.version_info.major,
                sys.version_info.minor,
                *_MIN_PYTHON,
            )

        # pip check
        pip_path = shutil.which("pip") or shutil.which("pip3")
        results["pip"] = pip_path is not None
        if pip_path:
            logger.info("pip found at: %s", pip_path)
        else:
            logger.warning("pip not found on PATH")

        # Optional deps
        for dep in _OPTIONAL_DEPS:
            found = shutil.which(dep) is not None
            results[dep] = found
            level = logging.INFO if found else logging.DEBUG
            logger.log(level, "%s: %s", dep, "found" if found else "not found")

        return results

    def install_missing_deps(self, deps: dict[str, bool]) -> list[str]:
        """Install missing required dependencies.

        Only attempts to install deps that are marked as False in the input dict.
        Optional deps (ollama, ffmpeg) are skipped — only required deps are installed.

        Args:
            deps: Dict from check_dependencies().

        Returns:
            List of dependency names that were successfully installed.
        """
        pkg_mgr = self.detect_package_manager()
        installed: list[str] = []

        for dep_name in _REQUIRED_DEPS:
            if deps.get(dep_name, True):
                continue  # Already present

            cmd = self._build_install_command(pkg_mgr, dep_name)
            if not cmd:
                logger.error(
                    "Cannot install %s: no install command for package manager '%s'",
                    dep_name,
                    pkg_mgr,
                )
                continue

            logger.info("Installing %s via %s...", dep_name, pkg_mgr)
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    timeout=_SUBPROCESS_TIMEOUT,
                )
                installed.append(dep_name)
                logger.info("Successfully installed %s", dep_name)
            except subprocess.CalledProcessError as exc:
                logger.error(
                    "Failed to install %s: %s",
                    dep_name,
                    exc.stderr.decode(errors="replace") if exc.stderr else str(exc),
                )
            except subprocess.TimeoutExpired:
                logger.error(
                    "Timed out installing %s (timeout=%ds)",
                    dep_name,
                    _SUBPROCESS_TIMEOUT,
                )

        return installed

    def register_service(self) -> bool:
        """Register Animus as a system service on the detected OS.

        Returns:
            True if the service was registered successfully.
        """
        os_name = self.detect_os()
        python_path = sys.executable
        module = "animus_bootstrap.daemon"

        try:
            if os_name == "linux":
                from animus_bootstrap.daemon.platforms.linux import LinuxService

                svc = LinuxService()
                unit_content = svc.generate_systemd_unit(python_path, module)
                if not unit_content:
                    return False
                return svc.install_unit()

            if os_name == "macos":
                from animus_bootstrap.daemon.platforms.macos import MacOSService

                svc = MacOSService()
                plist_content = svc.generate_plist(python_path, module)
                if not plist_content:
                    return False
                return svc.install_plist()

            if os_name == "windows":
                from animus_bootstrap.daemon.platforms.windows import WindowsService

                svc = WindowsService()
                return svc.install_service()

        except ImportError:
            logger.error("Platform module not available for %s", os_name)
            return False
        except NotImplementedError as exc:
            logger.error("Platform not yet supported: %s", exc)
            return False

        logger.error("Unsupported OS: %s", os_name)
        return False

    def is_service_running(self) -> bool:
        """Check if the Animus service is currently running."""
        os_name = self.detect_os()
        try:
            if os_name == "linux":
                from animus_bootstrap.daemon.platforms.linux import LinuxService

                return LinuxService().is_running()
            if os_name == "macos":
                from animus_bootstrap.daemon.platforms.macos import MacOSService

                return MacOSService().is_running()
            if os_name == "windows":
                from animus_bootstrap.daemon.platforms.windows import WindowsService

                return WindowsService().is_running()
        except (ImportError, NotImplementedError):
            logger.debug("Cannot check service status on %s", os_name)
        return False

    def start_service(self) -> bool:
        """Start the Animus service."""
        os_name = self.detect_os()
        try:
            if os_name == "linux":
                from animus_bootstrap.daemon.platforms.linux import LinuxService

                return LinuxService().enable_and_start()
            if os_name == "macos":
                from animus_bootstrap.daemon.platforms.macos import MacOSService

                return MacOSService().load_agent()
            if os_name == "windows":
                from animus_bootstrap.daemon.platforms.windows import WindowsService

                return WindowsService().start()
        except (ImportError, NotImplementedError) as exc:
            logger.error("Cannot start service on %s: %s", os_name, exc)
        return False

    def stop_service(self) -> bool:
        """Stop the Animus service."""
        os_name = self.detect_os()
        try:
            if os_name == "linux":
                from animus_bootstrap.daemon.platforms.linux import LinuxService

                return LinuxService().stop()
            if os_name == "macos":
                from animus_bootstrap.daemon.platforms.macos import MacOSService

                return MacOSService().stop()
            if os_name == "windows":
                from animus_bootstrap.daemon.platforms.windows import WindowsService

                return WindowsService().stop()
        except (ImportError, NotImplementedError) as exc:
            logger.error("Cannot stop service on %s: %s", os_name, exc)
        return False

    def restart_service(self) -> bool:
        """Restart the Animus service (stop then start)."""
        logger.info("Restarting Animus service...")
        stopped = self.stop_service()
        if not stopped:
            logger.warning("Stop returned False — attempting start anyway")
        return self.start_service()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_install_command(pkg_mgr: str, dep_name: str) -> list[str] | None:
        """Build the shell command to install a dependency.

        Returns:
            Command list suitable for subprocess.run(), or None if unsupported.
        """
        # Map abstract dep names to package-manager-specific package names
        pkg_map: dict[str, dict[str, str]] = {
            "apt": {"python3": "python3.11", "pip": "python3-pip"},
            "dnf": {"python3": "python3.11", "pip": "python3-pip"},
            "brew": {"python3": "python@3.11", "pip": "python@3.11"},
            "winget": {"python3": "Python.Python.3.11", "pip": "Python.Python.3.11"},
        }

        manager_packages = pkg_map.get(pkg_mgr)
        if not manager_packages:
            return None

        package = manager_packages.get(dep_name)
        if not package:
            return None

        install_cmds: dict[str, list[str]] = {
            "apt": ["sudo", "apt", "install", "-y", package],
            "dnf": ["sudo", "dnf", "install", "-y", package],
            "brew": ["brew", "install", package],
            "winget": ["winget", "install", "--accept-package-agreements", package],
        }

        return install_cmds.get(pkg_mgr)
