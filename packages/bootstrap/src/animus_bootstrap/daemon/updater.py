"""Auto-updater for the Animus Bootstrap package."""

from __future__ import annotations

import logging
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

logger = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT = 30
_PYPI_PACKAGE = "animus-bootstrap"
_PYPI_URL = f"https://pypi.org/pypi/{_PYPI_PACKAGE}/json"


class AnimusUpdater:
    """Checks for and applies updates to the animus-bootstrap package."""

    def get_current_version(self) -> str:
        """Get the currently installed version of animus-bootstrap.

        Returns:
            Version string (e.g. "0.1.0"), or "0.0.0" if not installed.
        """
        try:
            ver = version(_PYPI_PACKAGE)
            logger.debug("Current version: %s", ver)
            return ver
        except PackageNotFoundError:
            logger.warning("%s is not installed — returning 0.0.0", _PYPI_PACKAGE)
            return "0.0.0"

    def get_latest_version(self) -> str:
        """Fetch the latest version from PyPI.

        Uses httpx to query the PyPI JSON API.

        Returns:
            Latest version string, or "0.0.0" on failure.
        """
        try:
            import httpx
        except ImportError:
            logger.error("httpx is not installed — cannot check for updates")
            return "0.0.0"

        try:
            response = httpx.get(_PYPI_URL, timeout=_SUBPROCESS_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            latest = data["info"]["version"]
            logger.info("Latest version on PyPI: %s", latest)
            return latest
        except httpx.HTTPStatusError as exc:
            logger.error("PyPI returned HTTP %d: %s", exc.response.status_code, exc)
            return "0.0.0"
        except httpx.RequestError as exc:
            logger.error("Failed to reach PyPI: %s", exc)
            return "0.0.0"
        except (KeyError, ValueError) as exc:
            logger.error("Unexpected PyPI response format: %s", exc)
            return "0.0.0"

    def is_update_available(self) -> bool:
        """Check whether a newer version exists on PyPI.

        Uses packaging.version for proper semver comparison.

        Returns:
            True if a newer version is available.
        """
        try:
            from packaging.version import Version
        except ImportError:
            logger.error("packaging is not installed — cannot compare versions")
            return False

        current = self.get_current_version()
        latest = self.get_latest_version()

        try:
            update_available = Version(latest) > Version(current)
        except ValueError as exc:
            logger.error("Invalid version string: %s", exc)
            return False

        if update_available:
            logger.info("Update available: %s -> %s", current, latest)
        else:
            logger.info("Already up to date (%s)", current)

        return update_available

    def apply_update(self) -> bool:
        """Upgrade animus-bootstrap to the latest version via pip.

        Returns:
            True if the upgrade succeeded.
        """
        logger.info("Applying update for %s...", _PYPI_PACKAGE)
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            _PYPI_PACKAGE,
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )
            logger.info("Update successful")
            logger.debug("pip output: %s", result.stdout.decode(errors="replace"))
            return True
        except subprocess.CalledProcessError as exc:
            logger.error(
                "pip upgrade failed (exit %d): %s",
                exc.returncode,
                exc.stderr.decode(errors="replace") if exc.stderr else str(exc),
            )
            return False
        except subprocess.TimeoutExpired:
            logger.error("pip upgrade timed out after %ds", _SUBPROCESS_TIMEOUT)
            return False
