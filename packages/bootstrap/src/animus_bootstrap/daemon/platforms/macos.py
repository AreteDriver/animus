"""macOS launchd agent management."""

from __future__ import annotations

import logging
import subprocess
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT = 30
_AGENT_LABEL = "dev.animus"
_LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
_PLIST_FILE = _LAUNCH_AGENTS_DIR / f"{_AGENT_LABEL}.plist"


class MacOSService:
    """Manages Animus as a macOS launchd user agent."""

    def generate_plist(self, python_path: str, module: str) -> str:
        """Generate a launchd plist XML for the Animus daemon.

        Args:
            python_path: Absolute path to the Python interpreter.
            module: Python module to run (e.g. "animus_bootstrap.daemon").

        Returns:
            Plist XML content as a string.
        """
        log_dir = Path.home() / "Library" / "Logs" / "Animus"

        plist = textwrap.dedent(f"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
              "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
            <plist version="1.0">
            <dict>
                <key>Label</key>
                <string>{_AGENT_LABEL}</string>

                <key>ProgramArguments</key>
                <array>
                    <string>{python_path}</string>
                    <string>-m</string>
                    <string>{module}</string>
                </array>

                <key>RunAtLoad</key>
                <true/>

                <key>KeepAlive</key>
                <dict>
                    <key>SuccessfulExit</key>
                    <false/>
                </dict>

                <key>StandardOutPath</key>
                <string>{log_dir}/animus.out.log</string>

                <key>StandardErrorPath</key>
                <string>{log_dir}/animus.err.log</string>

                <key>EnvironmentVariables</key>
                <dict>
                    <key>PYTHONUNBUFFERED</key>
                    <string>1</string>
                </dict>
            </dict>
            </plist>
        """)
        logger.debug("Generated plist:\n%s", plist)
        return plist

    def install_plist(self) -> bool:
        """Write the plist file to ~/Library/LaunchAgents/.

        Creates the directory and log directory if they don't exist.

        Returns:
            True if the plist was written successfully.
        """
        import sys

        python_path = sys.executable
        module = "animus_bootstrap.daemon"
        content = self.generate_plist(python_path, module)

        # Ensure directories exist
        log_dir = Path.home() / "Library" / "Logs" / "Animus"
        try:
            _LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            _PLIST_FILE.write_text(content)
            logger.info("Installed plist: %s", _PLIST_FILE)
            return True
        except OSError as exc:
            logger.error("Failed to write plist: %s", exc)
            return False

    def load_agent(self) -> bool:
        """Load the launchd agent (starts the service).

        Returns:
            True if launchctl load succeeded.
        """
        try:
            subprocess.run(
                ["launchctl", "load", str(_PLIST_FILE)],
                check=True,
                capture_output=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )
            logger.info("Loaded launchd agent: %s", _AGENT_LABEL)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error(
                "launchctl load failed: %s",
                exc.stderr.decode(errors="replace") if exc.stderr else str(exc),
            )
            return False
        except subprocess.TimeoutExpired:
            logger.error("launchctl load timed out")
            return False
        except FileNotFoundError:
            logger.error("launchctl not found — is this macOS?")
            return False

    def stop(self) -> bool:
        """Unload the launchd agent (stops the service).

        Returns:
            True if launchctl unload succeeded.
        """
        try:
            subprocess.run(
                ["launchctl", "unload", str(_PLIST_FILE)],
                check=True,
                capture_output=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )
            logger.info("Unloaded launchd agent: %s", _AGENT_LABEL)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error(
                "launchctl unload failed: %s",
                exc.stderr.decode(errors="replace") if exc.stderr else str(exc),
            )
            return False
        except subprocess.TimeoutExpired:
            logger.error("launchctl unload timed out")
            return False
        except FileNotFoundError:
            logger.error("launchctl not found")
            return False

    def is_running(self) -> bool:
        """Check if the launchd agent is loaded and running.

        Returns:
            True if the agent appears in launchctl list output.
        """
        try:
            result = subprocess.run(
                ["launchctl", "list", _AGENT_LABEL],
                capture_output=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )
            # launchctl list <label> exits 0 if loaded, non-zero if not
            running = result.returncode == 0
            logger.debug("Agent %s: %s", _AGENT_LABEL, "running" if running else "not running")
            return running
        except subprocess.TimeoutExpired:
            logger.error("launchctl list timed out")
            return False
        except FileNotFoundError:
            logger.error("launchctl not found")
            return False
