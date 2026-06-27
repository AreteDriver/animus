"""CLI-based faucet driver for command-line tools (e.g., `sui client faucet`)."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING

from animus_faucet.drivers.base import BaseDriver
from animus_faucet.models import ClaimResult, ClaimStatus

if TYPE_CHECKING:
    from animus_faucet.config import FaucetConfig
    from animus_faucet.models import FaucetDefinition

logger = logging.getLogger("animus_faucet.drivers.cli")


class CLIDriver(BaseDriver):
    """Driver for faucets accessed via local CLI commands.

    The primary use case is `sui client faucet` which is the most reliable
    Sui testnet faucet — no captcha, no rate limit drama.

    Config expects `driver_config` with:
        - command: shell command template with {address} placeholder
        - success_pattern: regex pattern in stdout indicating success
        - amount_pattern: optional regex to extract amount from stdout
        - timeout: command timeout in seconds (default 30)
    """

    def __init__(self, faucet: FaucetDefinition, config: FaucetConfig):
        super().__init__(faucet, config)
        dc = self.faucet.driver_config
        self.command_template = dc.get("command", "")
        self.success_pattern = dc.get("success_pattern", "")
        self.amount_pattern = dc.get("amount_pattern")
        self.timeout = dc.get("timeout", 30)

    async def claim(self, wallet_address: str) -> ClaimResult:
        if not self.command_template:
            return self._failure(ClaimStatus.UNKNOWN_ERROR, "No command configured for CLI driver")

        command = self.command_template.replace("{address}", wallet_address)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                executable=sys.executable if command.startswith("python") else None,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)

            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""

            if proc.returncode != 0:
                return self._failure(
                    ClaimStatus.UNKNOWN_ERROR,
                    f"Exit code {proc.returncode}: {stderr_str or stdout_str}",
                )

            if self.success_pattern:
                import re

                if not re.search(self.success_pattern, stdout_str):
                    return self._failure(
                        ClaimStatus.UNKNOWN_ERROR,
                        f"Success pattern not found in output: {stdout_str[:200]}",
                    )

            amount = None
            if self.amount_pattern:
                import re

                match = re.search(self.amount_pattern, stdout_str)
                if match:
                    try:
                        amount = float(match.group(1))
                    except (ValueError, IndexError):
                        pass

            return self._success(amount=amount)

        except TimeoutError:
            return self._failure(
                ClaimStatus.NETWORK_ERROR,
                f"Command timed out after {self.timeout}s",
            )
        except OSError as e:
            return self._failure(ClaimStatus.UNKNOWN_ERROR, str(e))

    async def health_check(self) -> bool:
        """CLI drivers are healthy if the command binary exists."""
        if not self.command_template:
            return False
        binary = self.command_template.split()[0]
        try:
            proc = await asyncio.create_subprocess_exec(
                "which",
                binary,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            return proc.returncode == 0
        except OSError:
            return False
