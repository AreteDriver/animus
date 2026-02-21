"""Signal channel adapter via signal-cli subprocess."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import Callable, Coroutine
from typing import Any

from animus_bootstrap.gateway.models import (
    ChannelHealth,
    GatewayMessage,
    GatewayResponse,
    create_message,
)

logger = logging.getLogger(__name__)

MessageCallback = Callable[[GatewayMessage], Coroutine[Any, Any, None]]


class SignalAdapter:
    """Channel adapter for Signal via signal-cli subprocess.

    This adapter does *not* require a Python library — it wraps the
    ``signal-cli`` command-line tool which must be installed on the system.
    """

    name = "signal"

    def __init__(self, phone_number: str, signal_cli_path: str = "signal-cli") -> None:
        self.is_connected = False
        self._phone = phone_number
        self._cli = signal_cli_path
        self._callback: MessageCallback | None = None
        self._receive_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        """Verify signal-cli is available and start the receive loop."""
        resolved = shutil.which(self._cli)
        if not resolved:
            raise FileNotFoundError(
                f"signal-cli not found at '{self._cli}'. "
                "Install it from https://github.com/AsamK/signal-cli"
            )

        self.is_connected = True
        loop = asyncio.get_running_loop()
        self._receive_task = loop.create_task(self._receive_loop())
        logger.info("Signal adapter connected for %s", self._phone)

    async def disconnect(self) -> None:
        """Cancel the background receive loop."""
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        self.is_connected = False
        logger.info("Signal adapter disconnected")

    async def send_message(self, response: GatewayResponse) -> str:
        """Send a message via signal-cli.

        The ``recipient`` must be present in ``response.metadata``.
        Returns an empty string (signal-cli does not return message IDs).
        """
        recipient = response.metadata.get("recipient")
        if not recipient:
            raise ValueError("response.metadata must contain 'recipient'")

        proc = await asyncio.create_subprocess_exec(
            self._cli,
            "-u",
            self._phone,
            "send",
            "-m",
            response.text,
            recipient,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            raise RuntimeError(f"signal-cli send failed: {error_msg}")

        return ""

    async def _receive_loop(self) -> None:
        """Continuously receive messages via ``signal-cli receive --json``."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._cli,
                "-u",
                self._phone,
                "receive",
                "--json",
                "--timeout",
                "5",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            if proc.stdout is None:
                return

            while True:
                line = await proc.stdout.readline()
                if not line:
                    # Process exited — restart after a delay
                    await asyncio.sleep(2)
                    break

                try:
                    data = json.loads(line.decode())
                    await self._process_signal_message(data)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON line from signal-cli: %s", line.decode().strip())
                except Exception:
                    logger.exception("Error processing Signal message")

        except asyncio.CancelledError:
            if proc and proc.returncode is None:
                proc.terminate()
            raise

    async def _process_signal_message(self, data: dict[str, Any]) -> None:
        """Parse a JSON envelope from signal-cli and dispatch."""
        envelope = data.get("envelope", {})
        data_msg = envelope.get("dataMessage", {})
        text = data_msg.get("message", "")

        if not text:
            return

        source = envelope.get("source", "")
        source_name = envelope.get("sourceName", source)
        timestamp = str(envelope.get("timestamp", ""))

        gw_msg = create_message(
            channel="signal",
            sender_id=source,
            sender_name=source_name,
            text=text,
            channel_message_id=timestamp,
            metadata={
                "recipient": source,
                "group_id": data_msg.get("groupInfo", {}).get("groupId", ""),
            },
        )

        if self._callback:
            try:
                await self._callback(gw_msg)
            except Exception:
                logger.exception("Error in Signal message callback")

    async def on_message(self, callback: MessageCallback) -> None:
        """Register a callback to receive incoming messages."""
        self._callback = callback

    async def health_check(self) -> ChannelHealth:
        """Check signal-cli availability by running ``--version``."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._cli,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            if proc.returncode == 0:
                stdout.decode().strip()
                return ChannelHealth(
                    channel="signal",
                    connected=self.is_connected,
                    error=None,
                )
            return ChannelHealth(
                channel="signal",
                connected=False,
                error=f"signal-cli exited with code {proc.returncode}",
            )
        except FileNotFoundError:
            return ChannelHealth(
                channel="signal",
                connected=False,
                error="signal-cli not found",
            )
        except Exception as exc:
            return ChannelHealth(
                channel="signal",
                connected=False,
                error=str(exc),
            )
