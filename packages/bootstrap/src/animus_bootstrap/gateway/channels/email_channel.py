"""Email channel adapter via IMAP/SMTP."""

from __future__ import annotations

import asyncio
import email
import email.mime.text
import imaplib
import logging
import smtplib
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from animus_bootstrap.gateway.models import (
    ChannelHealth,
    GatewayMessage,
    GatewayResponse,
    create_message,
)

try:
    import aiosmtplib

    HAS_AIOSMTPLIB = True
except ImportError:
    HAS_AIOSMTPLIB = False

logger = logging.getLogger(__name__)

MessageCallback = Callable[[GatewayMessage], Coroutine[Any, Any, None]]


class EmailAdapter:
    """Channel adapter for email via IMAP (receive) and SMTP (send).

    Uses ``aiosmtplib`` for async sending if available, otherwise falls back
    to ``smtplib`` via ``asyncio.to_thread``.
    """

    name = "email"

    def __init__(
        self,
        imap_host: str,
        smtp_host: str,
        username: str,
        password: str,
        *,
        poll_interval: int = 60,
        imap_port: int = 993,
        smtp_port: int = 587,
    ) -> None:
        self.is_connected = False
        self._imap_host = imap_host
        self._smtp_host = smtp_host
        self._username = username
        self._password = password
        self._poll_interval = poll_interval
        self._imap_port = imap_port
        self._smtp_port = smtp_port
        self._callback: MessageCallback | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._last_uid: str = "0"

    async def connect(self) -> None:
        """Test IMAP connection and start the polling loop."""
        # Verify credentials via a quick IMAP login
        await asyncio.to_thread(self._test_imap)
        self.is_connected = True

        loop = asyncio.get_running_loop()
        self._poll_task = loop.create_task(self._poll_loop())
        logger.info("Email adapter connected (%s)", self._username)

    def _test_imap(self) -> None:
        """Synchronous IMAP connection test."""
        conn = imaplib.IMAP4_SSL(self._imap_host, self._imap_port)
        conn.login(self._username, self._password)
        conn.logout()

    async def disconnect(self) -> None:
        """Cancel the polling loop."""
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                logger.debug("Email poll task cancelled")
        self.is_connected = False
        logger.info("Email adapter disconnected")

    async def send_message(self, response: GatewayResponse) -> str:
        """Send an email via SMTP.

        ``response.metadata`` must contain:
        - ``to``: recipient email address
        - ``subject`` (optional): email subject line (default: "Re: Animus")
        """
        to_addr = response.metadata.get("to")
        if not to_addr:
            raise ValueError("response.metadata must contain 'to' (recipient email)")

        subject = response.metadata.get("subject", "Re: Animus")
        msg = email.mime.text.MIMEText(response.text)
        msg["Subject"] = subject
        msg["From"] = self._username
        msg["To"] = to_addr
        msg["Message-ID"] = f"<{uuid.uuid4()}@animus>"

        if HAS_AIOSMTPLIB:
            await aiosmtplib.send(
                msg,
                hostname=self._smtp_host,
                port=self._smtp_port,
                username=self._username,
                password=self._password,
                start_tls=True,
            )
        else:
            await asyncio.to_thread(self._send_smtp_sync, msg)

        return msg["Message-ID"]

    def _send_smtp_sync(self, msg: email.mime.text.MIMEText) -> None:
        """Synchronous SMTP fallback."""
        with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
            server.starttls()
            server.login(self._username, self._password)
            server.send_message(msg)

    async def _poll_loop(self) -> None:
        """Poll IMAP for new messages on a configurable interval."""
        while True:
            try:
                await asyncio.sleep(self._poll_interval)
                new_messages = await asyncio.to_thread(self._fetch_new_messages)
                for gw_msg in new_messages:
                    if self._callback:
                        try:
                            await self._callback(gw_msg)
                        except Exception:
                            logger.exception("Error in email message callback")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error polling IMAP")

    def _fetch_new_messages(self) -> list[GatewayMessage]:
        """Fetch unseen messages from IMAP."""
        messages: list[GatewayMessage] = []
        try:
            conn = imaplib.IMAP4_SSL(self._imap_host, self._imap_port)
            conn.login(self._username, self._password)
            conn.select("INBOX")

            _, data = conn.search(None, "UNSEEN")
            if not data or not data[0]:
                conn.logout()
                return messages

            for num in data[0].split():
                _, msg_data = conn.fetch(num, "(RFC822)")
                if not msg_data or not msg_data[0]:
                    continue

                raw = msg_data[0]
                if isinstance(raw, tuple) and len(raw) >= 2:
                    raw_bytes = raw[1]
                else:
                    continue

                parsed = email.message_from_bytes(raw_bytes)

                # Extract plain text body
                body = ""
                if parsed.is_multipart():
                    for part in parsed.walk():
                        if part.get_content_type() == "text/plain":
                            payload = part.get_payload(decode=True)
                            if payload:
                                body = payload.decode(errors="replace")
                            break
                else:
                    payload = parsed.get_payload(decode=True)
                    if payload:
                        body = payload.decode(errors="replace")

                sender = parsed.get("From", "")
                subject = parsed.get("Subject", "")
                message_id = parsed.get("Message-ID", "")

                gw_msg = create_message(
                    channel="email",
                    sender_id=sender,
                    sender_name=sender,
                    text=body,
                    channel_message_id=message_id,
                    metadata={
                        "to": self._username,
                        "subject": subject,
                        "message_id": message_id,
                    },
                )
                messages.append(gw_msg)

            conn.logout()
        except (OSError, ConnectionError, ValueError) as exc:
            logger.warning("Failed to fetch IMAP messages: %s", exc)

        return messages

    async def on_message(self, callback: MessageCallback) -> None:
        """Register a callback to receive incoming messages."""
        self._callback = callback

    async def health_check(self) -> ChannelHealth:
        """Check IMAP connectivity."""
        try:
            await asyncio.to_thread(self._test_imap)
            return ChannelHealth(
                channel="email",
                connected=self.is_connected,
                error=None,
            )
        except Exception as exc:
            return ChannelHealth(
                channel="email",
                connected=False,
                error=str(exc),
            )
