"""WhatsApp Cloud API channel adapter (Meta Business Platform)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

import httpx

from animus_bootstrap.gateway.models import (
    ChannelHealth,
    GatewayMessage,
    GatewayResponse,
    create_message,
)

logger = logging.getLogger(__name__)

MessageCallback = Callable[[GatewayMessage], Coroutine[Any, Any, None]]

_API_BASE = "https://graph.facebook.com/v18.0"


class WhatsAppAdapter:
    """Channel adapter for WhatsApp via the Cloud API (Meta Business).

    This adapter uses ``httpx`` (already a project dependency) rather than
    a dedicated WhatsApp library, so no optional-dependency guard is needed.
    """

    name = "whatsapp"

    def __init__(
        self,
        phone_number_id: str,
        access_token: str,
        verify_token: str = "",
    ) -> None:
        self.is_connected = False
        self._phone_id = phone_number_id
        self._access_token = access_token
        self._verify_token = verify_token
        self._callback: MessageCallback | None = None
        self._http: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Verify credentials by calling the WhatsApp Business API."""
        self._http = httpx.AsyncClient(
            base_url=_API_BASE,
            headers={"Authorization": f"Bearer {self._access_token}"},
            timeout=30.0,
        )

        # Verify the phone number ID is valid
        resp = await self._http.get(f"/{self._phone_id}")
        if resp.status_code != 200:
            await self._http.aclose()
            self._http = None
            raise ConnectionError(f"WhatsApp API returned {resp.status_code}: {resp.text}")

        self.is_connected = True
        logger.info("WhatsApp adapter connected for phone_id %s", self._phone_id)

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self._http:
            await self._http.aclose()
            self._http = None
        self.is_connected = False
        logger.info("WhatsApp adapter disconnected")

    async def send_message(self, response: GatewayResponse) -> str:
        """Send a text message via the WhatsApp Cloud API.

        The ``to`` phone number must be present in ``response.metadata``.
        Returns the WhatsApp message ID (``wamid``).
        """
        if not self._http:
            raise RuntimeError("WhatsApp adapter is not connected")

        to_number = response.metadata.get("to")
        if not to_number:
            raise ValueError("response.metadata must contain 'to' (recipient phone number)")

        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "text",
            "text": {"body": response.text},
        }

        resp = await self._http.post(
            f"/{self._phone_id}/messages",
            json=payload,
        )

        if resp.status_code not in (200, 201):
            raise RuntimeError(f"WhatsApp send failed ({resp.status_code}): {resp.text}")

        data = resp.json()
        messages = data.get("messages", [])
        return messages[0]["id"] if messages else ""

    async def handle_webhook(self, data: dict[str, Any]) -> None:
        """Parse an incoming webhook payload from WhatsApp and dispatch.

        This method is intended to be called from a FastAPI/webhook endpoint.
        """
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                for msg in value.get("messages", []):
                    if msg.get("type") != "text":
                        continue

                    contact = next(
                        (c for c in value.get("contacts", []) if c.get("wa_id") == msg.get("from")),
                        {},
                    )

                    gw_msg = create_message(
                        channel="whatsapp",
                        sender_id=msg.get("from", ""),
                        sender_name=contact.get("profile", {}).get("name", msg.get("from", "")),
                        text=msg.get("text", {}).get("body", ""),
                        channel_message_id=msg.get("id", ""),
                        metadata={
                            "to": msg.get("from", ""),
                            "phone_number_id": value.get("metadata", {}).get("phone_number_id", ""),
                        },
                    )

                    if self._callback:
                        try:
                            await self._callback(gw_msg)
                        except Exception:
                            logger.exception("Error in WhatsApp message callback")

    async def on_message(self, callback: MessageCallback) -> None:
        """Register a callback to receive incoming messages."""
        self._callback = callback

    async def health_check(self) -> ChannelHealth:
        """Check the WhatsApp API by fetching the phone number details."""
        if not self._http:
            return ChannelHealth(
                channel="whatsapp",
                connected=False,
                error="Not connected",
            )
        try:
            resp = await self._http.get(f"/{self._phone_id}")
            connected = resp.status_code == 200
            return ChannelHealth(
                channel="whatsapp",
                connected=connected,
                error=None if connected else f"API returned {resp.status_code}",
            )
        except Exception as exc:
            return ChannelHealth(
                channel="whatsapp",
                connected=False,
                error=str(exc),
            )
