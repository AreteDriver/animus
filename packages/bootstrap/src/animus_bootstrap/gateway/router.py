"""Message router — ties cognitive backend, sessions, and channels together."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from animus_bootstrap.gateway.cognitive import CognitiveBackend
from animus_bootstrap.gateway.middleware.auth import GatewayAuthMiddleware
from animus_bootstrap.gateway.middleware.logging import MessageLogger
from animus_bootstrap.gateway.middleware.ratelimit import RateLimiter
from animus_bootstrap.gateway.models import GatewayMessage, GatewayResponse
from animus_bootstrap.gateway.session import SessionManager

logger = logging.getLogger(__name__)


class MessageRouter:
    """Central router: receives messages, resolves sessions, generates responses."""

    def __init__(
        self,
        cognitive: CognitiveBackend,
        session_manager: SessionManager,
        *,
        auth: GatewayAuthMiddleware | None = None,
        rate_limiter: RateLimiter | None = None,
        message_logger: MessageLogger | None = None,
    ) -> None:
        self._cognitive = cognitive
        self._session_manager = session_manager
        self._channels: dict[str, Any] = {}
        self._running = False
        # Optional gateway middleware. All default to None → open mode, so
        # existing behaviour is unchanged when they are not supplied.
        self._auth = auth
        self._rate_limiter = rate_limiter
        self._message_logger = message_logger

    def _check_inbound(self, message: GatewayMessage) -> GatewayResponse | None:
        """Apply inbound middleware. Returns a short-circuit response if blocked.

        Logs the inbound message, enforces the allowlist, then the rate limit.
        Returns ``None`` when the message is permitted to proceed.
        """
        if self._message_logger is not None:
            try:
                self._message_logger.log_inbound(message)
            except Exception:
                logger.exception("Failed to log inbound message")

        if self._auth is not None and not self._auth.is_allowed(message):
            logger.warning(
                "Blocked unauthorized message from (%s, %s)",
                message.channel,
                message.sender_id,
            )
            return GatewayResponse(
                text="You are not authorized to use this assistant.",
                channel=message.channel,
            )

        if self._rate_limiter is not None and not self._rate_limiter.check(message.sender_id):
            return GatewayResponse(
                text="You're sending messages too quickly. Please wait a moment.",
                channel=message.channel,
            )

        return None

    def _log_outbound(self, response: GatewayResponse, channel: str) -> None:
        """Log an outbound response when a message logger is configured."""
        if self._message_logger is not None:
            try:
                self._message_logger.log_outbound(response, channel)
            except Exception:
                logger.exception("Failed to log outbound message")

    async def handle_message(self, message: GatewayMessage) -> GatewayResponse:
        """Process an incoming message through the full pipeline.

        0. Inbound middleware (log, authorize, rate-limit)
        1. Get or create session
        2. Store user message
        3. Build context from history
        4. Generate LLM response
        5. Store assistant message
        6. Return GatewayResponse
        """
        # 0. Inbound middleware — may short-circuit
        blocked = self._check_inbound(message)
        if blocked is not None:
            return blocked

        # 1. Session lookup / creation
        session = await self._session_manager.get_or_create_session(message)

        # 2. Persist user message
        await self._session_manager.add_message(session, message)

        # 3. Build conversation context
        context = await self._session_manager.get_context(session)

        # 4. Generate response
        response_text = await self._cognitive.generate_response(context)

        # 5. Create and persist assistant message
        assistant_msg = GatewayMessage(
            id=str(uuid.uuid4()),
            channel=message.channel,
            channel_message_id="",
            sender_id="animus",
            sender_name="Animus",
            text=response_text,
            timestamp=datetime.now(UTC),
            role="assistant",
        )
        await self._session_manager.add_message(session, assistant_msg)

        # 6. Return response
        response = GatewayResponse(
            text=response_text,
            channel=message.channel,
        )
        self._log_outbound(response, message.channel)
        return response

    async def broadcast(self, text: str, channels: list[str] | None = None) -> None:
        """Send a message to all (or specified) connected channels."""
        from animus_bootstrap.gateway.models import GatewayResponse

        targets = channels if channels is not None else list(self._channels.keys())
        response = GatewayResponse(text=text, channel="broadcast")
        for name in targets:
            adapter = self._channels.get(name)
            if adapter is None:
                logger.warning("broadcast: channel %r not registered", name)
                continue
            try:
                await adapter.send_message(response)
            except Exception:
                logger.exception("broadcast failed for channel %r", name)

    def register_channel(self, name: str, adapter: Any) -> None:
        """Register a channel adapter by name."""
        self._channels[name] = adapter

    def unregister_channel(self, name: str) -> None:
        """Remove a channel adapter by name."""
        self._channels.pop(name, None)

    @property
    def channels(self) -> dict[str, Any]:
        """Return a copy of registered channels."""
        return dict(self._channels)
