"""Message router — ties cognitive backend, sessions, and channels together."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from animus_bootstrap.gateway.cognitive import CognitiveBackend
from animus_bootstrap.gateway.models import GatewayMessage, GatewayResponse
from animus_bootstrap.gateway.session import SessionManager

logger = logging.getLogger(__name__)


class MessageRouter:
    """Central router: receives messages, resolves sessions, generates responses."""

    def __init__(
        self,
        cognitive: CognitiveBackend,
        session_manager: SessionManager,
    ) -> None:
        self._cognitive = cognitive
        self._session_manager = session_manager
        self._channels: dict[str, Any] = {}
        self._running = False

    async def handle_message(self, message: GatewayMessage) -> GatewayResponse:
        """Process an incoming message through the full pipeline.

        1. Get or create session
        2. Store user message
        3. Build context from history
        4. Generate LLM response
        5. Store assistant message
        6. Return GatewayResponse
        """
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
        return GatewayResponse(
            text=response_text,
            channel=message.channel,
        )

    async def broadcast(self, text: str, channels: list[str] | None = None) -> None:
        """Send a message to all (or specified) connected channels."""
        targets = channels if channels is not None else list(self._channels.keys())
        for name in targets:
            adapter = self._channels.get(name)
            if adapter is None:
                logger.warning("broadcast: channel %r not registered", name)
                continue
            try:
                await adapter.send(text)
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
