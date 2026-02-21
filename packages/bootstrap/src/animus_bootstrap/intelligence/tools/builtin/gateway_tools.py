"""Gateway tools — send messages through gateway channels via LLM tool use."""

from __future__ import annotations

import logging

from animus_bootstrap.intelligence.tools.executor import ToolDefinition

logger = logging.getLogger(__name__)

# In-memory message store for fallback when no router is wired
_sent_messages: list[dict[str, str]] = []

# Live router reference — set at runtime
_router = None


def set_gateway_router(router: object) -> None:
    """Wire the live MessageRouter for gateway tools."""
    global _router  # noqa: PLW0603
    _router = router


async def _send_message(channel: str, text: str) -> str:
    """Send a message to a gateway channel.

    Delegates to the live router if available, otherwise stores in
    the in-memory fallback list.
    """
    if _router is not None:
        try:
            await _router.broadcast(text, [channel])
            logger.info("Sent message to channel '%s' via router", channel)
            return f"Message sent to channel '{channel}': {text[:100]}"
        except Exception as exc:
            logger.warning("Router broadcast failed, using fallback: %s", exc)

    _sent_messages.append({"channel": channel, "text": text})
    return f"Message queued for channel '{channel}': {text[:100]}"


def get_sent_messages() -> list[dict[str, str]]:
    """Return all stored messages (for testing / inspection)."""
    return list(_sent_messages)


def clear_sent_messages() -> None:
    """Clear the stored messages."""
    _sent_messages.clear()


def get_gateway_tools() -> list[ToolDefinition]:
    """Return gateway tool definitions."""
    return [
        ToolDefinition(
            name="send_message",
            description=(
                "Send a text message to a gateway channel (e.g. 'telegram', 'discord', 'webchat')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Target channel name.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text to send.",
                    },
                },
                "required": ["channel", "text"],
            },
            handler=_send_message,
            category="gateway",
        ),
    ]
