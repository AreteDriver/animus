"""Gateway tools — message sending stubs for LLM tool use."""

from __future__ import annotations

import logging

from animus_bootstrap.intelligence.tools.executor import ToolDefinition

logger = logging.getLogger(__name__)

# In-memory message store for the stub implementation
_sent_messages: list[dict[str, str]] = []


async def _send_message(channel: str, text: str) -> str:
    """Stub: store a message for the gateway to send."""
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
            description="Send a text message to a gateway channel.",
            parameters={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Target channel name (e.g. 'telegram', 'discord').",
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
