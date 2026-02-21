"""Trigger evaluation — determine if a trigger matches the current context."""

from __future__ import annotations

import re

from animus_bootstrap.gateway.models import GatewayMessage
from animus_bootstrap.intelligence.automations.models import TriggerConfig


def evaluate_trigger(
    trigger: TriggerConfig,
    message: GatewayMessage | None = None,
    event: dict | None = None,
) -> bool:
    """Evaluate whether a trigger matches the current context."""
    if trigger.type == "message":
        if message is None:
            return False
        return _match_message_trigger(trigger, message)

    if trigger.type == "event":
        if event is None:
            return False
        return _match_event_trigger(trigger, event)

    if trigger.type == "webhook":
        if event is None:
            return False
        return _match_webhook_trigger(trigger, event)

    # schedule triggers are handled by the engine directly, not evaluated here
    return False


def _match_message_trigger(trigger: TriggerConfig, message: GatewayMessage) -> bool:
    """Check if message matches trigger params.

    params can have: keywords (list[str]), regex (str), sender_id (str), channel (str)
    """
    params = trigger.params

    # If no params, any message matches
    if not params:
        return True

    # Check keywords (any keyword present = match)
    keywords: list[str] = params.get("keywords", [])
    if keywords:
        text_lower = message.text.lower()
        if not any(kw.lower() in text_lower for kw in keywords):
            return False

    # Check regex
    pattern: str | None = params.get("regex")
    if pattern is not None:
        if not re.search(pattern, message.text):
            return False

    # Check sender_id
    sender_id: str | None = params.get("sender_id")
    if sender_id is not None and message.sender_id != sender_id:
        return False

    # Check channel
    channel: str | None = params.get("channel")
    if channel is not None and message.channel != channel:
        return False

    return True


def _match_event_trigger(trigger: TriggerConfig, event: dict) -> bool:
    """Check if event matches trigger params.

    params can have: event_type (str)
    """
    expected_type = trigger.params.get("event_type")
    if expected_type is None:
        return True
    return event.get("type") == expected_type


def _match_webhook_trigger(trigger: TriggerConfig, event: dict) -> bool:
    """Check if webhook payload matches. Always True if trigger type matches."""
    return True
