"""Condition evaluation — determine if all conditions pass."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from animus_bootstrap.gateway.models import GatewayMessage
from animus_bootstrap.intelligence.automations.models import Condition


def evaluate_conditions(
    conditions: list[Condition],
    message: GatewayMessage | None = None,
) -> bool:
    """Evaluate ALL conditions (AND logic). Empty list returns True."""
    return all(evaluate_condition(c, message) for c in conditions)


def evaluate_condition(
    condition: Condition,
    message: GatewayMessage | None = None,
) -> bool:
    """Evaluate a single condition."""
    if condition.type == "contains":
        return _check_contains(condition, message)
    if condition.type == "from_channel":
        return _check_from_channel(condition, message)
    if condition.type == "regex":
        return _check_regex(condition, message)
    if condition.type == "sender_is":
        return _check_sender_is(condition, message)
    if condition.type == "time_range":
        return _check_time_range(condition, message)
    return False


def _check_contains(condition: Condition, message: GatewayMessage | None) -> bool:
    """Check if message text contains a keyword (case-insensitive).

    params: text (str)
    """
    if message is None:
        return False
    text = condition.params.get("text", "")
    return text.lower() in message.text.lower()


def _check_from_channel(condition: Condition, message: GatewayMessage | None) -> bool:
    """Check if message is from a specific channel.

    params: channel (str)
    """
    if message is None:
        return False
    expected = condition.params.get("channel", "")
    return message.channel == expected


def _check_regex(condition: Condition, message: GatewayMessage | None) -> bool:
    """Check if message text matches a regex.

    params: pattern (str)
    """
    if message is None:
        return False
    pattern = condition.params.get("pattern", "")
    return bool(re.search(pattern, message.text))


def _check_sender_is(condition: Condition, message: GatewayMessage | None) -> bool:
    """Check if sender matches.

    params: sender_id (str)
    """
    if message is None:
        return False
    expected = condition.params.get("sender_id", "")
    return message.sender_id == expected


def _check_time_range(condition: Condition, message: GatewayMessage | None = None) -> bool:
    """Check if current time is within range.

    params: start (str HH:MM), end (str HH:MM)
    """
    start_str = condition.params.get("start", "00:00")
    end_str = condition.params.get("end", "23:59")

    now = datetime.now(UTC)
    start_parts = start_str.split(":")
    end_parts = end_str.split(":")

    start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
    end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])
    now_minutes = now.hour * 60 + now.minute

    if start_minutes <= end_minutes:
        return start_minutes <= now_minutes <= end_minutes
    # Wraps midnight: e.g. 22:00 - 06:00
    return now_minutes >= start_minutes or now_minutes <= end_minutes
