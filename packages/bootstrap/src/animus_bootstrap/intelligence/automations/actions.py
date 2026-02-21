"""Action execution — run actions when an automation rule fires."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from animus_bootstrap.intelligence.automations.models import ActionConfig

logger = logging.getLogger(__name__)


async def execute_action(action: ActionConfig, context: dict[str, Any]) -> str:
    """Execute a single action. Returns a result description string."""
    if action.type == "reply":
        return await _action_reply(action, context)
    if action.type == "forward":
        return await _action_forward(action, context)
    if action.type == "webhook":
        return await _action_webhook(action, context)
    if action.type == "store_memory":
        return await _action_store_memory(action, context)
    if action.type == "run_tool":
        return await _action_run_tool(action, context)
    return f"Unknown action type: {action.type}"


async def _action_reply(action: ActionConfig, context: dict[str, Any]) -> str:
    """Reply on the originating channel.

    params: text (str)
    context needs: channel (str), send_callback
    """
    text = action.params.get("text", "")
    channel = context.get("channel", "unknown")
    send_callback = context.get("send_callback")

    if send_callback is not None:
        await send_callback(channel, text)

    return f"Replied on {channel}"


async def _action_forward(action: ActionConfig, context: dict[str, Any]) -> str:
    """Forward message to another channel.

    params: target_channel (str)
    context needs: message_text (str), send_callback
    """
    target = action.params.get("target_channel", "")
    message_text = context.get("message_text", "")
    send_callback = context.get("send_callback")

    if send_callback is not None:
        await send_callback(target, message_text)

    return f"Forwarded to {target}"


async def _action_webhook(action: ActionConfig, context: dict[str, Any]) -> str:
    """POST to an external URL.

    params: url (str), payload (dict, optional)
    """
    url = action.params.get("url", "")
    payload = action.params.get("payload", {})

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload)
        return f"Webhook POST {url} -> {resp.status_code}"
    except httpx.HTTPError as exc:
        logger.warning("Webhook failed: %s", exc)
        return f"Webhook failed: {exc}"


async def _action_store_memory(action: ActionConfig, context: dict[str, Any]) -> str:
    """Store something in memory.

    params: content (str), memory_type (str, default "semantic")
    context needs: memory_manager (optional)
    """
    content = action.params.get("content", "")
    memory_type = action.params.get("memory_type", "semantic")
    memory_manager = context.get("memory_manager")

    if memory_manager is not None:
        await memory_manager.store_fact(content, "automation_stored", memory_type)
        return f"Stored memory ({memory_type}): {content[:50]}"

    return f"No memory_manager in context — would store: {content[:50]}"


async def _action_run_tool(action: ActionConfig, context: dict[str, Any]) -> str:
    """Execute a registered tool.

    params: tool_name (str), arguments (dict)
    context needs: tool_executor (optional)
    """
    tool_name = action.params.get("tool_name", "")
    arguments = action.params.get("arguments", {})
    tool_executor = context.get("tool_executor")

    if tool_executor is not None:
        result = await tool_executor(tool_name, arguments)
        return f"Tool {tool_name} returned: {result}"

    return f"No tool_executor in context — would run: {tool_name}"
