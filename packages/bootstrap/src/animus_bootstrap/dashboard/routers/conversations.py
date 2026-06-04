"""Conversations page router — message feed and history."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory message store — replaced by SessionManager once the DB layer lands.
# Each entry: {"id": str, "channel": str, "sender": str, "text": str, "timestamp": str}
_message_store: list[dict[str, str]] = []


def get_message_store() -> list[dict[str, str]]:
    """Return the module-level message store (test-patchable seam)."""
    return _message_store


@router.get("/conversations")
async def conversations_page(request: Request) -> object:
    """Render the conversations page with the recent message feed."""
    templates = request.app.state.templates
    messages = get_message_store()

    # Newest first for display
    recent = list(reversed(messages[-50:]))

    return templates.TemplateResponse(
        request,
        "conversations.html",
        {
            "messages": recent,
        },
    )


@router.get("/conversations/messages")
async def get_messages(limit: int = 50) -> JSONResponse:
    """Return recent messages as JSON (for HTMX polling).

    Args:
        limit: Maximum number of messages to return.
    """
    messages = get_message_store()
    recent = list(reversed(messages[-limit:]))
    return JSONResponse(content=recent)


@router.get("/api/conversations/history")
async def get_history(request: Request, limit: int = 50) -> JSONResponse:
    """Return persisted conversation history for the PWA.

    Reads from the runtime's :class:`SessionManager` (the durable
    ``gateway_messages`` table) and returns items in chronological order
    shaped to match the PWA's ``WSMessage`` type. Falls back to an empty
    list when the runtime/session manager is unavailable.
    """
    runtime = getattr(request.app.state, "runtime", None)
    session_manager = getattr(runtime, "session_manager", None) if runtime else None
    if session_manager is None:
        return JSONResponse(content=[])

    limit = max(1, min(limit, 200))
    messages = await session_manager.get_recent_messages(limit)

    # get_recent_messages returns newest-first; reverse for display order.
    items = [
        {
            "id": msg.id,
            "channel": msg.channel,
            "text": msg.text,
            "timestamp": msg.timestamp.isoformat(),
            "sender": "animus" if msg.role == "assistant" else msg.sender_name,
            "role": msg.role,
            "metadata": msg.metadata,
        }
        for msg in reversed(messages)
    ]
    return JSONResponse(content=items)
