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
        "conversations.html",
        {
            "request": request,
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
