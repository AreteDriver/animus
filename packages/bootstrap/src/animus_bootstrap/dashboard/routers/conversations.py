"""Conversations page router — message feed and history."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from animus_bootstrap.gateway.models import GatewayResponse, create_message

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


@router.post("/api/conversations/messages")
async def post_message(request: Request) -> JSONResponse:
    """Receive a message from the PWA (REST fallback), route it, and return the reply.

    Expects JSON body: {"text": str}
    Returns JSON: {"text": str}
    """
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return JSONResponse(content={"text": "Message text is required."}, status_code=400)

    store = get_message_store()
    store.append(
        {
            "channel": "pwa",
            "sender": "User",
            "text": text,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )

    runtime = getattr(request.app.state, "runtime", None)
    if runtime is not None and runtime.started and runtime.router is not None:
        msg = create_message(
            channel="pwa",
            sender_id="pwa-user",
            sender_name="User",
            text=text,
        )
        try:
            response: GatewayResponse = await runtime.router.handle_message(msg)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                logger.warning("Backend auth failed (%s) — PWA message rejected", status)
                response = GatewayResponse(
                    text=(
                        "Backend authentication failed. "
                        "Set ANTHROPIC_API_KEY or start Ollama (localhost:11434). "
                        "See docs/getting-started/quickstart.md"
                    ),
                    channel="pwa",
                )
            elif status == 429:
                logger.warning("Backend rate-limited (%s) — PWA message rejected", status)
                response = GatewayResponse(
                    text="Rate limit hit. Retry in a moment.",
                    channel="pwa",
                )
            else:
                logger.warning("Backend HTTP error (%s) — PWA message rejected", status)
                response = GatewayResponse(
                    text=f"Backend error ({status}). Check server logs.",
                    channel="pwa",
                )
        except Exception:
            logger.exception("Router failed to handle PWA message")
            response = GatewayResponse(
                text="Internal error. The team has been notified.", channel="pwa"
            )
    else:
        response = GatewayResponse(
            text="Animus is currently offline — message queued.", channel="pwa"
        )

    store.append(
        {
            "channel": response.channel,
            "sender": "Animus",
            "text": response.text,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )

    return JSONResponse(content={"text": response.text})


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
