"""Operational events router — event history and real-time SSE stream."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

router = APIRouter()

# SSE subscriber queues
_sse_subscribers: list[asyncio.Queue[dict[str, str]]] = []


def _get_event_ledger(request: Request) -> object | None:
    """Safely retrieve the event ledger from runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return None
    return getattr(runtime, "event_ledger", None)


def _notify_sse(event: dict[str, Any]) -> None:
    """Push an event to all SSE subscribers."""
    msg = {"event": "ledger", "data": json.dumps(event)}
    dead: list[asyncio.Queue[dict[str, str]]] = []
    for q in _sse_subscribers:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _sse_subscribers.remove(q)


@router.get("/events")
async def events_page(request: Request) -> object:
    """Render the operational events page."""
    templates = request.app.state.templates
    ledger = _get_event_ledger(request)

    events: list[dict] = []
    stats: dict = {}
    if ledger is not None:
        events = ledger.query(limit=100)
        stats = ledger.get_stats()

    return templates.TemplateResponse(
        request,
        "events.html",
        {"events": events, "stats": stats},
    )


@router.get("/events/feed")
async def events_feed(request: Request) -> object:
    """Return recent events as an HTML fragment (for HTMX polling)."""
    templates = request.app.state.templates
    ledger = _get_event_ledger(request)

    events: list[dict] = []
    if ledger is not None:
        events = ledger.query(limit=20)

    return templates.TemplateResponse(
        request,
        "fragments/events_feed.html",
        {"events": events},
    )


@router.get("/events/stream")
async def events_stream(request: Request) -> StreamingResponse:
    """Server-Sent Events stream of live operational events."""
    queue: asyncio.Queue[dict[str, str]] = asyncio.Queue(maxsize=100)
    _sse_subscribers.append(queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                msg = await queue.get()
                yield f"event: {msg['event']}\ndata: {msg['data']}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in _sse_subscribers:
                _sse_subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
