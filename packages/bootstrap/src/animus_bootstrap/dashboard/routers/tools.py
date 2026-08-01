"""Tools management page router with approval flow."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Form, Request
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------
# Approval queue — bridges LLM-initiated tool calls to the dashboard
# ------------------------------------------------------------------

_pending_approvals: dict[str, dict[str, Any]] = {}

# SSE subscribers — each is an asyncio.Queue receiving event dicts
_sse_subscribers: list[asyncio.Queue[dict[str, str]]] = []
# Each entry: {
#   "tool_name": str,
#   "arguments": dict,
#   "event": asyncio.Event,
#   "approved": bool | None,
# }


def get_pending_approvals() -> dict[str, dict[str, Any]]:
    """Return pending approvals (for testing/inspection)."""
    return _pending_approvals


def clear_pending_approvals() -> None:
    """Clear all pending approvals."""
    for entry in _pending_approvals.values():
        entry["approved"] = False
        entry["event"].set()
    _pending_approvals.clear()


def _notify_sse(event_type: str, data: dict[str, Any]) -> None:
    """Push an event to all SSE subscribers."""
    msg = {"event": event_type, "data": json.dumps(data)}
    dead: list[asyncio.Queue[dict[str, str]]] = []
    for q in _sse_subscribers:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _sse_subscribers.remove(q)


async def dashboard_approval_callback(tool_name: str, arguments: dict[str, Any]) -> bool:
    """Approval callback that queues requests for the dashboard UI.

    When the ToolExecutor encounters an APPROVE-gated tool, this callback
    creates a pending approval entry and waits for the dashboard user to
    approve or deny it.
    """
    request_id = str(uuid.uuid4())[:8]
    event = asyncio.Event()

    _pending_approvals[request_id] = {
        "tool_name": tool_name,
        "arguments": arguments,
        "event": event,
        "approved": None,
    }

    logger.info("Approval requested for tool '%s' (id=%s)", tool_name, request_id)
    _notify_sse(
        "approval_requested",
        {"id": request_id, "tool_name": tool_name, "arguments": arguments},
    )

    try:
        await asyncio.wait_for(event.wait(), timeout=300.0)
    except TimeoutError:
        logger.warning("Approval for '%s' (id=%s) timed out", tool_name, request_id)
        _pending_approvals.pop(request_id, None)
        _notify_sse("approval_timeout", {"id": request_id, "tool_name": tool_name})
        return False

    entry = _pending_approvals.pop(request_id, {})
    approved = bool(entry.get("approved", False))
    _notify_sse(
        "approval_resolved",
        {"id": request_id, "tool_name": tool_name, "approved": approved},
    )
    return approved


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_runtime(request: Request) -> object | None:
    """Safely retrieve the runtime from app state."""
    return getattr(request.app.state, "runtime", None)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------


@router.get("/tools")
async def tools_page(request: Request) -> object:
    """Render the tools management page."""
    templates = request.app.state.templates

    tool_list: list[dict[str, str]] = []
    history: list[object] = []

    runtime = _get_runtime(request)
    if runtime is not None and getattr(runtime, "tool_executor", None) is not None:
        for tool in runtime.tool_executor.list_tools():
            tool_list.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "permission": tool.permission,
                }
            )
        history = runtime.tool_executor.get_history(limit=50)

    pending = [
        {"id": rid, "tool_name": entry["tool_name"], "arguments": entry["arguments"]}
        for rid, entry in _pending_approvals.items()
        if entry.get("approved") is None
    ]

    return templates.TemplateResponse(
        request,
        "tools.html",
        {
            "tools": tool_list,
            "history": history,
            "pending_approvals": pending,
        },
    )


@router.get("/tools/pending")
async def tools_pending(request: Request) -> object:
    """Return pending approvals as an HTML fragment (for HTMX polling)."""
    templates = request.app.state.templates
    pending = [
        {
            "id": rid,
            "tool_name": entry["tool_name"],
            "arguments": json.dumps(entry["arguments"], indent=2)[:200],
        }
        for rid, entry in _pending_approvals.items()
        if entry.get("approved") is None
    ]

    return templates.TemplateResponse(
        request,
        "fragments/tool_pending_table.html",
        {"pending": pending},
    )


@router.post("/tools/approve/{request_id}")
async def approve_tool(request_id: str, request: Request, decision: str = Form("deny")) -> object:
    """Approve or deny a pending tool execution."""
    templates = request.app.state.templates
    entry = _pending_approvals.get(request_id)
    if entry is None:
        return templates.TemplateResponse(
            request,
            "fragments/tool_approval_result.html",
            {"color": "text-animus-red", "tool_name": "", "status": "not found or expired"},
        )

    approved = decision == "approve"
    entry["approved"] = approved
    entry["event"].set()

    status = "approved" if approved else "denied"
    color = "text-animus-green" if approved else "text-animus-red"
    return templates.TemplateResponse(
        request,
        "fragments/tool_approval_result.html",
        {"color": color, "tool_name": entry["tool_name"], "status": status},
    )


@router.get("/tools/events")
async def tools_events(request: Request) -> StreamingResponse:
    """SSE stream for real-time approval notifications."""
    queue: asyncio.Queue[dict[str, str]] = asyncio.Queue(maxsize=50)
    _sse_subscribers.append(queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"event: {msg['event']}\ndata: {msg['data']}\n\n"
                except TimeoutError:
                    # Keepalive
                    yield ": keepalive\n\n"
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


@router.post("/tools/execute")
async def execute_tool(
    request: Request,
    tool_name: str = Form(""),
    arguments_json: str = Form("{}"),
) -> object:
    """Execute a tool directly from the dashboard (user-initiated, auto-approved)."""
    templates = request.app.state.templates
    runtime = _get_runtime(request)
    if runtime is None or getattr(runtime, "tool_executor", None) is None:
        return templates.TemplateResponse(
            request,
            "fragments/tool_execution_result.html",
            {"error": "No tool executor available."},
        )

    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError:
        return templates.TemplateResponse(
            request, "fragments/tool_execution_result.html", {"error": "Invalid JSON arguments."}
        )

    executor = runtime.tool_executor
    tool = executor.get_tool(tool_name)
    if tool is None:
        return templates.TemplateResponse(
            request, "fragments/tool_execution_result.html", {"error": f"Unknown tool: {tool_name}"}
        )

    result = await executor.execute(tool_name, arguments)

    return templates.TemplateResponse(
        request,
        "fragments/tool_execution_result.html",
        {
            "tool_name": tool_name,
            "status": "OK" if result.success else "FAIL",
            "color": "text-animus-green" if result.success else "text-animus-red",
            "duration_ms": f"{result.duration_ms:.0f}",
            "output": result.output[:1000],
        },
    )
