"""Tools management page router with approval flow."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
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
        "tools.html",
        {
            "request": request,
            "tools": tool_list,
            "history": history,
            "pending_approvals": pending,
        },
    )


@router.get("/tools/pending")
async def tools_pending(request: Request) -> HTMLResponse:
    """Return pending approvals as an HTML fragment (for HTMX polling)."""
    pending = [
        {"id": rid, "tool_name": entry["tool_name"], "arguments": entry["arguments"]}
        for rid, entry in _pending_approvals.items()
        if entry.get("approved") is None
    ]

    if not pending:
        return HTMLResponse('<p class="text-animus-muted text-sm">No pending approvals.</p>')

    rows = []
    for p in pending:
        args_str = json.dumps(p["arguments"], indent=2)[:200]
        rows.append(
            f'<tr class="border-b border-animus-border">'
            f'<td class="py-2 pr-4 text-animus-yellow font-bold">{p["tool_name"]}</td>'
            f'<td class="py-2 pr-4 text-animus-muted text-xs"><pre>{args_str}</pre></td>'
            f'<td class="py-2">'
            f'<form method="post" action="/tools/approve/{p["id"]}" class="inline">'
            f'<button type="submit" name="decision" value="approve" '
            f'class="px-3 py-1 bg-animus-green text-animus-bg rounded text-xs '
            f'hover:opacity-80 mr-2">Approve</button>'
            f'<button type="submit" name="decision" value="deny" '
            f'class="px-3 py-1 bg-animus-red text-white rounded text-xs '
            f'hover:opacity-80">Deny</button>'
            f"</form>"
            f"</td></tr>"
        )

    table = (
        '<table class="w-full text-sm">'
        "<thead>"
        '<tr class="border-b border-animus-border text-animus-muted">'
        '<th class="text-left py-2 pr-4">Tool</th>'
        '<th class="text-left py-2 pr-4">Arguments</th>'
        '<th class="text-left py-2">Action</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    return HTMLResponse(table)


@router.post("/tools/approve/{request_id}")
async def approve_tool(request_id: str, decision: str = Form("deny")) -> HTMLResponse:
    """Approve or deny a pending tool execution."""
    entry = _pending_approvals.get(request_id)
    if entry is None:
        return HTMLResponse(
            '<p class="text-animus-red text-sm">Approval request not found or expired.</p>'
        )

    approved = decision == "approve"
    entry["approved"] = approved
    entry["event"].set()

    status = "approved" if approved else "denied"
    color = "text-animus-green" if approved else "text-animus-red"
    return HTMLResponse(f"<p class=\"{color} text-sm\">Tool '{entry['tool_name']}' {status}.</p>")


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
) -> HTMLResponse:
    """Execute a tool directly from the dashboard (user-initiated, auto-approved)."""
    runtime = _get_runtime(request)
    if runtime is None or getattr(runtime, "tool_executor", None) is None:
        return HTMLResponse('<p class="text-animus-red text-sm">No tool executor available.</p>')

    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError:
        return HTMLResponse('<p class="text-animus-red text-sm">Invalid JSON arguments.</p>')

    executor = runtime.tool_executor
    tool = executor.get_tool(tool_name)
    if tool is None:
        return HTMLResponse(f'<p class="text-animus-red text-sm">Unknown tool: {tool_name}</p>')

    result = await executor.execute(tool_name, arguments)

    color = "text-animus-green" if result.success else "text-animus-red"
    output_escaped = result.output.replace("<", "&lt;").replace(">", "&gt;")[:1000]
    return HTMLResponse(
        f'<div class="bg-animus-surface border border-animus-border rounded p-4 mt-2">'
        f'<p class="{color} text-sm font-bold">'
        f"{tool_name}: {'OK' if result.success else 'FAIL'} "
        f"({result.duration_ms:.0f}ms)</p>"
        f'<pre class="text-animus-text text-xs mt-2 whitespace-pre-wrap">'
        f"{output_escaped}</pre></div>"
    )
