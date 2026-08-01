"""Operational controls router — pause/resume, kill task, clear memory, etc."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter()


def _get_runtime(request: Request) -> object | None:
    """Safely retrieve the runtime from app state."""
    return getattr(request.app.state, "runtime", None)


def _get_event_ledger(request: Request) -> object | None:
    """Safely retrieve the event ledger from runtime."""
    runtime = _get_runtime(request)
    if runtime is None:
        return None
    return getattr(runtime, "event_ledger", None)


def _record_event(request: Request, event_type: str, payload: dict) -> None:
    """Record an event to the ledger if available."""
    ledger = _get_event_ledger(request)
    if ledger is not None:
        ledger.record(event_type, "dashboard", payload)


# ── Runtime Pause / Resume ────────────────────────────────────────────────


@router.post("/runtime/pause")
async def runtime_pause(request: Request) -> JSONResponse:
    """Pause the runtime."""
    runtime = _get_runtime(request)
    if runtime is not None and hasattr(runtime, "pause"):
        runtime.pause()
    _record_event(request, "runtime_paused", {})
    return JSONResponse({"status": "paused"})


@router.post("/runtime/resume")
async def runtime_resume(request: Request) -> JSONResponse:
    """Resume a paused runtime."""
    runtime = _get_runtime(request)
    if runtime is not None and hasattr(runtime, "resume"):
        runtime.resume()
    _record_event(request, "runtime_resumed", {})
    return JSONResponse({"status": "resumed"})


# ── Task Kill ─────────────────────────────────────────────────────────────


@router.post("/tasks/{task_id}/kill")
async def task_kill(request: Request, task_id: str) -> object:
    """Kill (delete) a task."""
    templates = request.app.state.templates
    runtime = _get_runtime(request)
    store = getattr(runtime, "_task_store", None) if runtime else None
    if store is not None and hasattr(store, "delete"):
        store.delete(task_id)
    _record_event(request, "task_killed", {"task_id": task_id})
    return templates.TemplateResponse(
        request,
        "fragments/task_action_result.html",
        {"css_class": "text-animus-red", "message": "Killed"},
    )


# ── Memory Clear ──────────────────────────────────────────────────────────


@router.post("/memory/clear")
async def memory_clear(request: Request) -> JSONResponse:
    """Clear all memories from the active backend."""
    runtime = _get_runtime(request)
    memory_manager = getattr(runtime, "memory_manager", None) if runtime else None
    cleared = 0
    if memory_manager is not None:
        backend = getattr(memory_manager, "_backend", None)
        if backend is not None:
            # Try to clear via backend protocol or common methods
            clear_method = getattr(backend, "clear", None)
            if clear_method is not None:
                try:
                    import asyncio

                    if asyncio.iscoroutinefunction(clear_method):
                        await clear_method()
                    else:
                        clear_method()
                    cleared = 1
                except Exception:
                    pass
            # Fallback: delete all entries via search-then-delete
            if cleared == 0:
                try:
                    import asyncio

                    search_fn = getattr(backend, "search", None)
                    delete_fn = getattr(backend, "delete", None)
                    if search_fn and delete_fn:
                        if asyncio.iscoroutinefunction(search_fn):
                            all_memories = await search_fn("", memory_type="all", limit=10000)
                        else:
                            all_memories = search_fn("", memory_type="all", limit=10000)
                        for mem in all_memories:
                            mem_id = mem.get("id", "")
                            if mem_id:
                                if asyncio.iscoroutinefunction(delete_fn):
                                    await delete_fn(mem_id)
                                else:
                                    delete_fn(mem_id)
                                cleared += 1
                except Exception:
                    pass
    _record_event(request, "memory_cleared", {"entries": cleared})
    return JSONResponse({"status": "cleared", "entries": cleared})


# ── Tool Re-run ───────────────────────────────────────────────────────────


@router.post("/tools/{tool_name}/rerun")
async def tool_rerun(request: Request, tool_name: str, arguments: str = Form("{}")) -> JSONResponse:
    """Re-run a tool with the given JSON arguments."""
    runtime = _get_runtime(request)
    executor = getattr(runtime, "tool_executor", None) if runtime else None
    if executor is None:
        return JSONResponse(
            {"status": "error", "message": "Tool executor not available"}, status_code=503
        )

    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return JSONResponse(
            {"status": "error", "message": "Invalid JSON arguments"}, status_code=400
        )

    if not hasattr(executor, "execute"):
        return JSONResponse(
            {"status": "error", "message": "Executor has no execute method"}, status_code=503
        )

    result = await executor.execute(tool_name, args)
    _record_event(
        request,
        "tool_rerun",
        {
            "tool_name": tool_name,
            "success": result.success,
            "duration_ms": round(result.duration_ms, 2),
        },
    )
    return JSONResponse(
        {
            "status": "success" if result.success else "error",
            "output": result.output,
            "duration_ms": round(result.duration_ms, 2),
        }
    )


# ── Events Export ─────────────────────────────────────────────────────────


@router.get("/events/export")
async def events_export(request: Request, format: str = "json") -> StreamingResponse:
    """Export events as JSON or CSV."""
    runtime = _get_runtime(request)
    ledger = getattr(runtime, "event_ledger", None) if runtime else None
    events = ledger.query(limit=10000) if ledger else []

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp", "type", "source", "payload"])
        for ev in events:
            writer.writerow(
                [
                    datetime.fromtimestamp(ev.get("timestamp", 0), tz=UTC).isoformat(),
                    ev.get("type", ""),
                    ev.get("source", ""),
                    json.dumps(ev.get("payload", {})),
                ]
            )
        content = output.getvalue()
        media_type = "text/csv"
        filename = "animus_events.csv"
    else:
        content = json.dumps(events, indent=2, default=str)
        media_type = "application/json"
        filename = "animus_events.json"

    _record_event(request, "events_exported", {"format": format, "count": len(events)})

    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Alert Acknowledge ─────────────────────────────────────────────────────


@router.post("/alerts/acknowledge")
async def alert_acknowledge(request: Request) -> JSONResponse:
    """Acknowledge an alert by type."""
    # Accept either form-encoded or JSON body
    alert_type = "all"
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            alert_type = body.get("alert_type", "all")
        except Exception:
            pass
    else:
        form = await request.form()
        alert_type = form.get("alert_type", "all")
    _record_event(request, "alert_acknowledged", {"alert_type": alert_type})
    return JSONResponse({"status": "acknowledged", "alert_type": alert_type})
