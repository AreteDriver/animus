"""Proactive engine activity page router."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


def _get_runtime(request: Request) -> object | None:
    """Safely retrieve the runtime from app state."""
    return getattr(request.app.state, "runtime", None)


@router.get("/activity")
async def activity_page(request: Request) -> object:
    """Render the proactive engine activity page."""
    templates = request.app.state.templates

    engine_status = "stopped"
    checks: list[object] = []
    nudge_history: list[object] = []

    runtime = _get_runtime(request)
    if runtime is not None and getattr(runtime, "proactive_engine", None) is not None:
        engine = runtime.proactive_engine
        engine_status = "running" if engine.running else "stopped"
        checks = engine.list_checks()
        nudge_history = engine.get_nudge_history(limit=50)

    return templates.TemplateResponse(
        "activity.html",
        {
            "request": request,
            "engine_status": engine_status,
            "checks": checks,
            "nudge_history": nudge_history,
        },
    )
