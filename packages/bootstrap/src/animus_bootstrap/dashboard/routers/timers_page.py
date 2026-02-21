"""Timer management dashboard router."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


def _get_runtime(request: Request) -> object | None:
    """Safely retrieve the runtime from app state."""
    return getattr(request.app.state, "runtime", None)


@router.get("/timers")
async def timers_page(request: Request) -> object:
    """Render the timer management page."""
    templates = request.app.state.templates

    from animus_bootstrap.intelligence.tools.builtin.timer_ctl import (
        get_dynamic_timers,
    )

    dynamic_timers = get_dynamic_timers()

    # Enrich with live status from ProactiveEngine
    engine_timers: list[dict] = []
    runtime = _get_runtime(request)
    if runtime is not None and getattr(runtime, "proactive_engine", None) is not None:
        engine = runtime.proactive_engine
        for check in engine.list_checks():
            if check.name.startswith("timer:"):
                engine_timers.append(
                    {
                        "name": check.name.removeprefix("timer:"),
                        "schedule": check.schedule,
                        "enabled": check.enabled,
                        "next_fire": getattr(check, "next_fire", None),
                    }
                )

    return templates.TemplateResponse(
        "timers.html",
        {
            "request": request,
            "dynamic_timers": dynamic_timers,
            "engine_timers": engine_timers,
        },
    )
