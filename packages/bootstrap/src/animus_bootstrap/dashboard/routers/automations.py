"""Automations management page router."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


def _get_runtime(request: Request) -> object | None:
    """Safely retrieve the runtime from app state."""
    return getattr(request.app.state, "runtime", None)


@router.get("/automations")
async def automations_page(request: Request) -> object:
    """Render the automations management page."""
    templates = request.app.state.templates

    rules: list[object] = []
    history: list[object] = []

    runtime = _get_runtime(request)
    if runtime is not None and getattr(runtime, "automation_engine", None) is not None:
        rules = runtime.automation_engine.list_rules()
        history = runtime.automation_engine.get_history(limit=50)

    return templates.TemplateResponse(
        "automations.html",
        {"request": request, "rules": rules, "history": history},
    )
