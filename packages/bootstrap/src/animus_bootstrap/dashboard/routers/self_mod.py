"""Self-modification activity dashboard router."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


def _get_runtime(request: Request) -> object | None:
    """Safely retrieve the runtime from app state."""
    return getattr(request.app.state, "runtime", None)


@router.get("/self-mod")
async def self_mod_page(request: Request) -> object:
    """Render the self-modification activity page."""
    templates = request.app.state.templates

    # Code-edit tool history (filter from tool executor)
    code_history: list[dict] = []
    runtime = _get_runtime(request)
    if runtime is not None and getattr(runtime, "tool_executor", None) is not None:
        code_tool_names = {"code_read", "code_write", "code_patch", "code_list"}
        for entry in runtime.tool_executor.get_history(limit=200):
            if entry.tool_name in code_tool_names:
                code_history.append(
                    {
                        "timestamp": entry.timestamp,
                        "tool": entry.tool_name,
                        "success": entry.success,
                        "duration": entry.duration_ms,
                        "output": getattr(entry, "output", "")[:200],
                    }
                )
        code_history = code_history[:50]

    # Improvement proposals from self_improve module
    from animus_bootstrap.intelligence.tools.builtin.self_improve import (
        get_improvement_log,
    )

    improvements = get_improvement_log()

    return templates.TemplateResponse(
        "self_mod.html",
        {
            "request": request,
            "code_history": code_history,
            "improvements": improvements,
        },
    )
