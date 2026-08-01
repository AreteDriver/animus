"""Self-modification activity dashboard router."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()

_templates: Jinja2Templates | None = None


def _get_templates(request: Request) -> Jinja2Templates:
    """Retrieve Jinja2 templates from app state."""
    global _templates
    if _templates is None:
        _templates = request.app.state.templates
    return _templates


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
        request,
        "self_mod.html",
        {
            "code_history": code_history,
            "improvements": improvements,
        },
    )


@router.get("/self-mod/improvement/{proposal_id}")
async def improvement_detail(proposal_id: int, request: Request) -> object:
    """Return an HTML fragment with the full detail for one improvement proposal."""
    from animus_bootstrap.intelligence.tools.builtin.self_improve import (
        get_improvement_log,
    )

    matching = [p for p in get_improvement_log() if p["id"] == proposal_id]
    if not matching:
        return _get_templates(request).TemplateResponse(
            request,
            "fragments/improvement_detail.html",
            {"area": "", "description": "Proposal not found.", "analysis": "", "patch": ""},
        )

    p = matching[0]
    return _get_templates(request).TemplateResponse(
        request,
        "fragments/improvement_detail.html",
        {
            "area": p.get("area", ""),
            "description": p.get("description", ""),
            "analysis": p.get("analysis") or "No analysis available.",
            "patch": p.get("patch") or "",
        },
    )
