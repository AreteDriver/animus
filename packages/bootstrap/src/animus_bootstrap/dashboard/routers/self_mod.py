"""Self-modification activity dashboard router."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

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


@router.get("/self-mod/improvement/{proposal_id}")
async def improvement_detail(proposal_id: int, request: Request) -> HTMLResponse:
    """Return an HTML fragment with the full detail for one improvement proposal."""
    from animus_bootstrap.intelligence.tools.builtin.self_improve import (
        get_improvement_log,
    )

    matching = [p for p in get_improvement_log() if p["id"] == proposal_id]
    if not matching:
        return HTMLResponse(
            '<p class="text-animus-red text-sm">Proposal not found.</p>'
        )

    p = matching[0]
    analysis = p.get("analysis") or "No analysis available."
    patch = p.get("patch") or ""

    lines = [
        '<div class="bg-animus-bg border border-animus-border rounded p-4 mt-2'
        ' mb-4 text-sm">',
        f'<p class="text-animus-muted text-xs mb-1">Area: '
        f'<span class="text-animus-green">{p.get("area", "")}</span></p>',
        f'<p class="text-animus-text mb-2">{p.get("description", "")}</p>',
        '<p class="text-animus-muted text-xs mb-1">Analysis:</p>',
        f'<p class="text-animus-text mb-2">{analysis}</p>',
    ]
    if patch:
        lines.append('<p class="text-animus-muted text-xs mb-1">Patch:</p>')
        lines.append(
            f'<pre class="bg-animus-surface p-2 rounded text-xs '
            f'text-animus-text overflow-x-auto">{patch}</pre>'
        )
    lines.append("</div>")
    return HTMLResponse("\n".join(lines))
