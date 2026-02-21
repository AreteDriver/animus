"""Tools management page router."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


def _get_runtime(request: Request) -> object | None:
    """Safely retrieve the runtime from app state."""
    return getattr(request.app.state, "runtime", None)


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

    return templates.TemplateResponse(
        "tools.html",
        {"request": request, "tools": tool_list, "history": history},
    )
