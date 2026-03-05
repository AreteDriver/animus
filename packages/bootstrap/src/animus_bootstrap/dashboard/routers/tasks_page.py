"""Task management dashboard router."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter()


def _get_task_store(request: Request) -> object | None:
    """Safely retrieve the task store from runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return None
    return getattr(runtime, "_task_store", None)


@router.get("/tasks")
async def tasks_page(request: Request) -> object:
    """Render the task management page."""
    templates = request.app.state.templates
    store = _get_task_store(request)
    tasks = store.list_all() if store else []
    return templates.TemplateResponse(
        "tasks.html",
        {"request": request, "tasks": tasks},
    )


@router.post("/tasks/create")
async def tasks_create(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    priority: str = Form("normal"),
    due_date: str = Form(""),
) -> RedirectResponse:
    """Create a new task via form submission."""
    store = _get_task_store(request)
    if store:
        store.create(name=name, description=description, priority=priority, due_date=due_date)
    return RedirectResponse(url="/tasks", status_code=303)


@router.post("/tasks/{task_id}/complete")
async def tasks_complete(request: Request, task_id: str) -> HTMLResponse:
    """Mark a task as completed (HTMX)."""
    store = _get_task_store(request)
    if store:
        store.complete(task_id)
    return HTMLResponse('<span class="text-animus-green">Done</span>')


@router.post("/tasks/{task_id}/delete")
async def tasks_delete(request: Request, task_id: str) -> HTMLResponse:
    """Delete a task (HTMX)."""
    store = _get_task_store(request)
    if store:
        store.delete(task_id)
    return HTMLResponse('<span class="text-animus-red">Deleted</span>')
