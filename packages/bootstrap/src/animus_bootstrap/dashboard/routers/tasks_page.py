"""Task management dashboard router."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

router = APIRouter()


def _get_task_store(request: Request) -> object | None:
    """Safely retrieve the task store from runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return None
    return getattr(runtime, "_task_store", None)


def _get_event_ledger(request: Request) -> object | None:
    """Safely retrieve the event ledger from runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return None
    return getattr(runtime, "event_ledger", None)


@router.get("/tasks")
async def tasks_page(request: Request) -> object:
    """Render the task management page."""
    templates = request.app.state.templates
    store = _get_task_store(request)
    tasks = store.list_all() if store else []
    return templates.TemplateResponse(request, "tasks.html", {"tasks": tasks})


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
    ledger = _get_event_ledger(request)
    if store:
        task_id = store.create(name=name, description=description, priority=priority, due_date=due_date)
        if ledger is not None:
            ledger.record("task_created", "dashboard", {"task_id": task_id, "name": name})
    return RedirectResponse(url="/tasks", status_code=303)


@router.post("/tasks/{task_id}/complete")
async def tasks_complete(request: Request, task_id: str) -> object:
    """Mark a task as completed (HTMX)."""
    templates = request.app.state.templates
    store = _get_task_store(request)
    ledger = _get_event_ledger(request)
    if store:
        store.complete(task_id)
        if ledger is not None:
            ledger.record("task_completed", "dashboard", {"task_id": task_id})
    return templates.TemplateResponse(
        request, "fragments/task_action_result.html",
        {"css_class": "text-animus-green", "message": "Done"}
    )


@router.post("/tasks/{task_id}/delete")
async def tasks_delete(request: Request, task_id: str) -> object:
    """Delete a task (HTMX)."""
    templates = request.app.state.templates
    store = _get_task_store(request)
    ledger = _get_event_ledger(request)
    if store:
        store.delete(task_id)
        if ledger is not None:
            ledger.record("task_deleted", "dashboard", {"task_id": task_id})
    return templates.TemplateResponse(
        request, "fragments/task_action_result.html",
        {"css_class": "text-animus-red", "message": "Deleted"}
    )
