"""Identity dashboard router — view and edit identity files."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request

router = APIRouter()


def _get_identity_manager(request: Request):  # noqa: ANN202
    """Safely retrieve the identity manager from runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return None
    return getattr(runtime, "identity_manager", None)


@router.get("/identity")
async def identity_page(request: Request) -> object:
    """Render the identity files dashboard page."""
    templates = request.app.state.templates
    mgr = _get_identity_manager(request)

    files: list[dict] = []
    if mgr is not None:
        for filename in mgr.ALL_FILES:
            content = mgr.read(filename)
            files.append(
                {
                    "filename": filename,
                    "content": content,
                    "exists": mgr.exists(filename),
                    "locked": filename in mgr.LOCKED_FILES,
                    "size": len(content),
                }
            )

    return templates.TemplateResponse(request, "identity.html", {"files": files})


@router.get("/identity/edit/{filename}")
async def identity_edit_form(filename: str, request: Request) -> object:
    """Return an HTMX partial with a textarea for editing an identity file."""
    templates = request.app.state.templates
    mgr = _get_identity_manager(request)
    if mgr is None:
        return templates.TemplateResponse(
            request,
            "fragments/identity_edit_form.html",
            {
                "filename": "",
                "card_id": "",
                "content": "",
                "locked": True,
                "error": "Identity manager not available.",
            },
        )

    try:
        content = mgr.read(filename)
    except ValueError:
        return templates.TemplateResponse(
            request,
            "fragments/identity_edit_form.html",
            {
                "filename": "",
                "card_id": "",
                "content": "",
                "locked": True,
                "error": "Unknown identity file.",
            },
        )

    locked = filename in mgr.LOCKED_FILES
    card_id = filename.replace(".", "-")
    return templates.TemplateResponse(
        request,
        "fragments/identity_edit_form.html",
        {"filename": filename, "card_id": card_id, "content": content, "locked": locked},
    )


@router.put("/identity/{filename}")
async def identity_save(filename: str, request: Request, content: str = Form("")) -> object:
    """Save content to an identity file and return the updated view."""
    templates = request.app.state.templates
    mgr = _get_identity_manager(request)
    if mgr is None:
        return templates.TemplateResponse(
            request,
            "fragments/identity_file_view.html",
            {
                "filename": "",
                "card_id": "",
                "preview": "",
                "locked": True,
                "error": "Identity manager not available.",
            },
        )

    locked = filename in mgr.LOCKED_FILES
    try:
        if locked:
            mgr.write_locked(filename, content)
        else:
            mgr.write(filename, content)
    except (ValueError, PermissionError):
        return templates.TemplateResponse(
            request,
            "fragments/identity_file_view.html",
            {
                "filename": filename,
                "card_id": filename.replace(".", "-"),
                "preview": "",
                "locked": locked,
                "error": "Failed to save file.",
            },
        )

    return _render_file_view(request, filename, content, locked)


@router.get("/identity/view/{filename}")
async def identity_view(filename: str, request: Request) -> object:
    """Return an HTMX partial with the rendered identity file view."""
    templates = request.app.state.templates
    mgr = _get_identity_manager(request)
    if mgr is None:
        return templates.TemplateResponse(
            request,
            "fragments/identity_file_view.html",
            {
                "filename": "",
                "card_id": "",
                "preview": "",
                "locked": True,
                "error": "Identity manager not available.",
            },
        )

    try:
        content = mgr.read(filename)
    except ValueError:
        return templates.TemplateResponse(
            request,
            "fragments/identity_file_view.html",
            {
                "filename": "",
                "card_id": "",
                "preview": "",
                "locked": True,
                "error": "Unknown identity file.",
            },
        )

    locked = filename in mgr.LOCKED_FILES
    return _render_file_view(request, filename, content, locked)


def _render_file_view(request: Request, filename: str, content: str, locked: bool) -> object:
    """Render a file card's inner content with Edit button."""
    templates = request.app.state.templates
    preview = content[:500] if content else ""
    card_id = filename.replace(".", "-")
    return templates.TemplateResponse(
        request,
        "fragments/identity_file_view.html",
        {"filename": filename, "card_id": card_id, "preview": preview, "locked": locked},
    )
