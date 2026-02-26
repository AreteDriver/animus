"""Identity dashboard router — view and edit identity files."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

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

    return templates.TemplateResponse(
        "identity.html",
        {"request": request, "files": files},
    )


@router.get("/identity/edit/{filename}")
async def identity_edit_form(filename: str, request: Request) -> HTMLResponse:
    """Return an HTMX partial with a textarea for editing an identity file."""
    mgr = _get_identity_manager(request)
    if mgr is None:
        return HTMLResponse(
            '<p class="text-animus-red text-sm">Identity manager not available.</p>'
        )

    try:
        content = mgr.read(filename)
    except ValueError:
        return HTMLResponse(f'<p class="text-animus-red text-sm">Unknown file: {filename}</p>')

    locked = filename in mgr.LOCKED_FILES
    ro = 'readonly class="opacity-60 cursor-not-allowed"' if locked else ""
    card_id = filename.replace(".", "-")
    ta_cls = (
        "w-full bg-animus-bg border border-animus-border "
        "rounded p-3 text-sm text-animus-text font-mono resize-y"
    )
    save_btn = (
        '<button type="submit" class="bg-animus-green '
        "text-animus-bg font-bold px-4 py-1 rounded text-xs "
        'hover:bg-animus-green/80">Save</button>'
        if not locked
        else ""
    )
    cancel_cls = (
        "bg-animus-border text-animus-text px-4 py-1 rounded text-xs hover:bg-animus-muted/20"
    )
    return HTMLResponse(
        f'<form hx-put="/identity/{filename}" '
        f'hx-target="#card-{card_id}" hx-swap="innerHTML">'
        f'<textarea name="content" rows="12" class="{ta_cls}" '
        f"{ro}>{content}</textarea>"
        f'<div class="flex gap-2 mt-2">{save_btn}'
        f'<button type="button" hx-get="/identity/view/{filename}" '
        f'hx-target="#card-{card_id}" hx-swap="innerHTML" '
        f'class="{cancel_cls}">Cancel</button></div></form>'
    )


@router.put("/identity/{filename}")
async def identity_save(filename: str, request: Request, content: str = Form("")) -> HTMLResponse:
    """Save content to an identity file and return the updated view."""
    mgr = _get_identity_manager(request)
    if mgr is None:
        return HTMLResponse(
            '<p class="text-animus-red text-sm">Identity manager not available.</p>'
        )

    locked = filename in mgr.LOCKED_FILES
    try:
        if locked:
            mgr.write_locked(filename, content)
        else:
            mgr.write(filename, content)
    except (ValueError, PermissionError) as exc:
        return HTMLResponse(f'<p class="text-animus-red text-sm">{exc}</p>')

    return _render_file_view(filename, content, locked)


@router.get("/identity/view/{filename}")
async def identity_view(filename: str, request: Request) -> HTMLResponse:
    """Return an HTMX partial with the rendered identity file view."""
    mgr = _get_identity_manager(request)
    if mgr is None:
        return HTMLResponse(
            '<p class="text-animus-red text-sm">Identity manager not available.</p>'
        )

    try:
        content = mgr.read(filename)
    except ValueError:
        return HTMLResponse(f'<p class="text-animus-red text-sm">Unknown file: {filename}</p>')

    locked = filename in mgr.LOCKED_FILES
    return _render_file_view(filename, content, locked)


def _render_file_view(filename: str, content: str, locked: bool) -> HTMLResponse:
    """Render a file card's inner content with Edit button."""
    lock_icon = ' <span title="Immutable — human-edit only">&#128274;</span>' if locked else ""
    edit_btn = (
        ""
        if filename == "LEARNED.md"
        else f'<button hx-get="/identity/edit/{filename}" '
        f'hx-target="#card-{filename.replace(".", "-")}" hx-swap="innerHTML" '
        f'class="text-xs text-animus-green hover:underline">Edit</button>'
    )

    preview = (
        content[:500].replace("<", "&lt;").replace(">", "&gt;") if content else "<em>Empty</em>"
    )
    safe_name = filename.replace("<", "&lt;").replace(">", "&gt;")

    return HTMLResponse(f"""
    <div class="flex items-center justify-between mb-2">
        <h4 class="text-sm font-bold text-animus-text">{safe_name}{lock_icon}</h4>
        {edit_btn}
    </div>
    <pre class="text-xs text-animus-muted whitespace-pre-wrap">{preview}</pre>
    """)
