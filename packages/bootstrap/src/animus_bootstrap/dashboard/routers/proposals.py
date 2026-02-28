"""Proposals dashboard router — approve/reject identity change proposals."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


def _get_stores(request: Request) -> tuple:
    """Get improvement store and identity manager from runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return None, None
    improvement_store = getattr(runtime, "_improvement_store", None)
    identity_manager = getattr(runtime, "identity_manager", None)
    return improvement_store, identity_manager


@router.get("/proposals")
async def proposals_page(request: Request) -> object:
    """Render the proposals dashboard — THE CRITICAL PAGE."""
    templates = request.app.state.templates
    store, _ = _get_stores(request)

    pending: list[dict] = []
    history: list[dict] = []

    if store is not None:
        all_proposals = store.list_all()
        for p in all_proposals:
            if p.get("area", "").startswith("identity:"):
                if p["status"] == "proposed":
                    pending.append(p)
                else:
                    history.append(p)

    return templates.TemplateResponse(
        "proposals.html",
        {
            "request": request,
            "pending": pending,
            "history": history,
            "pending_count": len(pending),
        },
    )


@router.post("/proposals/{proposal_id}/approve")
async def approve_proposal(proposal_id: int, request: Request) -> HTMLResponse:
    """Approve a proposal — apply the change to the identity file."""
    store, mgr = _get_stores(request)
    if store is None or mgr is None:
        return HTMLResponse('<p class="text-animus-red text-sm">Not available.</p>')

    proposal = store.get(proposal_id)
    if proposal is None:
        return HTMLResponse('<p class="text-animus-red text-sm">Proposal not found.</p>')

    # Extract filename from area (e.g., "identity:CONTEXT.md")
    area = proposal.get("area", "")
    filename = area.split(":", 1)[1] if ":" in area else ""

    if not filename:
        return HTMLResponse('<p class="text-animus-red text-sm">Invalid proposal area.</p>')

    # Apply the change
    content = proposal.get("patch", "")
    try:
        if filename in mgr.LOCKED_FILES:
            return HTMLResponse('<p class="text-animus-red text-sm">Cannot modify locked file.</p>')
        mgr.write(filename, content)
        from datetime import UTC, datetime

        store.update_status(proposal_id, "approved", datetime.now(UTC).isoformat())
    except (ValueError, PermissionError):
        return HTMLResponse('<p class="text-animus-red text-sm">Failed to apply proposal.</p>')

    return HTMLResponse(
        f'<div class="bg-animus-green/10 border border-animus-green rounded p-3 text-sm">'
        f'<span class="text-animus-green">Approved</span> — {filename} updated.</div>'
    )


@router.post("/proposals/{proposal_id}/reject")
async def reject_proposal(proposal_id: int, request: Request) -> HTMLResponse:
    """Reject a proposal — log the rejection."""
    store, _ = _get_stores(request)
    if store is None:
        return HTMLResponse('<p class="text-animus-red text-sm">Not available.</p>')

    proposal = store.get(proposal_id)
    if proposal is None:
        return HTMLResponse('<p class="text-animus-red text-sm">Proposal not found.</p>')

    from datetime import UTC, datetime

    store.update_status(proposal_id, "rejected", datetime.now(UTC).isoformat())

    return HTMLResponse(
        '<div class="bg-animus-red/10 border border-animus-red rounded p-3 text-sm">'
        '<span class="text-animus-red">Rejected</span> — proposal dismissed.</div>'
    )
