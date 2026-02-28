"""Proposals dashboard router — approve/reject identity change proposals."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from animus_bootstrap.intelligence.proposals import IdentityProposalManager, Proposal

logger = logging.getLogger(__name__)

router = APIRouter()


def _proposal_to_template(p: Proposal) -> dict:
    """Convert a typed Proposal to a dict matching the template's expected keys."""
    return {
        "id": p.id,
        "area": f"identity:{p.file}",
        "timestamp": p.created_at,
        "description": p.reason,
        "analysis": p.diff or None,
        "patch": p.proposed,
        "status": p.status,
        "applied_at": p.resolved_at,
    }


def _get_stores(request: Request) -> tuple:
    """Get improvement store and identity manager from runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return None, None
    improvement_store = getattr(runtime, "_improvement_store", None)
    identity_manager = getattr(runtime, "identity_manager", None)
    return improvement_store, identity_manager


def _get_proposal_manager(request: Request) -> IdentityProposalManager | None:
    """Get or build an IdentityProposalManager from runtime state."""
    store, mgr = _get_stores(request)
    if store is None or mgr is None:
        return None
    return IdentityProposalManager(store, mgr)


@router.get("/proposals")
async def proposals_page(request: Request) -> object:
    """Render the proposals dashboard — THE CRITICAL PAGE."""
    templates = request.app.state.templates
    pm = _get_proposal_manager(request)

    pending: list = []
    history: list = []

    if pm is not None:
        pending = [_proposal_to_template(p) for p in pm.list_pending()]
        history = [_proposal_to_template(p) for p in pm.history() if p.status != "pending"]

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
    pm = _get_proposal_manager(request)
    if pm is None:
        return HTMLResponse('<p class="text-animus-red text-sm">Not available.</p>')

    try:
        result = pm.approve(proposal_id)
    except ValueError as exc:
        return HTMLResponse(f'<p class="text-animus-red text-sm">{exc}</p>')
    except PermissionError:
        return HTMLResponse('<p class="text-animus-red text-sm">Cannot modify locked file.</p>')

    return HTMLResponse(
        f'<div class="bg-animus-green/10 border border-animus-green rounded p-3 text-sm">'
        f'<span class="text-animus-green">Approved</span> — {result.file} updated.</div>'
    )


@router.post("/proposals/{proposal_id}/reject")
async def reject_proposal(proposal_id: int, request: Request) -> HTMLResponse:
    """Reject a proposal — log the rejection."""
    pm = _get_proposal_manager(request)
    if pm is None:
        return HTMLResponse('<p class="text-animus-red text-sm">Not available.</p>')

    try:
        pm.reject(proposal_id)
    except ValueError as exc:
        return HTMLResponse(f'<p class="text-animus-red text-sm">{exc}</p>')

    return HTMLResponse(
        '<div class="bg-animus-red/10 border border-animus-red rounded p-3 text-sm">'
        '<span class="text-animus-red">Rejected</span> — proposal dismissed.</div>'
    )
