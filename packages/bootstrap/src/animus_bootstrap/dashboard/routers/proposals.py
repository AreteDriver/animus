"""Proposals dashboard router — approve/reject identity change proposals."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

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
        request,
        "proposals.html",
        {
            "pending": pending,
            "history": history,
            "pending_count": len(pending),
        },
    )


@router.post("/proposals/{proposal_id}/approve")
async def approve_proposal(proposal_id: int, request: Request) -> object:
    """Approve a proposal — apply the change to the identity file."""
    templates = request.app.state.templates
    pm = _get_proposal_manager(request)
    if pm is None:
        return templates.TemplateResponse(
            request, "fragments/proposal_action_result.html",
            {"approved": False, "file": "", "error": "Not available."}
        )

    try:
        result = pm.approve(proposal_id)
    except ValueError:
        return templates.TemplateResponse(
            request, "fragments/proposal_action_result.html",
            {"approved": False, "file": "", "error": "Proposal not found."}
        )
    except PermissionError:
        return templates.TemplateResponse(
            request, "fragments/proposal_action_result.html",
            {"approved": False, "file": "", "error": "Cannot modify locked file."}
        )

    return templates.TemplateResponse(
        request,
        "fragments/proposal_action_result.html",
        {"approved": True, "file": result.file},
    )


@router.post("/proposals/{proposal_id}/reject")
async def reject_proposal(proposal_id: int, request: Request) -> object:
    """Reject a proposal — log the rejection."""
    templates = request.app.state.templates
    pm = _get_proposal_manager(request)
    if pm is None:
        return templates.TemplateResponse(
            request, "fragments/proposal_action_result.html",
            {"approved": False, "file": "", "error": "Not available."}
        )

    try:
        pm.reject(proposal_id)
    except ValueError:
        return templates.TemplateResponse(
            request, "fragments/proposal_action_result.html",
            {"approved": False, "file": "", "error": "Proposal not found."}
        )

    return templates.TemplateResponse(
        request,
        "fragments/proposal_action_result.html",
        {"approved": False, "file": ""},
    )
