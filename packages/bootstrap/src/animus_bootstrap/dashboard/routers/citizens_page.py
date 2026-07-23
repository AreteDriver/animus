"""Citizens dashboard router — unified citizen control surface."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from animus_bootstrap.intelligence.citizen_bridge import CitizenBridge, CitizenProposalView

router = APIRouter()


def _get_bridge(request: Request) -> CitizenBridge:
    """Build a CitizenBridge wired to the current runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    return CitizenBridge(runtime)


def _get_event_ledger(request: Request) -> object | None:
    """Safely retrieve the event ledger from runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return None
    return getattr(runtime, "event_ledger", None)


def _record_event(request: Request, event_type: str, payload: dict) -> None:
    """Record an event to the ledger if available."""
    ledger = _get_event_ledger(request)
    if ledger is not None:
        ledger.record(event_type, "dashboard", payload)


# ── Main Citizens Page ────────────────────────────────────────────────────


@router.get("/citizens")
async def citizens_overview(request: Request) -> object:
    """Render the unified citizens overview page."""
    templates = request.app.state.templates
    bridge = _get_bridge(request)

    statuses = bridge.get_citizen_statuses()
    summary = bridge.summary()
    proposals = bridge.list_proposals(limit=20)

    # Split proposals by status for the template
    pending = [p for p in proposals if p.status in ("draft", "submitted", "pending_review")]
    approved = [p for p in proposals if p.status == "approved"]
    completed = [p for p in proposals if p.status in ("complete", "implemented")]

    return templates.TemplateResponse(
        request,
        "citizens.html",
        {
            "statuses": statuses,
            "summary": summary,
            "pending": pending,
            "approved": approved,
            "completed": completed,
        },
    )


# ── Proposals Sub-Page ────────────────────────────────────────────────────


@router.get("/citizens/proposals")
async def citizens_proposals(request: Request) -> object:
    """Render the citizen proposals list page."""
    templates = request.app.state.templates
    bridge = _get_bridge(request)

    status_filter = request.query_params.get("status", "")
    citizen_filter = request.query_params.get("citizen", "")

    proposals = bridge.list_proposals(
        citizen_name=citizen_filter or None,
        status=status_filter or None,
        limit=100,
    )

    return templates.TemplateResponse(
        request,
        "citizens_proposals.html",
        {
            "proposals": proposals,
            "status_filter": status_filter,
            "citizen_filter": citizen_filter,
        },
    )


# ── Per-Citizen Detail ────────────────────────────────────────────────────


@router.get("/citizens/{name}")
async def citizen_detail(request: Request, name: str) -> object:
    """Render detail page for a single citizen."""
    templates = request.app.state.templates
    bridge = _get_bridge(request)

    # Find the citizen status
    statuses = bridge.get_citizen_statuses()
    citizen = next((s for s in statuses if s.name == name), None)

    # Get proposals for this citizen
    proposals = bridge.list_proposals(citizen_name=name, limit=50)

    # Count by status
    proposal_count = len(proposals)
    approved_count = sum(1 for p in proposals if p.status == "approved")
    pending_count = sum(1 for p in proposals if p.status in ("draft", "submitted", "pending_review"))
    rejected_count = sum(1 for p in proposals if p.status == "rejected")

    return templates.TemplateResponse(
        request,
        "citizen_detail.html",
        {
            "citizen": citizen,
            "proposals": proposals,
            "name": name,
            "display_name": citizen.display_name if citizen else name.replace("_", " ").title(),
            "description": citizen.description if citizen else "",
            "state": citizen.state if citizen else "unavailable",
            "proposal_count": proposal_count,
            "approved_count": approved_count,
            "pending_count": pending_count,
            "rejected_count": rejected_count,
        },
    )


# ── Actions (POST) ──────────────────────────────────────────────────────────


def _fallback_proposal(proposal_id: str) -> CitizenProposalView:
    """Return a minimal proposal view for when the real one can't be loaded."""
    return CitizenProposalView(
        id=proposal_id,
        title="Proposal " + proposal_id[:8],
        status="unknown",
        source_citizen="",
    )


@router.post("/citizens/proposals/{proposal_id}/approve")
async def approve_citizen_proposal(request: Request, proposal_id: str) -> object:
    """Approve a citizen proposal."""
    templates = request.app.state.templates
    bridge = _get_bridge(request)

    result = bridge.approve(proposal_id)
    _record_event(request, "citizen_proposal_approved", {
        "proposal_id": proposal_id,
        "success": result.get("success", False),
    })

    # Re-fetch the proposal so the fragment renders the updated row
    proposal = bridge.get_proposal(proposal_id)
    if proposal is None:
        proposal = _fallback_proposal(proposal_id)
        proposal.status = "approved"

    return templates.TemplateResponse(
        request,
        "fragments/citizen_proposal_action.html",
        {
            "proposal": proposal,
            "action": "approved",
            "success": result.get("success", False),
        },
    )


@router.post("/citizens/proposals/{proposal_id}/reject")
async def reject_citizen_proposal(request: Request, proposal_id: str) -> object:
    """Reject a citizen proposal."""
    templates = request.app.state.templates
    bridge = _get_bridge(request)

    result = bridge.reject(proposal_id)
    _record_event(request, "citizen_proposal_rejected", {
        "proposal_id": proposal_id,
        "success": result.get("success", False),
    })

    proposal = bridge.get_proposal(proposal_id)
    if proposal is None:
        proposal = _fallback_proposal(proposal_id)
        proposal.status = "rejected"

    return templates.TemplateResponse(
        request,
        "fragments/citizen_proposal_action.html",
        {
            "proposal": proposal,
            "action": "rejected",
            "success": result.get("success", False),
        },
    )


@router.post("/citizens/proposals/{proposal_id}/commission")
async def commission_citizen_proposal(request: Request, proposal_id: str) -> object:
    """Commission an approved proposal to Forge."""
    templates = request.app.state.templates
    bridge = _get_bridge(request)

    result = bridge.commission(proposal_id)
    _record_event(request, "citizen_proposal_commissioned", {
        "proposal_id": proposal_id,
        "success": result.get("success", False),
        "stage_reached": result.get("stage_reached", ""),
    })

    proposal = bridge.get_proposal(proposal_id)
    if proposal is None:
        proposal = _fallback_proposal(proposal_id)
        proposal.status = "commissioned"

    return templates.TemplateResponse(
        request,
        "fragments/citizen_proposal_action.html",
        {
            "proposal": proposal,
            "action": "commissioned",
            "success": result.get("success", False),
            "error": result.get("error", ""),
            "simulated": result.get("simulated", False),
        },
    )


# ── HTMX Fragments (for polling) ──────────────────────────────────────────


@router.get("/citizens/fragments/status-cards")
async def citizens_status_cards_fragment(request: Request) -> object:
    """Return citizen status cards as an HTML fragment (for HTMX polling)."""
    templates = request.app.state.templates
    bridge = _get_bridge(request)
    statuses = bridge.get_citizen_statuses()
    return templates.TemplateResponse(
        request,
        "fragments/citizens_status_cards.html",
        {"statuses": statuses},
    )


@router.get("/citizens/fragments/proposals-table")
async def citizens_proposals_table_fragment(request: Request) -> object:
    """Return recent proposals table as an HTML fragment (for HTMX polling)."""
    templates = request.app.state.templates
    bridge = _get_bridge(request)
    proposals = bridge.list_proposals(limit=20)
    pending = [p for p in proposals if p.status in ("draft", "submitted", "pending_review")]
    approved = [p for p in proposals if p.status == "approved"]
    return templates.TemplateResponse(
        request,
        "fragments/citizens_proposals_table.html",
        {"pending": pending, "approved": approved},
    )


# ── API Endpoints (JSON) ──────────────────────────────────────────────────


@router.get("/api/citizens/summary")
async def citizens_summary_api(request: Request) -> JSONResponse:
    """Return JSON summary of citizen activity."""
    bridge = _get_bridge(request)
    return JSONResponse(bridge.summary())
