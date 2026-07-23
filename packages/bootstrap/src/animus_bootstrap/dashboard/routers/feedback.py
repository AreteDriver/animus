"""Feedback dashboard router — thumbs up/down and feedback stats."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request

router = APIRouter()


def _get_feedback_store(request: Request):  # noqa: ANN202
    """Safely retrieve the feedback store from runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return None
    return getattr(runtime, "feedback_store", None)


def _get_event_ledger(request: Request):  # noqa: ANN202
    """Safely retrieve the event ledger from runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return None
    return getattr(runtime, "event_ledger", None)


@router.post("/api/feedback")
async def record_feedback(
    request: Request,
    message_text: str = Form(""),
    response_text: str = Form(""),
    rating: int = Form(0),
    comment: str = Form(""),
    channel: str = Form("webchat"),
) -> object:
    """Record a thumbs up/down feedback entry, return HTMX partial."""
    templates = request.app.state.templates
    store = _get_feedback_store(request)
    if store is None:
        return templates.TemplateResponse(
            request, "fragments/feedback_result.html",
            {"error": "Feedback not available"}
        )

    store.record(
        message_text=message_text,
        response_text=response_text,
        rating=rating,
        comment=comment,
        channel=channel,
    )

    ledger = _get_event_ledger(request)
    if ledger is not None:
        ledger.record("feedback_recorded", "dashboard", {"rating": rating, "channel": channel})

    icon = "&#128077;" if rating > 0 else "&#128078;"
    return templates.TemplateResponse(
        request, "fragments/feedback_result.html",
        {"icon": icon}
    )


@router.get("/feedback")
async def feedback_page(request: Request) -> object:
    """Render the feedback dashboard page."""
    templates = request.app.state.templates
    store = _get_feedback_store(request)

    stats = {"total": 0, "positive": 0, "negative": 0, "positive_pct": 0, "negative_pct": 0}
    recent: list[dict] = []

    if store is not None:
        stats = store.get_stats()
        recent = store.get_recent(limit=50)

    return templates.TemplateResponse(request, "feedback.html", {"stats": stats, "recent": recent})
