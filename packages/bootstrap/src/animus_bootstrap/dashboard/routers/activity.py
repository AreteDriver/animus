"""Proactive engine activity page router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from animus_bootstrap.intelligence.proactive.metrics import hit_rate_all

router = APIRouter()


def _get_runtime(request: Request) -> object | None:
    """Safely retrieve the runtime from app state."""
    return getattr(request.app.state, "runtime", None)


@router.get("/activity")
async def activity_page(request: Request) -> object:
    """Render the proactive engine activity page."""
    templates = request.app.state.templates

    engine_status = "stopped"
    checks: list[object] = []
    nudge_history: list[object] = []
    hit_rates: list[object] = []

    runtime = _get_runtime(request)
    if runtime is not None and getattr(runtime, "proactive_engine", None) is not None:
        engine = runtime.proactive_engine
        engine_status = "running" if engine.running else "stopped"
        checks = engine.list_checks()
        nudge_history = engine.get_nudge_history(limit=50)

        outcome_store = getattr(runtime, "_outcome_store", None) or engine.outcome_store
        if outcome_store is not None:
            try:
                data_dir = runtime.config.get_data_path()
                hit_rates = hit_rate_all(
                    proactive_log_db=data_dir / "proactive.db",
                    outcomes_db=data_dir / "proactive_outcomes.db",
                    window_days=7,
                )
            except Exception:
                hit_rates = []

    return templates.TemplateResponse(
        request,
        "activity.html",
        {
            "engine_status": engine_status,
            "checks": checks,
            "nudge_history": nudge_history,
            "hit_rates": hit_rates,
        },
    )


@router.post("/activity/feedback/{nudge_id}")
async def record_nudge_feedback(request: Request, nudge_id: str, rating: int) -> dict:
    """Record an explicit thumbs ±1 against a delivered nudge."""
    if rating not in (-1, 1):
        raise HTTPException(status_code=400, detail="rating must be -1 or 1")

    runtime = _get_runtime(request)
    outcome_store = getattr(runtime, "_outcome_store", None) if runtime is not None else None
    if outcome_store is None:
        raise HTTPException(status_code=503, detail="outcome store not initialized")

    outcome_id = outcome_store.record_outcome(
        nudge_id=nudge_id,
        signal_type="explicit",
        signal_value=float(rating),
        source="dashboard",
    )
    return {"ok": True, "outcome_id": outcome_id}
