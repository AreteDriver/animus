"""Cognitive Event Ledger router — append-only event store UI and API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from animus_bootstrap.ledger import EventType, LedgerEvent, LedgerStore

router = APIRouter()

_LEDGER_DB_KEY = "ledger_db_path"


def _get_ledger_store(request: Request) -> LedgerStore | None:
    """Resolve the LedgerStore from app state or runtime."""
    # Prefer explicit app.state.ledger_store if wired by lifespan
    store: LedgerStore | None = getattr(request.app.state, "ledger_store", None)
    if store is not None:
        return store

    # Fallback: derive from runtime data directory
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is not None:
        data_dir = getattr(runtime, "data_dir", None)
        if data_dir is not None:
            db_path = Path(data_dir) / "cognitive_ledger.db"
            return LedgerStore(db_path=db_path)
    return None


# ------------------------------------------------------------------
# HTML pages
# ------------------------------------------------------------------


@router.get("/ledger")
async def ledger_page(request: Request) -> object:
    """Render the Cognitive Event Ledger browser page."""
    templates = request.app.state.templates
    store = _get_ledger_store(request)

    events: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"total": 0}
    if store is not None:
        entries = store.query(limit=100)
        events = [e.model_dump(mode="json") for e in entries]
        stats = {"total": store.count()}

    return templates.TemplateResponse(
        request,
        "ledger.html",
        {"events": events, "stats": stats},
    )


# ------------------------------------------------------------------
# JSON API
# ------------------------------------------------------------------


@router.get("/api/ledger/events")
async def list_events(
    request: Request,
    object_id: str | None = None,
    event_type: str | None = None,
    workspace_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> JSONResponse:
    """Query ledger events with optional filters."""
    store = _get_ledger_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Ledger store not available")

    et = EventType(event_type) if event_type else None
    entries = store.query(
        object_id=object_id,
        event_type=et,
        workspace_id=workspace_id,
        limit=limit,
        offset=offset,
    )
    return JSONResponse(
        {
            "events": [e.model_dump(mode="json") for e in entries],
            "total": store.count(object_id) if object_id else store.count(),
        }
    )


@router.get("/api/ledger/events/{event_id}")
async def get_event(request: Request, event_id: str) -> JSONResponse:
    """Retrieve a single event by its canonical event_id."""
    store = _get_ledger_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Ledger store not available")

    entry = store.get_by_event_id(event_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return JSONResponse(entry.model_dump(mode="json"))


@router.post("/api/ledger/events")
async def append_event(request: Request) -> JSONResponse:
    """Append a new event to the ledger."""
    store = _get_ledger_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Ledger store not available")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid JSON body") from None

    try:
        event = LedgerEvent.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Validation error: {exc}") from exc

    try:
        entry = store.append(event)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"Duplicate or invalid event: {exc}") from exc
    return JSONResponse(entry.model_dump(mode="json"), status_code=201)


@router.get("/api/ledger/objects/{object_id}/chain")
async def get_object_chain(request: Request, object_id: str) -> JSONResponse:
    """Return the integrity-verified event chain for an object."""
    store = _get_ledger_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Ledger store not available")

    entries = store.get_chain(object_id)
    valid = store.verify_chain(object_id)
    return JSONResponse(
        {
            "object_id": object_id,
            "events": [e.model_dump(mode="json") for e in entries],
            "count": len(entries),
            "integrity_valid": valid,
        }
    )


@router.get("/api/ledger/verify")
async def verify_ledger(request: Request, object_id: str | None = None) -> JSONResponse:
    """Verify integrity chain for the entire ledger or a single object."""
    store = _get_ledger_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Ledger store not available")

    valid = store.verify_chain(object_id)
    return JSONResponse(
        {
            "object_id": object_id,
            "integrity_valid": valid,
        }
    )
