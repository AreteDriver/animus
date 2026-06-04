"""Web Push subscription router for the PWA."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from animus_bootstrap.config import ConfigManager
from animus_bootstrap.intelligence.push_sender import ensure_vapid_keys

logger = logging.getLogger(__name__)

router = APIRouter()


class SubscribeRequest(BaseModel):
    """A browser PushSubscription payload."""

    subscription: dict


class UnsubscribeRequest(BaseModel):
    """Endpoint of the subscription to remove."""

    endpoint: str


def _get_store(request: Request) -> object | None:
    """Return the push subscription store from app.state, if available."""
    return getattr(request.app.state, "push_store", None)


@router.get("/api/push/vapid-public-key")
async def vapid_public_key(request: Request) -> JSONResponse:
    """Return the VAPID public key, generating the keypair on first use."""
    config = getattr(request.app.state, "config", None)
    if config is None:
        config = ConfigManager().load()
        request.app.state.config = config
    _, public_key = ensure_vapid_keys(config, ConfigManager())
    return JSONResponse(content={"publicKey": public_key})


@router.post("/api/push/subscribe")
async def subscribe(request: Request, payload: SubscribeRequest) -> JSONResponse:
    """Store a browser push subscription."""
    store = _get_store(request)
    if store is None:
        return JSONResponse(status_code=503, content={"detail": "Push not available."})
    try:
        store.add(payload.subscription)  # type: ignore[attr-defined]
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    return JSONResponse(content={"ok": True})


@router.post("/api/push/unsubscribe")
async def unsubscribe(request: Request, payload: UnsubscribeRequest) -> JSONResponse:
    """Remove a push subscription by endpoint."""
    store = _get_store(request)
    if store is None:
        return JSONResponse(status_code=503, content={"detail": "Push not available."})
    store.remove(payload.endpoint)  # type: ignore[attr-defined]
    return JSONResponse(content={"ok": True})
