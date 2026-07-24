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


class SendTestRequest(BaseModel):
    """Payload for a test push notification."""

    title: str
    body: str
    url: str = "/"


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


@router.post("/api/push/send-test")
async def send_test(request: Request, payload: SendTestRequest) -> JSONResponse:
    """Send a test push notification to all stored subscriptions.

    Returns the number of successful deliveries and how many stale
    subscriptions were pruned.
    """
    store = _get_store(request)
    if store is None:
        return JSONResponse(status_code=503, content={"detail": "Push not available."})

    before_count = store.count()  # type: ignore[attr-defined]
    if before_count == 0:
        return JSONResponse(content={"sent": 0, "pruned": 0, "detail": "No subscriptions."})

    config = getattr(request.app.state, "config", None)
    if config is None:
        config = ConfigManager().load()
        request.app.state.config = config

    private_key, _ = ensure_vapid_keys(config, ConfigManager())

    from animus_bootstrap.intelligence.push_sender import PushSender

    sender = PushSender(store, private_key, "mailto:animus@localhost")
    sent = sender.send(payload.title, payload.body, payload.url)
    after_count = store.count()  # type: ignore[attr-defined]

    return JSONResponse(content={"sent": sent, "pruned": before_count - after_count})
