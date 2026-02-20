"""Webhook management and trigger endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Header, Request

from animus_forge import api_state as state
from animus_forge.api_errors import (
    AUTH_RESPONSES,
    CRUD_RESPONSES,
    bad_request,
    internal_error,
    not_found,
    responses,
    unauthorized,
)
from animus_forge.api_routes.auth import verify_auth
from animus_forge.webhooks import Webhook

router = APIRouter()

# Public webhook trigger — registered on app, not v1_router
trigger_router = APIRouter()


# ---------------------------------------------------------------------------
# DLQ management endpoints  (must be before /webhooks/{webhook_id} to avoid
# the path parameter swallowing "dlq" as a webhook_id)
# ---------------------------------------------------------------------------


@router.get("/webhooks/dlq", responses=AUTH_RESPONSES)
def list_dlq_items(
    limit: int = 100,
    authorization: str | None = Header(None),
):
    """List pending dead-letter queue items."""
    verify_auth(authorization)
    if not state.delivery_manager:
        raise internal_error("Delivery manager not initialized")
    return state.delivery_manager.get_dlq_items(limit=limit)


@router.get("/webhooks/dlq/stats", responses=AUTH_RESPONSES)
def get_dlq_stats(authorization: str | None = Header(None)):
    """Get DLQ statistics: count by URL, oldest item age."""
    verify_auth(authorization)
    if not state.delivery_manager:
        raise internal_error("Delivery manager not initialized")
    return state.delivery_manager.get_dlq_stats()


@router.post("/webhooks/dlq/retry-all", responses=AUTH_RESPONSES)
def retry_all_dlq(
    max_items: int = 50,
    authorization: str | None = Header(None),
):
    """Batch retry all pending DLQ items."""
    verify_auth(authorization)
    if not state.delivery_manager:
        raise internal_error("Delivery manager not initialized")
    results = state.delivery_manager.reprocess_all_dlq(max_items=max_items)
    return {"status": "success", "processed": len(results), "results": results}


@router.post("/webhooks/dlq/{dlq_id}/retry", responses=CRUD_RESPONSES)
def retry_dlq_item(
    dlq_id: int,
    authorization: str | None = Header(None),
):
    """Retry a specific DLQ item."""
    verify_auth(authorization)
    if not state.delivery_manager:
        raise internal_error("Delivery manager not initialized")
    try:
        delivery = state.delivery_manager.reprocess_dlq_item(dlq_id)
        return {
            "status": "success",
            "delivery_status": delivery.status.value,
            "dlq_id": dlq_id,
        }
    except ValueError:
        raise not_found("DLQ item", str(dlq_id))


@router.delete("/webhooks/dlq/{dlq_id}", responses=CRUD_RESPONSES)
def delete_dlq_item(
    dlq_id: int,
    authorization: str | None = Header(None),
):
    """Remove a DLQ item without retrying."""
    verify_auth(authorization)
    if not state.delivery_manager:
        raise internal_error("Delivery manager not initialized")
    if state.delivery_manager.delete_dlq_item(dlq_id):
        return {"status": "success"}
    raise not_found("DLQ item", str(dlq_id))


# ---------------------------------------------------------------------------
# Webhook CRUD endpoints
# ---------------------------------------------------------------------------


@router.get("/webhooks", responses=AUTH_RESPONSES)
def list_webhooks(authorization: str | None = Header(None)):
    """List all webhooks."""
    verify_auth(authorization)
    return state.webhook_manager.list_webhooks()


@router.get("/webhooks/{webhook_id}", responses=CRUD_RESPONSES)
def get_webhook(webhook_id: str, authorization: str | None = Header(None)):
    """Get a specific webhook (secret redacted)."""
    verify_auth(authorization)
    webhook = state.webhook_manager.get_webhook(webhook_id)

    if not webhook:
        raise not_found("Webhook", webhook_id)

    if hasattr(webhook, "model_dump"):
        result = webhook.model_dump()
    elif isinstance(webhook, dict):
        result = dict(webhook)
    else:
        result = vars(webhook)
    result["secret"] = "***REDACTED***"
    return result


@router.post("/webhooks", responses=CRUD_RESPONSES)
def create_webhook(webhook: Webhook, authorization: str | None = Header(None)):
    """Create a new webhook."""
    verify_auth(authorization)

    try:
        if state.webhook_manager.create_webhook(webhook):
            return {
                "status": "success",
                "webhook_id": webhook.id,
                "secret": webhook.secret,
                "trigger_url": f"/hooks/{webhook.id}",
            }
        raise internal_error("Failed to save webhook")
    except ValueError as e:
        raise bad_request(str(e))


@router.put("/webhooks/{webhook_id}", responses=CRUD_RESPONSES)
def update_webhook(
    webhook_id: str,
    webhook: Webhook,
    authorization: str | None = Header(None),
):
    """Update an existing webhook."""
    verify_auth(authorization)

    if webhook.id != webhook_id:
        raise bad_request("Webhook ID mismatch", {"expected": webhook_id, "got": webhook.id})

    try:
        if state.webhook_manager.update_webhook(webhook):
            return {"status": "success", "webhook_id": webhook.id}
        raise internal_error("Failed to update webhook")
    except ValueError:
        raise not_found("Webhook", webhook_id)


@router.delete("/webhooks/{webhook_id}", responses=CRUD_RESPONSES)
def delete_webhook(webhook_id: str, authorization: str | None = Header(None)):
    """Delete a webhook."""
    verify_auth(authorization)

    if state.webhook_manager.delete_webhook(webhook_id):
        return {"status": "success"}

    raise not_found("Webhook", webhook_id)


@router.post("/webhooks/{webhook_id}/regenerate-secret", responses=CRUD_RESPONSES)
def regenerate_webhook_secret(webhook_id: str, authorization: str | None = Header(None)):
    """Regenerate the secret for a webhook."""
    verify_auth(authorization)

    try:
        new_secret = state.webhook_manager.regenerate_secret(webhook_id)
        return {"status": "success", "secret": new_secret}
    except ValueError:
        raise not_found("Webhook", webhook_id)


@router.get("/webhooks/{webhook_id}/history", responses=CRUD_RESPONSES)
def get_webhook_history(
    webhook_id: str,
    limit: int = 10,
    authorization: str | None = Header(None),
):
    """Get trigger history for a webhook."""
    verify_auth(authorization)

    webhook = state.webhook_manager.get_webhook(webhook_id)
    if not webhook:
        raise not_found("Webhook", webhook_id)

    history = state.webhook_manager.get_trigger_history(webhook_id, limit)
    return [h.model_dump(mode="json") for h in history]


# ---------------------------------------------------------------------------
# Public webhook trigger (signature-verified, not JWT)
# ---------------------------------------------------------------------------


@trigger_router.post("/hooks/{webhook_id}", responses=responses(400, 401, 404, 429))
@state.limiter.limit("30/minute")
async def trigger_webhook(
    webhook_id: str,
    request: Request,
    x_webhook_signature: str | None = Header(None, alias="X-Webhook-Signature"),
):
    """Public endpoint to trigger a webhook. Rate limited to 30/minute per IP.

    Authentication is via HMAC-SHA256 signature in X-Webhook-Signature header.
    """
    webhook = state.webhook_manager.get_webhook(webhook_id)
    if not webhook:
        raise not_found("Webhook", webhook_id)

    body = await request.body()

    if not x_webhook_signature:
        raise unauthorized("Missing X-Webhook-Signature header")
    if not state.webhook_manager.verify_signature(webhook_id, body, x_webhook_signature):
        raise unauthorized("Invalid webhook signature")

    try:
        payload = await request.json() if body else {}
    except Exception:
        payload = {}

    client_ip = request.client.host if request.client else None

    try:
        result = state.webhook_manager.trigger(webhook_id, payload, source_ip=client_ip)
        return result
    except ValueError as e:
        raise bad_request(str(e))
