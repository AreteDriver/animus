"""WebSocket endpoint for real-time execution updates."""

from __future__ import annotations

from fastapi import APIRouter, Query, WebSocket

from animus_forge import api_state as state
from animus_forge.auth import verify_token

router = APIRouter()


@router.websocket("/ws/executions")
async def websocket_executions(
    websocket: WebSocket,
    token: str | None = Query(None),
):
    """WebSocket endpoint for real-time execution updates.

    Authentication via query parameter: ws://host/ws/executions?token=<jwt>

    Protocol:
    - Client sends: subscribe, unsubscribe, ping
    - Server sends: connected, execution_status, execution_log, execution_metrics, pong, error
    """
    if not token:
        await websocket.close(code=4001, reason="Missing token parameter")
        return

    user_id = verify_token(token)
    if not user_id:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    if state.ws_manager is None:
        await websocket.close(code=4500, reason="WebSocket not available")
        return

    await state.ws_manager.handle_connection(websocket)
