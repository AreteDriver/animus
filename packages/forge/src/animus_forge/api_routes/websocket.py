"""WebSocket endpoints for real-time execution and agent updates."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from animus_forge import api_state as state
from animus_forge.auth import verify_token

logger = logging.getLogger(__name__)

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


@router.websocket("/ws/agents")
async def websocket_agents(
    websocket: WebSocket,
    token: str | None = Query(None),
):
    """WebSocket endpoint for real-time agent status updates.

    Streams agent task progress, run status, and tool execution events.
    Auto-subscribes to all agent events (no manual subscription needed).

    Authentication via query parameter: ws://host/ws/agents?token=<jwt>

    Server sends JSON messages:
        {"type": "agent_status", "task_id": "...", "agent": "...", "status": "...", ...}
        {"type": "agent_log", "task_id": "...", "level": "...", "message": "...", ...}
        {"type": "agent_run", "run_id": "...", "agent": "...", "status": "...", ...}
        {"type": "ping"}

    Client can send:
        {"type": "ping"} -> receives {"type": "pong"}
    """
    if not token:
        await websocket.close(code=4001, reason="Missing token parameter")
        return

    user_id = verify_token(token)
    if not user_id:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    await websocket.accept()

    # Message queue for this connection
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=256)

    # Register broadcaster callback
    def on_agent_event(
        event_type: str,
        execution_id: str,
        **kwargs,
    ) -> None:
        """Callback from Broadcaster — enqueue for this WS client."""
        msg = {
            "type": f"agent_{event_type}",
            "task_id": execution_id,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        try:
            queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass  # Drop oldest if client can't keep up

    # Wire callback into broadcaster if available
    broadcaster = state.ws_broadcaster
    _cb_registered = False
    if broadcaster is not None:
        try:
            broadcaster._agent_ws_callbacks = getattr(broadcaster, "_agent_ws_callbacks", [])
            broadcaster._agent_ws_callbacks.append(on_agent_event)
            _cb_registered = True
        except Exception:
            pass

    try:
        await websocket.send_json({"type": "connected", "user_id": user_id})

        # Run sender and receiver concurrently
        async def _sender():
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                    await websocket.send_json(msg)
                except TimeoutError:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break

        async def _receiver():
            while True:
                try:
                    data = await websocket.receive_text()
                    parsed = json.loads(data)
                    if parsed.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except (WebSocketDisconnect, Exception):
                    break

        # Also poll SubAgentManager for run status changes
        async def _run_poller():
            """Periodically poll SubAgentManager and push run updates."""
            sam = state.subagent_manager
            if sam is None:
                return
            seen_states: dict[str, str] = {}
            while True:
                try:
                    await asyncio.sleep(2.0)
                    runs = sam.list_runs()
                    for run in runs:
                        prev = seen_states.get(run.run_id)
                        current = run.status.value
                        if prev != current:
                            seen_states[run.run_id] = current
                            try:
                                queue.put_nowait(
                                    {
                                        "type": "agent_run",
                                        "run_id": run.run_id,
                                        "agent": run.agent,
                                        "status": current,
                                        "task": run.task[:200],
                                    }
                                )
                            except asyncio.QueueFull:
                                pass
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(5.0)

        done, pending = await asyncio.wait(
            [
                asyncio.create_task(_sender()),
                asyncio.create_task(_receiver()),
                asyncio.create_task(_run_poller()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("Agent WS error: %s", e)
    finally:
        # Unregister callback
        if _cb_registered and broadcaster is not None:
            try:
                broadcaster._agent_ws_callbacks.remove(on_agent_event)
            except (ValueError, AttributeError):
                pass
