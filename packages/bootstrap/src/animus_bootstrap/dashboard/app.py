"""Animus Bootstrap dashboard — FastAPI application."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import animus_bootstrap
from animus_bootstrap.dashboard.routers import (
    activity,
    automations,
    channels_page,
    config,
    conversations,
    forge_page,
    home,
    logs,
    memory,
    personas_page,
    routing_page,
    self_mod,
    timers_page,
    tools,
    update,
)
from animus_bootstrap.gateway.channels.webchat import WebChatAdapter
from animus_bootstrap.runtime import AnimusRuntime, get_runtime

logger = logging.getLogger(__name__)

_DASHBOARD_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _DASHBOARD_DIR / "static"
_TEMPLATE_DIR = _DASHBOARD_DIR / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start runtime on boot, stop on shutdown."""
    runtime = get_runtime()
    try:
        await runtime.start()
    except Exception:
        logger.warning("Runtime start failed — dashboard running in limited mode")
    app.state.runtime = runtime

    # Wire the approval callback so dashboard can approve LLM-initiated tools
    if runtime.started and getattr(runtime, "tool_executor", None) is not None:
        from animus_bootstrap.dashboard.routers.tools import dashboard_approval_callback

        runtime.tool_executor.set_approval_callback(dashboard_approval_callback)

    # Wire the WebChat adapter to the router so messages get processed
    webchat: WebChatAdapter = app.state.webchat
    await webchat.connect()
    if runtime.started and runtime.router is not None:
        _wire_webchat(webchat, runtime)

    yield

    await webchat.disconnect()
    if runtime.started:
        await runtime.stop()


def _wire_webchat(webchat: WebChatAdapter, runtime: AnimusRuntime) -> None:
    """Connect webchat incoming messages to the router and responses back."""
    from animus_bootstrap.dashboard.routers.conversations import get_message_store
    from animus_bootstrap.gateway.models import GatewayMessage, GatewayResponse

    async def _on_webchat_message(message: GatewayMessage) -> None:
        """Route an incoming webchat message and send the reply back."""
        store = get_message_store()
        # Log user message in the feed
        store.append(
            {
                "channel": message.channel,
                "sender": message.sender_name,
                "text": message.text,
                "timestamp": message.timestamp.isoformat(),
            }
        )

        try:
            response: GatewayResponse = await runtime.router.handle_message(message)
        except Exception:
            logger.exception("Router failed to handle webchat message")
            response = GatewayResponse(text="Sorry, something went wrong.", channel="webchat")

        # Log assistant message in the feed
        store.append(
            {
                "channel": response.channel,
                "sender": "Animus",
                "text": response.text,
                "timestamp": message.timestamp.isoformat(),
            }
        )

        # Send reply back through the WebSocket
        await webchat.send_message(response)

    import asyncio

    asyncio.ensure_future(webchat.on_message(_on_webchat_message))


app = FastAPI(title="Animus Dashboard", version=animus_bootstrap.__version__, lifespan=lifespan)

# Static files
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Jinja2 templates (shared across routers via app.state)
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
templates.env.globals["version"] = animus_bootstrap.__version__
app.state.templates = templates

# Shared WebChat adapter instance
_webchat = WebChatAdapter()
app.state.webchat = _webchat

# Include routers
app.include_router(home.router)
app.include_router(conversations.router)
app.include_router(channels_page.router)
app.include_router(config.router)
app.include_router(memory.router)
app.include_router(logs.router)
app.include_router(update.router)
app.include_router(tools.router)
app.include_router(automations.router)
app.include_router(activity.router)
app.include_router(personas_page.router)
app.include_router(routing_page.router)
app.include_router(self_mod.router)
app.include_router(forge_page.router)
app.include_router(timers_page.router)


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    """Return JSON health status of the runtime and its components."""
    runtime: AnimusRuntime | None = getattr(request.app.state, "runtime", None)
    return JSONResponse(
        {
            "status": "ok" if runtime and runtime.started else "degraded",
            "version": animus_bootstrap.__version__,
            "components": {
                "memory": runtime.memory_manager is not None if runtime else False,
                "tools": runtime.tool_executor is not None if runtime else False,
                "proactive": runtime.proactive_engine is not None if runtime else False,
                "automations": runtime.automation_engine is not None if runtime else False,
            },
        }
    )


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """WebSocket endpoint for the browser-based WebChat channel."""
    await app.state.webchat.handle_websocket(websocket)


def serve() -> None:
    """Launch the dashboard server."""
    from animus_bootstrap.config import ConfigManager

    cfg = ConfigManager().load()
    port = cfg.services.port
    uvicorn.run(
        "animus_bootstrap.dashboard.app:app",
        host="0.0.0.0",  # noqa: S104
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    serve()
