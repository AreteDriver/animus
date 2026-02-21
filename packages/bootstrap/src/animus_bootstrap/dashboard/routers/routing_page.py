"""Channel-to-persona routing page router."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

_CHANNELS = (
    "webchat",
    "telegram",
    "discord",
    "slack",
    "matrix",
    "signal",
    "whatsapp",
    "email",
)


@router.get("/routing", response_class=HTMLResponse)
async def routing_page(request: Request) -> HTMLResponse:
    """Channel-to-persona routing page."""
    templates = request.app.state.templates
    # Show which persona handles which channel
    routing: list[dict[str, str]] = []
    runtime = getattr(request.app.state, "runtime", None)
    if runtime and hasattr(runtime, "persona_engine") and runtime.persona_engine:
        for ch in _CHANNELS:
            assigned = "Default"
            for p in runtime.persona_engine.list_personas():
                if ch in p.channel_bindings and p.channel_bindings[ch]:
                    assigned = p.name
                    break
            routing.append({"channel": ch, "persona": assigned})
    else:
        routing = [{"channel": ch, "persona": "Default"} for ch in _CHANNELS]
    return templates.TemplateResponse("routing.html", {"request": request, "routing": routing})
