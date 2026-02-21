"""Persona management page router."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/personas", response_class=HTMLResponse)
async def personas_page(request: Request) -> HTMLResponse:
    """Persona management page."""
    templates = request.app.state.templates
    runtime = getattr(request.app.state, "runtime", None)
    personas: list[dict[str, object]] = []
    if runtime and hasattr(runtime, "persona_engine") and runtime.persona_engine:
        for p in runtime.persona_engine.list_personas():
            personas.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "tone": p.voice.tone,
                    "active": p.active,
                    "is_default": p.is_default,
                    "domains": (
                        ", ".join(p.knowledge_domains) if p.knowledge_domains else "General"
                    ),
                    "channels": (", ".join(k for k, v in p.channel_bindings.items() if v) or "All"),
                }
            )
    return templates.TemplateResponse("personas.html", {"request": request, "personas": personas})
