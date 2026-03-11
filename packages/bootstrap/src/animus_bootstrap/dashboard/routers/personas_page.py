"""Persona management page router with CRUD API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

router = APIRouter()


class PersonaCreateRequest(BaseModel):
    """Request body for creating a persona."""

    name: str
    description: str = ""
    system_prompt: str = ""
    tone: str = "balanced"
    knowledge_domains: list[str] = []
    excluded_topics: list[str] = []
    channel_bindings: dict[str, bool] = {}
    is_default: bool = False


class PersonaUpdateRequest(BaseModel):
    """Request body for updating a persona."""

    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    tone: str | None = None
    knowledge_domains: list[str] | None = None
    excluded_topics: list[str] | None = None
    channel_bindings: dict[str, bool] | None = None
    active: bool | None = None
    is_default: bool | None = None


def _get_engine(request: Request):
    """Get persona engine from runtime, raising 503 if unavailable."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime and hasattr(runtime, "persona_engine") and runtime.persona_engine:
        return runtime.persona_engine
    raise HTTPException(status_code=503, detail="Persona engine not available")


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


@router.get("/api/personas")
async def list_personas(request: Request) -> list[dict]:
    """List all personas as JSON."""
    engine = _get_engine(request)
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "tone": p.voice.tone,
            "active": p.active,
            "is_default": p.is_default,
            "knowledge_domains": p.knowledge_domains,
            "excluded_topics": p.excluded_topics,
            "channel_bindings": p.channel_bindings,
        }
        for p in engine.list_personas()
    ]


@router.post("/api/personas")
async def create_persona(request: Request, body: PersonaCreateRequest) -> dict:
    """Create a new persona."""
    from animus_bootstrap.personas.engine import PersonaProfile
    from animus_bootstrap.personas.voice import VoiceConfig

    engine = _get_engine(request)

    valid_tones = ("formal", "casual", "technical", "mentor", "creative", "balanced")
    if body.tone not in valid_tones:
        msg = f"Invalid tone. Must be: {', '.join(valid_tones)}"
        raise HTTPException(status_code=400, detail=msg)

    persona = PersonaProfile(
        name=body.name,
        description=body.description or f"{body.name} persona",
        system_prompt=body.system_prompt or f"You are {body.name}, a personal AI assistant.",
        voice=VoiceConfig(tone=body.tone),
        knowledge_domains=body.knowledge_domains,
        excluded_topics=body.excluded_topics,
        channel_bindings=body.channel_bindings,
        is_default=body.is_default,
    )
    engine.register_persona(persona)
    return {"status": "created", "id": persona.id, "name": persona.name}


@router.patch("/api/personas/{persona_id}")
async def update_persona(request: Request, persona_id: str, body: PersonaUpdateRequest) -> dict:
    """Update an existing persona."""
    engine = _get_engine(request)

    persona = engine.get_persona(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    if body.name is not None:
        persona.name = body.name
    if body.description is not None:
        persona.description = body.description
    if body.system_prompt is not None:
        persona.system_prompt = body.system_prompt
    if body.tone is not None:
        valid_tones = ("formal", "casual", "technical", "mentor", "creative", "balanced")
        if body.tone not in valid_tones:
            msg = f"Invalid tone. Must be: {', '.join(valid_tones)}"
            raise HTTPException(status_code=400, detail=msg)
        persona.voice.tone = body.tone
    if body.knowledge_domains is not None:
        persona.knowledge_domains = body.knowledge_domains
    if body.excluded_topics is not None:
        persona.excluded_topics = body.excluded_topics
    if body.channel_bindings is not None:
        persona.channel_bindings = body.channel_bindings
    if body.active is not None:
        persona.active = body.active
    if body.is_default is not None:
        persona.is_default = body.is_default
        if body.is_default:
            engine.set_default(persona.id)

    engine.update_persona(persona)
    return {"status": "updated", "id": persona.id}


@router.delete("/api/personas/{persona_id}")
async def delete_persona(request: Request, persona_id: str) -> dict:
    """Delete a persona."""
    engine = _get_engine(request)

    persona = engine.get_persona(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    name = persona.name
    engine.unregister_persona(persona_id)
    return {"status": "deleted", "id": persona_id, "name": name}


@router.post("/api/personas/{persona_id}/set-default")
async def set_default_persona(request: Request, persona_id: str) -> dict:
    """Set a persona as the default."""
    engine = _get_engine(request)

    persona = engine.get_persona(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    engine.set_default(persona_id)
    return {"status": "default_set", "id": persona_id, "name": persona.name}
