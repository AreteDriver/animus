"""Memory browser router."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from animus_bootstrap.config import ConfigManager

router = APIRouter()


@router.get("/memory")
async def memory_page(request: Request) -> object:
    """Render the memory browser placeholder."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "memory.html",
        {"request": request},
    )


@router.get("/memory/export")
async def memory_export() -> JSONResponse:
    """Return config data as a JSON placeholder for memory export."""
    config_manager = ConfigManager()
    cfg = config_manager.load()
    return JSONResponse(content=cfg.model_dump())
