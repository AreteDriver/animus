"""Update management router."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import animus_bootstrap

router = APIRouter()


@router.get("/update")
async def update_page(request: Request) -> object:
    """Render the update management page."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "update.html",
        {
            "request": request,
            "current_version": animus_bootstrap.__version__,
        },
    )


@router.post("/update/apply")
async def apply_update() -> JSONResponse:
    """Placeholder for applying updates."""
    return JSONResponse(
        content={"status": "info", "message": "Update functionality coming soon"},
    )
