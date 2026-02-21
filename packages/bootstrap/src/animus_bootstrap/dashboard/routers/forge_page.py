"""Forge status dashboard router."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


def _get_runtime(request: Request) -> object | None:
    """Safely retrieve the runtime from app state."""
    return getattr(request.app.state, "runtime", None)


@router.get("/forge")
async def forge_page(request: Request) -> object:
    """Render the Forge status page."""
    templates = request.app.state.templates

    forge_status = "unknown"
    forge_health: dict = {}
    forge_enabled = False

    runtime = _get_runtime(request)
    if runtime is not None:
        cfg = getattr(runtime, "config", None)
        if cfg is not None:
            forge_cfg = getattr(cfg, "forge", None)
            if forge_cfg is not None:
                forge_enabled = getattr(forge_cfg, "enabled", False)

    # Probe Forge health if enabled
    if forge_enabled:
        try:
            import httpx

            host = getattr(cfg.forge, "host", "127.0.0.1")
            port = getattr(cfg.forge, "port", 8000)
            url = f"http://{host}:{port}/health"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    forge_status = "running"
                    forge_health = resp.json()
                else:
                    forge_status = "error"
                    forge_health = {"error": f"HTTP {resp.status_code}"}
        except ImportError:
            forge_status = "unknown"
            forge_health = {"error": "httpx not installed"}
        except Exception:
            forge_status = "stopped"

    return templates.TemplateResponse(
        "forge.html",
        {
            "request": request,
            "forge_enabled": forge_enabled,
            "forge_status": forge_status,
            "forge_health": forge_health,
        },
    )
