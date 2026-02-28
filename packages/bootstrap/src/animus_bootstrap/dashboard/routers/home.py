"""Home page router — system status overview."""

from __future__ import annotations

import time

import httpx
from fastapi import APIRouter, Request

from animus_bootstrap.config import ConfigManager

router = APIRouter()

_BOOT_TIME = time.monotonic()


def _format_uptime(seconds: float) -> str:
    """Format seconds into a human-readable uptime string."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _format_size(size_bytes: int | float) -> str:
    """Format bytes into a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _get_memory_size(runtime: object | None) -> str:
    """Return human-readable memory stats from the runtime."""
    if runtime is None or getattr(runtime, "memory_manager", None) is None:
        return "N/A"
    # Check the intelligence memory DB path from config
    config = getattr(runtime, "config", None)
    if config is None:
        return "Active"
    from pathlib import Path

    db_path = Path(config.intelligence.memory_db_path).expanduser()
    if db_path.is_file():
        return _format_size(db_path.stat().st_size)
    return "Active"


async def _check_forge_status(cfg_manager: ConfigManager) -> str:
    """Ping the Forge health endpoint."""
    cfg = cfg_manager.load()
    url = f"http://{cfg.forge.host}:{cfg.forge.port}/health"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return "connected"
    except httpx.HTTPError:
        return "disconnected"
    return "disconnected"


def _count_runtime_components(runtime: object | None) -> int:
    """Count how many runtime components are initialized."""
    if runtime is None:
        return 0
    count = 0
    for attr in ("memory_manager", "tool_executor", "proactive_engine", "automation_engine"):
        if getattr(runtime, attr, None) is not None:
            count += 1
    return count


@router.get("/")
async def home_page(request: Request) -> object:
    """Render the home page with system status."""
    templates = request.app.state.templates
    config_manager = ConfigManager()

    # Runtime status
    runtime = getattr(request.app.state, "runtime", None)
    runtime_started = runtime and getattr(runtime, "started", False)
    runtime_status = "running" if runtime_started else "stopped"
    component_count = _count_runtime_components(runtime)

    # Forge status
    forge_status = await _check_forge_status(config_manager)

    # Memory stats — check actual runtime memory manager
    memory_size = _get_memory_size(runtime)

    # Uptime
    uptime = _format_uptime(time.monotonic() - _BOOT_TIME)

    # Cognitive backend type
    cognitive_type = "none"
    if runtime_started and getattr(runtime, "cognitive_backend", None) is not None:
        cognitive_type = type(runtime.cognitive_backend).__name__.replace("Backend", "").lower()

    # Persona count
    persona_count = 0
    if runtime_started and getattr(runtime, "persona_engine", None) is not None:
        persona_count = runtime.persona_engine.persona_count

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "runtime_status": runtime_status,
            "forge_status": forge_status,
            "memory_size": memory_size,
            "uptime": uptime,
            "component_count": component_count,
            "cognitive_type": cognitive_type,
            "persona_count": persona_count,
        },
    )
