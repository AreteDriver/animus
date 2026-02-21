"""Configuration editor router."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from animus_bootstrap.config import ConfigManager

router = APIRouter()


def _mask_key(key: str) -> str:
    """Mask an API key, showing only the last 4 characters."""
    if not key or len(key) <= 4:
        return key
    return "*" * (len(key) - 4) + key[-4:]


@router.get("/config")
async def config_page(request: Request) -> object:
    """Render the configuration editor."""
    templates = request.app.state.templates
    config_manager = ConfigManager()
    cfg = config_manager.load()

    # Build masked view of config for display
    masked = {
        "animus": {
            "version": cfg.animus.version,
            "first_run": cfg.animus.first_run,
            "data_dir": cfg.animus.data_dir,
        },
        "api": {
            "anthropic_key": _mask_key(cfg.api.anthropic_key),
            "openai_key": _mask_key(cfg.api.openai_key),
        },
        "forge": {
            "enabled": cfg.forge.enabled,
            "host": cfg.forge.host,
            "port": cfg.forge.port,
            "api_key": _mask_key(cfg.forge.api_key),
        },
        "memory": {
            "backend": cfg.memory.backend,
            "path": cfg.memory.path,
            "max_context_tokens": cfg.memory.max_context_tokens,
        },
        "identity": {
            "name": cfg.identity.name,
            "timezone": cfg.identity.timezone,
            "locale": cfg.identity.locale,
        },
        "services": {
            "autostart": cfg.services.autostart,
            "port": cfg.services.port,
            "log_level": cfg.services.log_level,
            "update_check": cfg.services.update_check,
        },
    }

    return templates.TemplateResponse(
        "config.html",
        {
            "request": request,
            "config": masked,
            "saved": request.query_params.get("saved", ""),
        },
    )


@router.post("/config")
async def save_config(
    request: Request,
    anthropic_key: str = Form(""),
    openai_key: str = Form(""),
    forge_enabled: str = Form(""),
    forge_host: str = Form("localhost"),
    forge_port: int = Form(8000),
    forge_api_key: str = Form(""),
    memory_backend: str = Form("sqlite"),
    memory_path: str = Form("~/.local/share/animus/memory.db"),
    max_context_tokens: int = Form(100_000),
    identity_name: str = Form(""),
    identity_timezone: str = Form(""),
    identity_locale: str = Form(""),
    autostart: str = Form(""),
    services_port: int = Form(7700),
    log_level: str = Form("info"),
    update_check: str = Form(""),
    data_dir: str = Form("~/.local/share/animus"),
) -> RedirectResponse:
    """Accept form data, validate, and save configuration."""
    config_manager = ConfigManager()
    cfg = config_manager.load()

    # Only update API keys if they were actually changed (not masked)
    if anthropic_key and not anthropic_key.startswith("*"):
        cfg.api.anthropic_key = anthropic_key
    if openai_key and not openai_key.startswith("*"):
        cfg.api.openai_key = openai_key
    if forge_api_key and not forge_api_key.startswith("*"):
        cfg.forge.api_key = forge_api_key

    # Non-secret fields — always update
    cfg.forge.enabled = forge_enabled == "on"
    cfg.forge.host = forge_host
    cfg.forge.port = forge_port
    cfg.memory.backend = memory_backend
    cfg.memory.path = memory_path
    cfg.memory.max_context_tokens = max_context_tokens
    cfg.identity.name = identity_name
    cfg.identity.timezone = identity_timezone
    cfg.identity.locale = identity_locale
    cfg.services.autostart = autostart == "on"
    cfg.services.port = services_port
    cfg.services.log_level = log_level
    cfg.services.update_check = update_check == "on"
    cfg.animus.data_dir = data_dir

    config_manager.save(cfg)

    return RedirectResponse(url="/config?saved=1", status_code=303)
