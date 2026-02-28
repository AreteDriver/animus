"""Channel management page router — view and toggle communication channels."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Channel registry — defines all known channels and their visual identity.
# "enabled" is toggled at runtime via the dashboard.
CHANNEL_REGISTRY: list[dict[str, str | bool]] = [
    {
        "name": "webchat",
        "display_name": "WebChat",
        "color": "#00ff88",
        "icon": "💬",
        "enabled": True,
        "description": "Browser-based chat via WebSocket",
    },
    {
        "name": "telegram",
        "display_name": "Telegram",
        "color": "#0088cc",
        "icon": "✈",
        "enabled": False,
        "description": "Telegram Bot API integration",
    },
    {
        "name": "discord",
        "display_name": "Discord",
        "color": "#5865F2",
        "icon": "🎮",
        "enabled": False,
        "description": "Discord bot via discord.py",
    },
    {
        "name": "slack",
        "display_name": "Slack",
        "color": "#4A154B",
        "icon": "📋",
        "enabled": False,
        "description": "Slack workspace integration",
    },
    {
        "name": "matrix",
        "display_name": "Matrix",
        "color": "#0DBD8B",
        "icon": "🔗",
        "enabled": False,
        "description": "Matrix/Element federation",
    },
]


def get_channel_registry() -> list[dict[str, str | bool]]:
    """Return the module-level channel registry (test-patchable seam)."""
    return CHANNEL_REGISTRY


@router.get("/channels")
async def channels_page(request: Request) -> object:
    """Render the channel management page."""
    templates = request.app.state.templates
    channels = get_channel_registry()

    return templates.TemplateResponse(
        "channels.html",
        {
            "request": request,
            "channels": channels,
        },
    )


@router.post("/channels/{channel_name}/toggle")
async def toggle_channel(channel_name: str) -> RedirectResponse:
    """Toggle the enabled/disabled state of a channel.

    Args:
        channel_name: The channel identifier to toggle.
    """
    channels = get_channel_registry()
    for ch in channels:
        if ch["name"] == channel_name:
            ch["enabled"] = not ch["enabled"]
            logger.info("Channel %s toggled to %s", ch["name"], ch["enabled"])
            break

    return RedirectResponse(url="/channels", status_code=303)
