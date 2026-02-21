"""Default configuration values for Animus Bootstrap."""

from __future__ import annotations

DEFAULT_CONFIG: dict[str, dict[str, object]] = {
    "animus": {
        "version": "0.1.0",
        "first_run": True,
        "data_dir": "~/.local/share/animus",
    },
    "api": {
        "anthropic_key": "",
        "openai_key": "",
    },
    "forge": {
        "enabled": False,
        "host": "localhost",
        "port": 8000,
        "api_key": "",
    },
    "memory": {
        "backend": "sqlite",
        "path": "~/.local/share/animus/memory.db",
        "max_context_tokens": 100_000,
    },
    "identity": {
        "name": "",
        "timezone": "",
        "locale": "",
    },
    "services": {
        "autostart": True,
        "port": 7700,
        "log_level": "info",
        "update_check": True,
    },
    "gateway": {
        "enabled": True,
        "default_backend": "anthropic",
        "system_prompt": "",
        "max_response_tokens": 4096,
    },
    "channels": {
        "webchat": {"enabled": True},
        "telegram": {"enabled": False, "bot_token": ""},
        "discord": {"enabled": False, "bot_token": "", "allowed_guilds": []},
        "slack": {"enabled": False, "bot_token": "", "app_token": ""},
        "matrix": {
            "enabled": False,
            "homeserver": "",
            "access_token": "",
            "room_ids": [],
        },
        "signal": {"enabled": False, "phone_number": ""},
        "whatsapp": {"enabled": False, "phone_number": ""},
        "email": {
            "enabled": False,
            "imap_host": "",
            "smtp_host": "",
            "username": "",
            "password": "",
            "poll_interval": 60,
        },
    },
    "intelligence": {
        "enabled": True,
        "memory_backend": "sqlite",
        "memory_db_path": "~/.local/share/animus/intelligence.db",
        "tool_approval_default": "auto",
        "max_tool_calls_per_turn": 5,
        "tool_timeout_seconds": 30,
        "mcp": {
            "config_path": "~/.config/animus/mcp.json",
            "auto_discover": True,
        },
    },
    "proactive": {
        "enabled": True,
        "quiet_hours_start": "22:00",
        "quiet_hours_end": "07:00",
        "timezone": "UTC",
        "checks": {},
    },
    "personas": {
        "enabled": True,
        "default_name": "Animus",
        "default_tone": "balanced",
        "default_max_response_length": "medium",
        "default_emoji_policy": "minimal",
        "default_system_prompt": "You are Animus, a personal AI assistant.",
        "profiles": {},
    },
}
