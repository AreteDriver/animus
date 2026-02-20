"""Provider manager factory for the TUI."""

from __future__ import annotations

import logging

from animus_forge.config.settings import get_settings
from animus_forge.providers.base import ProviderType
from animus_forge.providers.manager import ProviderManager

logger = logging.getLogger(__name__)


def create_provider_manager() -> ProviderManager:
    """Create a ProviderManager wired from settings.

    Registers all providers whose API keys are configured.
    Sets Anthropic as default if available, else OpenAI, else Ollama.
    """
    settings = get_settings()
    manager = ProviderManager()

    # Register providers based on available keys
    registered: list[str] = []

    if settings.anthropic_api_key:
        try:
            manager.register(
                "anthropic",
                provider_type=ProviderType.ANTHROPIC,
                api_key=settings.anthropic_api_key,
            )
            registered.append("anthropic")
        except Exception as e:
            logger.warning(f"Failed to register Anthropic: {e}")

    if settings.openai_api_key:
        try:
            manager.register(
                "openai",
                provider_type=ProviderType.OPENAI,
                api_key=settings.openai_api_key,
            )
            registered.append("openai")
        except Exception as e:
            logger.warning(f"Failed to register OpenAI: {e}")

    # Ollama doesn't need an API key — try to register it always
    try:
        manager.register(
            "ollama",
            provider_type=ProviderType.OLLAMA,
        )
        registered.append("ollama")
    except Exception as e:
        logger.debug(f"Ollama not available: {e}")

    # Set default: anthropic > openai > ollama > first available
    default_set = False
    for preferred in ("anthropic", "openai", "ollama"):
        if preferred in registered:
            manager.set_default(preferred)
            default_set = True
            break
    if not default_set and registered:
        manager.set_default(registered[0])

    if registered:
        logger.info(f"Providers registered: {registered}, default: {manager._default_provider}")
    else:
        logger.warning("No AI providers configured. Set API keys in .env")

    return manager
