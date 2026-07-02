"""Cloud fallback controller for Animus Head.

Manages escalation from local Ollama models to cloud providers
(Claude/Anthropic) when quality gates fail or the user explicitly
requests hybrid mode. Preserves full conversation continuity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from animus_kernel.providers.base import CompletionRequest, CompletionResponse, ProviderError
from animus_kernel.providers.manager import ProviderManager

logger = logging.getLogger(__name__)


@dataclass
class FallbackStatus:
    """Current fallback status for reporting."""

    enabled: bool = False
    configured: bool = False
    provider_name: str = ""
    fallbacks_this_session: int = 0
    max_fallbacks: int = 10
    last_reason: str = ""


class HeadFallbackController:
    """Manages cloud fallback with conversation continuity.

    Args:
        provider_manager: ProviderManager with registered providers
        fallback_provider: Name of cloud provider to use (default: "anthropic")
        enabled: Whether fallback is enabled at all
        max_fallbacks_per_session: Hard cap on cloud calls per session
    """

    def __init__(
        self,
        provider_manager: ProviderManager | None = None,
        fallback_provider: str = "anthropic",
        enabled: bool = False,
        max_fallbacks_per_session: int = 10,
    ) -> None:
        self._pm = provider_manager or ProviderManager()
        self.fallback_provider = fallback_provider
        self.enabled = enabled
        self.max_fallbacks = max_fallbacks_per_session
        self._fallbacks_used = 0
        self._last_reason = ""

    @property
    def status(self) -> FallbackStatus:
        """Current fallback status."""
        return FallbackStatus(
            enabled=self.enabled,
            configured=self.is_configured(),
            provider_name=self.fallback_provider,
            fallbacks_this_session=self._fallbacks_used,
            max_fallbacks=self.max_fallbacks,
            last_reason=self._last_reason,
        )

    def is_configured(self) -> bool:
        """Check if the fallback provider is available and healthy."""
        try:
            provider = self._pm.get(self.fallback_provider)
            if provider is None:
                return False
            return provider.is_configured()
        except Exception:
            return False

    def can_fallback(self) -> bool:
        """Check if fallback is currently possible."""
        if not self.enabled:
            return False
        if self._fallbacks_used >= self.max_fallbacks:
            return False
        return self.is_configured()

    def try_fallback(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        reason: str = "quality gate failure",
    ) -> CompletionResponse | None:
        """Attempt cloud fallback. Returns None if unavailable.

        Args:
            messages: Full conversation context
            tools: Available tool schemas
            model: Specific cloud model to use
            reason: Why fallback was triggered

        Returns:
            CompletionResponse from cloud provider, or None
        """
        if not self.can_fallback():
            logger.debug("Fallback not available: enabled=%s, used=%d/%d",
                        self.enabled, self._fallbacks_used, self.max_fallbacks)
            return None

        self._last_reason = reason

        try:
            request = CompletionRequest(
                prompt="",
                messages=messages,
                model=model,
                temperature=0.7,
                tools=tools,
                tool_choice="auto" if tools else None,
            )
            response = self._pm.complete(
                request, provider_name=self.fallback_provider, use_fallback=False
            )
            self._fallbacks_used += 1
            logger.info(
                "Fallback to %s succeeded (session: %d/%d). Reason: %s",
                self.fallback_provider,
                self._fallbacks_used,
                self.max_fallbacks,
                reason,
            )
            return response
        except ProviderError as exc:
            logger.warning("Fallback to %s failed: %s", self.fallback_provider, exc)
            return None
        except Exception:
            logger.exception("Unexpected error during fallback")
            return None

    def reset(self) -> None:
        """Reset fallback counters."""
        self._fallbacks_used = 0
        self._last_reason = ""
