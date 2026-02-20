"""Tier-based routing layer for intelligent provider selection.

Wraps ProviderManager with model-tier routing, budget-aware switching,
and force-local mode for air-gapped deployments.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .base import (
    CompletionRequest,
    CompletionResponse,
    ModelTier,
    ProviderError,
    ProviderType,
)
from .manager import ProviderManager
from .ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)

# Provider types considered "local" (no network/cloud API calls)
_LOCAL_PROVIDER_TYPES = frozenset({ProviderType.OLLAMA})


class RoutingMode(Enum):
    """How the router selects between local and cloud providers."""

    CLOUD = "cloud"  # Only use cloud providers
    LOCAL = "local"  # Only use local providers (air-gapped)
    HYBRID = "hybrid"  # Route by tier: fast/embedding→local, reasoning→cloud


@dataclass
class RoutingConfig:
    """Configuration for TierRouter behavior."""

    mode: RoutingMode = RoutingMode.HYBRID

    # In HYBRID mode: token threshold below which fast/standard requests go local
    prefer_local_under_tokens: int = 2000

    # Per-tier provider preference override (tier name → list of provider names)
    # If not set, HYBRID defaults: REASONING→cloud, FAST/EMBEDDING→local, STANDARD→local-preferred
    tier_preferences: dict[str, list[str]] = field(default_factory=dict)

    # Budget threshold (tokens remaining) — below this, force local
    budget_force_local_threshold: int = 5000

    # Max routing decisions to keep in history
    history_limit: int = 200

    # Explicit fallback chain: ordered list of provider names to try on failure.
    # If empty, falls back to remaining registered providers in registration order.
    fallback_chain: list[str] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """Record of a single routing decision for observability."""

    provider_name: str
    model: str | None
    model_tier: ModelTier | None
    reason: str
    was_fallback: bool = False
    forced_local: bool = False
    timestamp: float = field(default_factory=time.time)


class TierRouter:
    """Intelligent routing layer on top of ProviderManager.

    Routes requests to the best provider/model based on:
    - Model tier (REASONING/STANDARD/FAST/EMBEDDING)
    - Routing mode (CLOUD/LOCAL/HYBRID)
    - Budget remaining (auto-switch to local when low)
    - Force-local toggle (for air-gapped deployments)
    """

    def __init__(
        self,
        provider_manager: ProviderManager,
        config: RoutingConfig | None = None,
        budget_manager: object | None = None,
    ):
        """Initialize router.

        Args:
            provider_manager: Underlying provider manager with registered providers
            config: Routing configuration
            budget_manager: Optional BudgetManager for budget-aware routing.
                            Must have a `remaining` property returning int.
        """
        self._pm = provider_manager
        self._config = config or RoutingConfig()
        self._budget = budget_manager
        self._force_local = False
        self._history: deque[RoutingDecision] = deque(maxlen=self._config.history_limit)

    @property
    def config(self) -> RoutingConfig:
        return self._config

    def force_local_only(self, enabled: bool = True) -> None:
        """Toggle air-gapped mode — only local providers will be used."""
        self._force_local = enabled
        logger.info("Force-local mode %s", "enabled" if enabled else "disabled")

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Route and execute a completion request.

        Selects the best provider based on tier/mode/budget, resolves the
        model, then delegates to ProviderManager.
        """
        decision = self._select_provider(request)
        self._history.append(decision)

        # Resolve tier to concrete model for Ollama providers
        resolved_model = self._resolve_tier_to_model(request, decision.provider_name)
        if resolved_model:
            request = CompletionRequest(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                model=resolved_model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop_sequences=request.stop_sequences,
                metadata=request.metadata,
                messages=request.messages,
                model_tier=request.model_tier,
                agent_id=request.agent_id,
                workflow_id=request.workflow_id,
            )
            decision.model = resolved_model

        try:
            response = self._pm.complete(
                request, provider_name=decision.provider_name, use_fallback=False
            )
            if not self._validate_response(response):
                logger.warning(
                    "Empty response from %s — attempting fallback",
                    decision.provider_name,
                )
                raise ProviderError(f"Empty response from {decision.provider_name}")
            return response
        except ProviderError:
            # Try fallback
            fallback = self._get_fallback(decision.provider_name, request)
            if fallback is None:
                raise
            fallback.was_fallback = True
            self._history.append(fallback)
            logger.info(
                "Falling back from %s to %s: %s",
                decision.provider_name,
                fallback.provider_name,
                fallback.reason,
            )
            return self._pm.complete(
                request, provider_name=fallback.provider_name, use_fallback=False
            )

    async def complete_async(self, request: CompletionRequest) -> CompletionResponse:
        """Async version of complete() with same routing logic."""
        decision = self._select_provider(request)
        self._history.append(decision)

        resolved_model = self._resolve_tier_to_model(request, decision.provider_name)
        if resolved_model:
            request = CompletionRequest(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                model=resolved_model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop_sequences=request.stop_sequences,
                metadata=request.metadata,
                messages=request.messages,
                model_tier=request.model_tier,
                agent_id=request.agent_id,
                workflow_id=request.workflow_id,
            )
            decision.model = resolved_model

        try:
            response = await self._pm.complete_async(
                request, provider_name=decision.provider_name, use_fallback=False
            )
            if not self._validate_response(response):
                logger.warning(
                    "Empty response from %s — attempting fallback",
                    decision.provider_name,
                )
                raise ProviderError(f"Empty response from {decision.provider_name}")
            return response
        except ProviderError:
            fallback = self._get_fallback(decision.provider_name, request)
            if fallback is None:
                raise
            fallback.was_fallback = True
            self._history.append(fallback)
            return await self._pm.complete_async(
                request, provider_name=fallback.provider_name, use_fallback=False
            )

    def get_routing_history(self, limit: int = 20) -> list[RoutingDecision]:
        """Return recent routing decisions for observability."""
        items = list(self._history)
        return items[-limit:]

    @staticmethod
    def _validate_response(response: CompletionResponse) -> bool:
        """Validate that a provider response is usable.

        Returns True if the response contains meaningful content.
        """
        if not response.content or not response.content.strip():
            return False
        return True

    # -- Private routing logic --

    def _select_provider(self, request: CompletionRequest) -> RoutingDecision:
        """Core routing: pick the best provider for this request."""
        tier = request.model_tier
        forced_local = self._force_local
        reason_parts: list[str] = []

        # Budget check: force local if remaining tokens are low
        if not forced_local and self._budget is not None:
            remaining = getattr(self._budget, "remaining", None)
            if remaining is not None and remaining < self._config.budget_force_local_threshold:
                forced_local = True
                reason_parts.append(f"budget low ({remaining} tokens remaining)")

        # Determine candidate set
        if forced_local or self._config.mode == RoutingMode.LOCAL:
            candidates = self._get_local_providers()
            if not reason_parts:
                reason_parts.append("force_local" if self._force_local else "mode=LOCAL")
        elif self._config.mode == RoutingMode.CLOUD:
            candidates = self._get_cloud_providers()
            reason_parts.append("mode=CLOUD")
        else:
            # HYBRID: route by tier
            candidates, tier_reason = self._hybrid_candidates(tier)
            reason_parts.append(tier_reason)

        # Tier preference overrides take priority over mode-based candidates
        if tier and tier.value in self._config.tier_preferences:
            pref = self._config.tier_preferences[tier.value]
            all_registered = set(self._pm.list_providers())
            preferred = [p for p in pref if p in all_registered]
            if preferred:
                candidates = preferred
                reason_parts.append(f"tier_preference override for {tier.value}")

        if not candidates:
            # Last resort: try anything registered
            candidates = self._pm.list_providers()
            reason_parts.append("no matching providers, trying all")

        if not candidates:
            raise ProviderError("No providers available for routing")

        provider_name = candidates[0]
        reason = "; ".join(reason_parts) + f" → {provider_name}"

        return RoutingDecision(
            provider_name=provider_name,
            model=request.model,
            model_tier=tier,
            reason=reason,
            forced_local=forced_local or self._force_local,
        )

    def _hybrid_candidates(self, tier: ModelTier | None) -> tuple[list[str], str]:
        """In HYBRID mode, decide local vs cloud based on tier."""
        local = self._get_local_providers()
        cloud = self._get_cloud_providers()

        if tier is None:
            # No tier specified — prefer default provider order
            return self._pm.list_providers(), "no tier specified, default order"

        if tier in (ModelTier.FAST, ModelTier.EMBEDDING):
            if local:
                return local, f"tier={tier.value} → local preferred"
            return cloud, f"tier={tier.value} → no local, fallback to cloud"

        if tier == ModelTier.REASONING:
            if cloud:
                return cloud, f"tier={tier.value} → cloud preferred"
            return local, f"tier={tier.value} → no cloud, fallback to local"

        # STANDARD: prefer local with cloud fallback
        if local:
            return local + cloud, f"tier={tier.value} → local preferred, cloud fallback"
        return cloud, f"tier={tier.value} → no local, cloud only"

    def _get_local_providers(self) -> list[str]:
        """Return names of registered local providers."""
        result = []
        for name in self._pm.list_providers():
            provider = self._pm.get(name)
            if provider and provider.provider_type in _LOCAL_PROVIDER_TYPES:
                result.append(name)
        return result

    def _get_cloud_providers(self) -> list[str]:
        """Return names of registered cloud providers."""
        result = []
        for name in self._pm.list_providers():
            provider = self._pm.get(name)
            if provider and provider.provider_type not in _LOCAL_PROVIDER_TYPES:
                result.append(name)
        return result

    def _resolve_tier_to_model(self, request: CompletionRequest, provider_name: str) -> str | None:
        """If the provider is Ollama and a tier is set, pick a concrete model."""
        if request.model:
            return None  # Explicit model overrides tier

        tier = request.model_tier
        if tier is None:
            return None

        provider = self._pm.get(provider_name)
        if not isinstance(provider, OllamaProvider):
            return None

        return provider.select_model_for_tier(tier)

    def _get_fallback(self, failed_name: str, request: CompletionRequest) -> RoutingDecision | None:
        """Find a fallback provider after the primary fails.

        Uses the explicit fallback_chain if configured, otherwise falls back
        to remaining registered providers in registration order.
        """
        registered = set(self._pm.list_providers())

        # Build candidate list: prefer explicit chain, then remaining providers
        if self._config.fallback_chain:
            candidates = [
                p for p in self._config.fallback_chain if p != failed_name and p in registered
            ]
        else:
            candidates = [p for p in self._pm.list_providers() if p != failed_name]

        # In force-local or LOCAL mode, restrict to local
        if self._force_local or self._config.mode == RoutingMode.LOCAL:
            candidates = [
                p
                for p in candidates
                if self._pm.get(p) and self._pm.get(p).provider_type in _LOCAL_PROVIDER_TYPES
            ]

        if not candidates:
            return None

        provider_name = candidates[0]
        return RoutingDecision(
            provider_name=provider_name,
            model=request.model,
            model_tier=request.model_tier,
            reason=f"fallback after {failed_name} failed",
        )
