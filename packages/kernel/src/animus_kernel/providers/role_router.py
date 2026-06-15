"""Role-based routing layer for intelligent provider selection.

Wraps TierRouter with role-aware model selection, enabling
different agent roles (Builder, Planner, Tester, etc.) to route
to role-specialized models.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from animus_kernel.agents.supervisor import AgentRole

from .base import CompletionRequest, ModelTier
from .manager import ProviderManager
from .router import RoutingConfig, RoutingDecision, TierRouter

logger = logging.getLogger(__name__)


@dataclass
class RoleRoutingConfig:
    """Configuration for RoleRouter behavior."""

    # Offline (no cloud API keys): role → (provider_name, model)
    offline_defaults: dict[AgentRole, tuple[str, str | None]] = field(
        default_factory=lambda: {
            AgentRole.BUILDER: ("ollama", "hermes3:8b"),
            AgentRole.PLANNER: ("ollama", "qwen2.5:14b"),
        }
    )

    # Cloud (API keys present): role → (provider_name, model)
    cloud_defaults: dict[AgentRole, tuple[str, str | None]] = field(
        default_factory=lambda: {
            AgentRole.BUILDER: ("anthropic", None),
            AgentRole.PLANNER: ("anthropic", None),
            AgentRole.TESTER: ("anthropic", "claude-3-5-haiku-20241022"),
        }
    )


class RoleRouter:
    """Role-aware routing layer on top of TierRouter.

    Selects provider and model based on agent role, with automatic
    offline/cloud mode switching and fallback to tier-based routing
    for unmapped roles.
    """

    def __init__(
        self,
        provider_manager: ProviderManager,
        config: RoutingConfig | None = None,
        role_config: RoleRoutingConfig | None = None,
        budget_manager: object | None = None,
    ):
        self._tier_router = TierRouter(provider_manager, config, budget_manager)
        self._role_config = role_config or RoleRoutingConfig()
        # Snapshot env at init so route() is a pure dict lookup
        self._cloud_available = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self._mapping = (
            self._role_config.cloud_defaults
            if self._cloud_available
            else self._role_config.offline_defaults
        )

    def route(self, role: AgentRole, instruction: str) -> RoutingDecision:
        """Select provider and model for a given agent role.

        Args:
            role: Agent role
            instruction: Task instruction (used for fallback tier routing)

        Returns:
            RoutingDecision with provider and model
        """
        if role in self._mapping:
            provider_name, model = self._mapping[role]
            return RoutingDecision(
                provider_name=provider_name,
                model=model,
                model_tier=None,
                reason=f"role={role.value} → {provider_name}/{model or 'default'}",
            )

        # Fallback to existing TierRouter behavior
        request = CompletionRequest(
            prompt=instruction,
            model_tier=ModelTier.STANDARD,
        )
        return self._tier_router.route(request)
