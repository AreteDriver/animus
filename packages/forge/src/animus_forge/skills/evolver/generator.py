"""Skill generation — creates new skills from capability gaps or templates."""

from __future__ import annotations

import logging

from animus_forge.skills.models import (
    SkillCapability,
    SkillDefinition,
    SkillRouting,
)

from .models import CapabilityGap

logger = logging.getLogger(__name__)


class SkillGenerator:
    """Generates new skill definitions from detected gaps or explicit templates.

    Generated skills use conservative defaults:
    - status = "experimental"
    - risk_level = "medium"
    - consensus_level = "majority"
    - trust = "supervised"
    """

    def generate_from_gap(self, gap: CapabilityGap) -> SkillDefinition:
        """Generate a skill definition from a detected capability gap.

        Args:
            gap: The capability gap to fill.

        Returns:
            A new ``SkillDefinition`` with experimental defaults.
        """
        name = self._gap_to_name(gap)
        agent = gap.suggested_agent or "system"
        category = gap.suggested_category or "generated"

        capability = SkillCapability(
            name=f"{name}_handler",
            description=gap.description,
            risk_level="medium",
            consensus_required="majority",
        )

        routing = SkillRouting(
            use_when=[gap.description],
        )

        return SkillDefinition(
            name=name,
            version="0.1.0",
            agent=agent,
            description=f"Auto-generated skill to address: {gap.description}",
            capabilities=[capability],
            capability_names=[capability.name],
            status="experimental",
            type="agent",
            category=category,
            risk_level="medium",
            consensus_level="majority",
            trust="supervised",
            routing=routing,
        )

    def generate_from_template(
        self,
        name: str,
        agent: str,
        category: str,
        capabilities: list[str],
        description: str = "",
    ) -> SkillDefinition:
        """Generate a skill from explicit parameters.

        Args:
            name: Skill name.
            agent: Agent role to assign.
            category: Skill category.
            capabilities: List of capability names.
            description: Skill description.

        Returns:
            A new ``SkillDefinition`` with experimental defaults.
        """
        caps = [
            SkillCapability(
                name=cap_name,
                description=f"Capability: {cap_name}",
                risk_level="medium",
                consensus_required="majority",
            )
            for cap_name in capabilities
        ]

        return SkillDefinition(
            name=name,
            version="0.1.0",
            agent=agent,
            description=description or f"Template-generated skill: {name}",
            capabilities=caps,
            capability_names=capabilities,
            status="experimental",
            type="agent",
            category=category,
            risk_level="medium",
            consensus_level="majority",
            trust="supervised",
        )

    @staticmethod
    def _gap_to_name(gap: CapabilityGap) -> str:
        """Derive a skill name from a capability gap.

        Uses the suggested agent or first significant word from description.
        """
        if gap.suggested_agent:
            base = gap.suggested_agent.replace(" ", "_").lower()
        else:
            words = [w.lower() for w in gap.description.split() if len(w) >= 4]
            base = "_".join(words[:3]) if words else "generated_skill"
        return f"{base}_auto"
