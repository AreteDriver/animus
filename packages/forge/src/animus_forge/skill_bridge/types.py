"""Type aliases and dataclasses for SkillBridge.

Tier classification and resolved-skill payload returned by the resolver
to the executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Tier = Literal["persona", "agent", "workflow"]
RiskLevel = Literal["safe", "caution", "destructive"]

TIER_OUTPUT_CEILING: dict[str, int] = {
    "persona": 4096,
    "agent": 2048,
    "workflow": 0,
}


@dataclass(frozen=True)
class RegistryEntry:
    """One entry parsed from ai-skills registry.yaml."""

    name: str
    path: str
    tier: Tier
    description: str
    has_references: bool = False
    risk_level: RiskLevel | None = None


@dataclass(frozen=True)
class AgentSchema:
    """Parsed schema.yaml for an agent-tier skill.

    Only the fields SkillBridge consumes are modelled; extra YAML keys
    are tolerated and ignored.
    """

    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    risk_level: RiskLevel = "safe"
    has_output_schema: bool = False


@dataclass(frozen=True)
class ResolvedSkill:
    """A resolved ai-skills entry ready for execution."""

    name: str
    tier: Tier
    skill_md_path: Path
    risk_level: RiskLevel
    schema: AgentSchema | None = None
    registry_entry: RegistryEntry | None = None

    @property
    def from_registry(self) -> bool:
        return self.registry_entry is not None
