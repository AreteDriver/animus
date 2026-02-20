"""Pydantic models for skill definitions (v1 + v2 schema support)."""

from __future__ import annotations

from pydantic import BaseModel, Field

# --- v2 sub-models ---


class RoutingExclusion(BaseModel):
    """A condition where a skill should NOT be used."""

    condition: str
    instead: str = ""
    reason: str = ""


class SkillRouting(BaseModel):
    """When to use / not use a skill."""

    use_when: list[str] = Field(default_factory=list)
    do_not_use_when: list[RoutingExclusion] = Field(default_factory=list)


class VerificationCheckpoint(BaseModel):
    """A trigger-action pair for runtime verification."""

    trigger: str
    action: str


class SkillVerification(BaseModel):
    """Pre/post conditions and checkpoints for a skill."""

    pre_conditions: list[str] = Field(default_factory=list)
    post_conditions: list[str] = Field(default_factory=list)
    checkpoints: list[VerificationCheckpoint] = Field(default_factory=list)
    completion_checklist: list[str] = Field(default_factory=list)


class EscalationRule(BaseModel):
    """How to handle a specific error class."""

    error_class: str
    description: str = ""
    action: str = "report"
    max_retries: int = 0
    fallback: str = ""


class SkillErrorHandling(BaseModel):
    """Error handling strategies for a skill."""

    escalation: list[EscalationRule] = Field(default_factory=list)
    self_correction: list[str] = Field(default_factory=list)


class ContractProvides(BaseModel):
    """An output this skill provides to other agents."""

    name: str
    type: str = "object"
    consumers: list[str] = Field(default_factory=list)
    description: str = ""


class ContractRequires(BaseModel):
    """An input this skill requires from another agent."""

    name: str
    type: str = "object"
    provider: str = ""
    description: str = ""


class SkillContracts(BaseModel):
    """Inter-agent contracts: what this skill provides and requires."""

    provides: list[ContractProvides] = Field(default_factory=list)
    requires: list[ContractRequires] = Field(default_factory=list)


# --- Core models ---


class SkillCapability(BaseModel):
    """A single capability within a skill."""

    name: str
    description: str = ""
    inputs: dict = Field(default_factory=dict)
    outputs: dict = Field(default_factory=dict)
    risk_level: str = "low"
    consensus_required: str = "any"
    requires_user_confirmation: bool = False
    side_effects: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    examples: list[dict] = Field(default_factory=list)
    # v2 fields
    parallel_safe: bool = True
    intent_required: bool = False
    post_execution: list[str] = Field(default_factory=list)


class SkillDefinition(BaseModel):
    """Full skill definition loaded from schema.yaml + SKILL.md."""

    name: str
    version: str = "1.0.0"
    agent: str
    description: str = ""
    capabilities: list[SkillCapability] = Field(default_factory=list)
    capability_names: list[str] = Field(default_factory=list)
    protected_paths: list[str] = Field(default_factory=list)
    blocked_patterns: list[str] = Field(default_factory=list)
    dependencies: dict = Field(default_factory=dict)
    status: str = "active"
    skill_doc: str = ""  # Contents of SKILL.md
    # v2 fields
    type: str = "agent"
    category: str = ""
    risk_level: str = "low"
    consensus_level: str = "any"
    trust: str = "supervised"
    parallel_safe: bool = False
    tools: list[str] = Field(default_factory=list)
    routing: SkillRouting | None = None
    verification: SkillVerification | None = None
    error_handling: SkillErrorHandling | None = None
    contracts: SkillContracts | None = None
    skill_inputs: dict = Field(default_factory=dict)
    skill_outputs: dict = Field(default_factory=dict)

    def get_capability(self, name: str) -> SkillCapability | None:
        """Get a capability by name."""
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        return None


class SkillRegistry(BaseModel):
    """Collection of loaded skills."""

    version: str = "1.0.0"
    skills: list[SkillDefinition] = Field(default_factory=list)
    categories: dict[str, str] = Field(default_factory=dict)
    consensus_levels: dict[str, dict] = Field(default_factory=dict)

    def get_skill(self, name: str) -> SkillDefinition | None:
        """Find a skill by name."""
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    def get_skills_for_agent(self, agent: str) -> list[SkillDefinition]:
        """Get all active skills assigned to an agent."""
        return [s for s in self.skills if s.agent == agent and s.status == "active"]
