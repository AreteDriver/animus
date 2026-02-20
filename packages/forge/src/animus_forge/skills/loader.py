"""Load skill definitions from disk (v1 + v2 schema support)."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .models import (
    ContractProvides,
    ContractRequires,
    EscalationRule,
    RoutingExclusion,
    SkillCapability,
    SkillContracts,
    SkillDefinition,
    SkillErrorHandling,
    SkillRegistry,
    SkillRouting,
    SkillVerification,
    VerificationCheckpoint,
)

logger = logging.getLogger(__name__)


def _parse_capability_v1(cap_name: str, cap_data: dict) -> SkillCapability:
    """Parse a capability from v1 dict format."""
    return SkillCapability(
        name=cap_name,
        description=cap_data.get("description", ""),
        inputs=cap_data.get("inputs", {}),
        outputs=cap_data.get("outputs", {}),
        risk_level=cap_data.get("risk_level", "low"),
        consensus_required=cap_data.get("consensus_required", "any"),
        requires_user_confirmation=cap_data.get("requires_user_confirmation", False),
        side_effects=cap_data.get("side_effects", []),
        warnings=cap_data.get("warnings", []),
        examples=cap_data.get("examples", []),
    )


def _parse_capability_v2(cap_data: dict) -> SkillCapability:
    """Parse a capability from v2 list format.

    v2 schemas use 'risk' and 'consensus' shorthand field names.
    """
    return SkillCapability(
        name=cap_data.get("name", ""),
        description=cap_data.get("description", ""),
        inputs=cap_data.get("inputs", {}),
        outputs=cap_data.get("outputs", {}),
        risk_level=cap_data.get("risk", cap_data.get("risk_level", "low")),
        consensus_required=cap_data.get("consensus", cap_data.get("consensus_required", "any")),
        requires_user_confirmation=cap_data.get("requires_user_confirmation", False),
        side_effects=cap_data.get("side_effects", []),
        warnings=cap_data.get("warnings", []),
        examples=cap_data.get("examples", []),
        parallel_safe=cap_data.get("parallel_safe", True),
        intent_required=cap_data.get("intent_required", False),
        post_execution=cap_data.get("post_execution", []),
    )


def _parse_routing(raw: dict) -> SkillRouting | None:
    """Parse the routing section from a v2 schema."""
    routing_data = raw.get("routing")
    if not routing_data or not isinstance(routing_data, dict):
        return None
    exclusions = [
        RoutingExclusion(**exc)
        for exc in routing_data.get("do_not_use_when", [])
        if isinstance(exc, dict)
    ]
    return SkillRouting(
        use_when=routing_data.get("use_when", []),
        do_not_use_when=exclusions,
    )


def _parse_verification(raw: dict) -> SkillVerification | None:
    """Parse the verification section from a v2 schema."""
    ver_data = raw.get("verification")
    if not ver_data or not isinstance(ver_data, dict):
        return None
    checkpoints = [
        VerificationCheckpoint(**cp)
        for cp in ver_data.get("checkpoints", [])
        if isinstance(cp, dict)
    ]
    return SkillVerification(
        pre_conditions=ver_data.get("pre_conditions", []),
        post_conditions=ver_data.get("post_conditions", []),
        checkpoints=checkpoints,
        completion_checklist=ver_data.get("completion_checklist", []),
    )


def _parse_error_handling(raw: dict) -> SkillErrorHandling | None:
    """Parse the error_handling section from a v2 schema."""
    eh_data = raw.get("error_handling")
    if not eh_data or not isinstance(eh_data, dict):
        return None
    rules = [
        EscalationRule(**rule) for rule in eh_data.get("escalation", []) if isinstance(rule, dict)
    ]
    return SkillErrorHandling(
        escalation=rules,
        self_correction=eh_data.get("self_correction", []),
    )


def _parse_contracts(raw: dict) -> SkillContracts | None:
    """Parse the contracts section from a v2 schema."""
    contracts_data = raw.get("contracts")
    if not contracts_data or not isinstance(contracts_data, dict):
        return None
    provides = [
        ContractProvides(**p) for p in contracts_data.get("provides", []) if isinstance(p, dict)
    ]
    requires = [
        ContractRequires(**r) for r in contracts_data.get("requires", []) if isinstance(r, dict)
    ]
    return SkillContracts(provides=provides, requires=requires)


def load_skill(skill_dir: Path) -> SkillDefinition:
    """Load a single skill from a directory containing schema.yaml and optionally SKILL.md."""
    schema_path = skill_dir / "schema.yaml"
    if not schema_path.exists():
        raise FileNotFoundError(f"No schema.yaml in {skill_dir}")

    with open(schema_path) as f:
        raw = yaml.safe_load(f)

    # Parse capabilities — v1 uses dict, v2 uses list
    capabilities: list[SkillCapability] = []
    raw_caps = raw.get("capabilities", {})
    if isinstance(raw_caps, list):
        # v2 format: list of capability dicts
        for cap_data in raw_caps:
            if isinstance(cap_data, dict):
                capabilities.append(_parse_capability_v2(cap_data))
    elif isinstance(raw_caps, dict):
        # v1 format: dict keyed by capability name
        for cap_name, cap_data in raw_caps.items():
            if not isinstance(cap_data, dict):
                continue
            capabilities.append(_parse_capability_v1(cap_name, cap_data))

    # Read SKILL.md if present
    skill_doc = ""
    skill_md_path = skill_dir / "SKILL.md"
    if skill_md_path.exists():
        skill_doc = skill_md_path.read_text()

    # v2 uses 'name' at top level, v1 uses 'skill_name'
    name = raw.get("name", raw.get("skill_name", skill_dir.name))

    return SkillDefinition(
        name=name,
        version=raw.get("version", "1.0.0"),
        agent=raw.get("agent", "unknown"),
        description=raw.get("description", ""),
        capabilities=capabilities,
        capability_names=[c.name for c in capabilities],
        protected_paths=raw.get("protected_paths", []),
        blocked_patterns=raw.get("blocked_patterns", []),
        dependencies=raw.get("dependencies", {}),
        skill_doc=skill_doc,
        # v2 fields
        type=raw.get("type", "agent"),
        category=raw.get("category", ""),
        risk_level=raw.get("risk_level", "low"),
        consensus_level=raw.get("consensus_level", "any"),
        trust=raw.get("trust", "supervised"),
        parallel_safe=raw.get("parallel_safe", False),
        tools=raw.get("tools", []),
        routing=_parse_routing(raw),
        verification=_parse_verification(raw),
        error_handling=_parse_error_handling(raw),
        contracts=_parse_contracts(raw),
        skill_inputs=raw.get("inputs", {}),
        skill_outputs=raw.get("outputs", {}),
    )


def load_registry(skills_dir: Path) -> SkillRegistry:
    """Load the full skill registry from a directory containing registry.yaml."""
    registry_path = skills_dir / "registry.yaml"
    if not registry_path.exists():
        raise FileNotFoundError(f"No registry.yaml in {skills_dir}")

    with open(registry_path) as f:
        raw = yaml.safe_load(f)

    skills: list[SkillDefinition] = []
    for entry in raw.get("skills", []):
        skill_path = skills_dir / entry["path"]
        if not skill_path.exists():
            logger.warning("Skill path not found: %s", skill_path)
            continue
        try:
            skill = load_skill(skill_path)
            # Override status from registry if present
            skill.status = entry.get("status", "active")
            skills.append(skill)
        except Exception:
            logger.exception("Failed to load skill from %s", skill_path)

    return SkillRegistry(
        version=raw.get("version", "1.0.0"),
        skills=skills,
        categories=raw.get("categories", {}),
        consensus_levels=raw.get("consensus_levels", {}),
    )
