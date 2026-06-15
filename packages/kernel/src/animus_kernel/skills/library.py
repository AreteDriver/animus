"""Runtime API for querying skills."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .consensus import consensus_level_order
from .loader import load_registry
from .models import (
    SkillCapability,
    SkillDefinition,
    SkillRegistry,
)

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_DIR = Path(__file__).parent.parent.parent.parent / "skills"


def _tokenize(text: str) -> set[str]:
    """Extract lowercase word tokens (3+ chars) from text."""
    return {w.lower() for w in text.split() if len(w) >= 3}


class SkillLibrary:
    """Load and query skill definitions at runtime."""

    def __init__(self, skills_dir: Path | None = None, ab_manager: Any = None) -> None:
        self._skills_dir = skills_dir or _DEFAULT_SKILLS_DIR
        self._registry: SkillRegistry = load_registry(self._skills_dir)
        self._ab_manager = ab_manager  # Optional ABTestManager for A/B routing

    @property
    def registry(self) -> SkillRegistry:
        return self._registry

    def get_skill(self, name: str) -> SkillDefinition | None:
        return self._registry.get_skill(name)

    def get_skills_for_agent(self, agent: str) -> list[SkillDefinition]:
        return self._registry.get_skills_for_agent(agent)

    def get_capabilities(self, name: str) -> list[SkillCapability]:
        skill = self._registry.get_skill(name)
        if skill is None:
            return []
        return skill.capabilities

    def get_consensus_level(self, skill: str, capability: str) -> str | None:
        """Return the consensus level for a specific capability, or None if not found."""
        skill_def = self._registry.get_skill(skill)
        if skill_def is None:
            return None
        cap = skill_def.get_capability(capability)
        if cap is None:
            return None
        return cap.consensus_required

    def get_highest_consensus_for_role(
        self, role: str, role_skill_agents: dict[str, list[str]]
    ) -> str | None:
        """Return the highest consensus level across all capabilities for a role.

        Iterates the role's skill agents, collects all capabilities, and returns
        the maximum consensus_required value.  Returns None if the role has no
        mapped agents or no capabilities.
        """
        agents = role_skill_agents.get(role)
        if not agents:
            return None

        highest: int = -1
        highest_level: str | None = None

        for agent in agents:
            for skill in self.get_skills_for_agent(agent):
                for cap in skill.capabilities:
                    order = consensus_level_order(cap.consensus_required)
                    if order > highest:
                        highest = order
                        highest_level = cap.consensus_required
        return highest_level

    # --- v2 routing queries ---

    def find_skills_for_task(self, task_description: str) -> list[SkillDefinition]:
        """Find skills whose routing.use_when matches the task description.

        Uses keyword-overlap matching: a skill matches if any word token (3+ chars)
        from the task appears in any of its use_when phrases.
        """
        task_tokens = _tokenize(task_description)
        if not task_tokens:
            return []

        matches: list[SkillDefinition] = []
        for skill in self._registry.skills:
            if skill.status != "active" or not skill.routing:
                continue
            for phrase in skill.routing.use_when:
                phrase_tokens = _tokenize(phrase)
                if task_tokens & phrase_tokens:
                    matches.append(skill)
                    break
        return matches

    def get_routing_exclusions(
        self, task_description: str
    ) -> list[dict[str, str | SkillDefinition]]:
        """Return do_not_use_when exclusions that match the task description.

        Returns a list of dicts with keys: skill, exclusion (RoutingExclusion).
        """
        task_tokens = _tokenize(task_description)
        if not task_tokens:
            return []

        results: list[dict[str, str | SkillDefinition]] = []
        for skill in self._registry.skills:
            if skill.status != "active" or not skill.routing:
                continue
            for exc in skill.routing.do_not_use_when:
                exc_tokens = _tokenize(exc.condition)
                if task_tokens & exc_tokens:
                    results.append({"skill": skill, "exclusion": exc})
        return results

    def get_skill_consensus(self, skill_name: str, capability_name: str | None = None) -> str:
        """Get consensus level for a skill/capability.

        If capability_name is given and found, returns its consensus_required.
        Otherwise falls back to the skill-level consensus_level.
        Returns "any" if the skill is not found.
        """
        skill = self._registry.get_skill(skill_name)
        if skill is None:
            return "any"
        if capability_name:
            cap = skill.get_capability(capability_name)
            if cap is not None:
                return cap.consensus_required
        return skill.consensus_level

    # --- Context building ---

    def build_skill_context(self, agent: str) -> str:
        """Render skill docs into a prompt-injectable string for an agent."""
        skills = self.get_skills_for_agent(agent)
        if not skills:
            return ""

        sections: list[str] = [f"# Skills for {agent} agent\n"]
        for skill in skills:
            sections.append(f"## {skill.name} (v{skill.version})")
            sections.append(skill.description.strip())
            sections.append("")

            # Routing guidance (v2)
            if skill.routing:
                if skill.routing.use_when:
                    sections.append("### When to use")
                    for phrase in skill.routing.use_when:
                        sections.append(f"- {phrase}")
                    sections.append("")
                if skill.routing.do_not_use_when:
                    sections.append("### When NOT to use")
                    for exc in skill.routing.do_not_use_when:
                        line = f"- {exc.condition}"
                        if exc.instead:
                            line += f" → use {exc.instead}"
                        sections.append(line)
                    sections.append("")

            sections.append("### Capabilities")
            for cap in skill.capabilities:
                risk = cap.risk_level
                consensus = cap.consensus_required
                sections.append(
                    f"- **{cap.name}** — {cap.description} [risk={risk}, consensus={consensus}]"
                )
                # v2: inputs/outputs types
                if cap.inputs:
                    for inp_name, inp_spec in cap.inputs.items():
                        if isinstance(inp_spec, dict):
                            inp_type = inp_spec.get("type", "any")
                            req = " (required)" if inp_spec.get("required") else ""
                            sections.append(f"  - input: `{inp_name}`: {inp_type}{req}")
                if cap.outputs:
                    for out_name, out_spec in cap.outputs.items():
                        if isinstance(out_spec, dict):
                            out_type = out_spec.get("type", "any")
                            sections.append(f"  - output: `{out_name}`: {out_type}")
                # v2: post-execution guidance
                if cap.post_execution:
                    sections.append("  - post-execution:")
                    for step in cap.post_execution:
                        sections.append(f"    - {step}")

            if skill.protected_paths:
                sections.append("")
                sections.append("### Protected paths")
                for p in skill.protected_paths:
                    sections.append(f"- `{p}`")

            # v2: verification
            if skill.verification:
                if skill.verification.completion_checklist:
                    sections.append("")
                    sections.append("### Completion checklist")
                    for item in skill.verification.completion_checklist:
                        sections.append(f"- [ ] {item}")

            # v2: error handling
            if skill.error_handling and skill.error_handling.escalation:
                sections.append("")
                sections.append("### Error escalation")
                for rule in skill.error_handling.escalation:
                    sections.append(
                        f"- **{rule.error_class}**: {rule.action}"
                        + (f" (max {rule.max_retries} retries)" if rule.max_retries else "")
                    )

            sections.append("")

        return "\n".join(sections)

    def build_routing_summary(self) -> str:
        """Render a compact routing summary for the supervisor prompt.

        Lists each active skill with its use_when and do_not_use_when entries.
        """
        lines: list[str] = ["# Available Skills\n"]
        for skill in self._registry.skills:
            if skill.status != "active":
                continue
            lines.append(f"## {skill.name}")
            lines.append(f"{skill.description.strip()}")
            if skill.routing:
                if skill.routing.use_when:
                    lines.append("**Use when:** " + "; ".join(skill.routing.use_when))
                if skill.routing.do_not_use_when:
                    exclusions = [exc.condition for exc in skill.routing.do_not_use_when]
                    lines.append("**Do NOT use when:** " + "; ".join(exclusions))
            cap_names = [c.name for c in skill.capabilities]
            if cap_names:
                lines.append(f"**Capabilities:** {', '.join(cap_names)}")
            lines.append("")
        return "\n".join(lines)
