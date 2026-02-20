"""Skill tuning — generates SkillChange proposals from analysis findings."""

from __future__ import annotations

import logging
import re

from animus_forge.skills.models import (
    SkillDefinition,
)

from .models import SkillChange, SkillMetrics

logger = logging.getLogger(__name__)

# Consensus escalation order
_CONSENSUS_LEVELS = ["any", "majority", "unanimous"]


class SkillTuner:
    """Generates tuning proposals for underperforming skills.

    Applies heuristic rules to produce ``SkillChange`` objects that can be
    reviewed and applied through the evolver pipeline.
    """

    def tune_declining_skill(
        self,
        skill_name: str,
        metrics: SkillMetrics,
        failure_contexts: list[str] | None = None,
    ) -> SkillChange | None:
        """Propose tuning changes for a declining skill.

        Strategies:
        - Tighten consensus_level if quality is borderline (> 0.3 but < 0.7)
        - Add do_not_use_when entries from failure context patterns
        - Add error_handling escalation rules

        Args:
            skill_name: Name of the skill.
            metrics: Current metrics for the skill.
            failure_contexts: Optional list of failure context strings.

        Returns:
            A ``SkillChange`` or ``None`` if no tuning is warranted.
        """
        modifications: dict = {}
        reasons: list[str] = []

        # Tighten consensus if quality is borderline
        if 0.3 < metrics.success_rate < 0.7:
            modifications["consensus_escalation"] = True
            reasons.append(f"Escalating consensus (success rate {metrics.success_rate:.0%})")

        # Add routing exclusions from failure contexts
        if failure_contexts:
            exclusions = []
            for ctx in failure_contexts[:3]:  # Limit to top 3
                exclusions.append({"condition": ctx, "reason": "Observed failure pattern"})
            if exclusions:
                modifications["add_routing_exclusions"] = exclusions
                reasons.append(f"Adding {len(exclusions)} routing exclusion(s)")

        # Add escalation rule if none exists
        if metrics.failure_count > 0:
            modifications["add_escalation_rule"] = {
                "error_class": "execution_failure",
                "description": f"Auto-added: {metrics.failure_count} failures observed",
                "action": "escalate_to_supervisor",
                "max_retries": 2,
            }
            reasons.append("Adding escalation rule for execution failures")

        if not modifications:
            return None

        new_version = self.bump_version(metrics.skill_version or "1.0.0", "patch")

        return SkillChange(
            skill_name=skill_name,
            old_version=metrics.skill_version or "1.0.0",
            new_version=new_version,
            change_type="tune",
            description="; ".join(reasons),
            modifications=modifications,
        )

    def tune_cost_anomaly(
        self,
        skill_name: str,
        cost_ratio: float,
        peer_skills: list[str] | None = None,
    ) -> SkillChange | None:
        """Propose tuning for a skill with abnormally high costs.

        Adds routing exclusions pointing to cheaper alternatives.

        Args:
            skill_name: Name of the expensive skill.
            cost_ratio: How many times above average the cost is.
            peer_skills: Optional list of cheaper alternative skill names.

        Returns:
            A ``SkillChange`` or ``None`` if no tuning is warranted.
        """
        if cost_ratio < 1.5:
            return None

        modifications: dict = {}

        if peer_skills:
            exclusions = []
            for peer in peer_skills[:3]:
                exclusions.append(
                    {
                        "condition": f"Task can be handled by {peer}",
                        "instead": peer,
                        "reason": f"Cost optimization ({cost_ratio:.1f}x average)",
                    }
                )
            modifications["add_routing_exclusions"] = exclusions

        modifications["cost_flag"] = {
            "ratio": cost_ratio,
            "recommendation": "Consider cheaper provider/model",
        }

        return SkillChange(
            skill_name=skill_name,
            old_version="1.0.0",
            new_version="1.0.1",
            change_type="tune",
            description=f"Cost optimization: {cost_ratio:.1f}x fleet average",
            modifications=modifications,
        )

    def apply_change_to_definition(
        self,
        skill: SkillDefinition,
        change: SkillChange,
    ) -> SkillDefinition:
        """Apply a SkillChange to produce a new SkillDefinition.

        Args:
            skill: The original skill definition.
            change: The change to apply.

        Returns:
            New ``SkillDefinition`` with modifications applied and version bumped.
        """
        data = skill.model_dump()
        data["version"] = change.new_version
        mods = change.modifications

        # Escalate consensus
        if mods.get("consensus_escalation"):
            current = data.get("consensus_level", "any")
            idx = _CONSENSUS_LEVELS.index(current) if current in _CONSENSUS_LEVELS else 0
            if idx < len(_CONSENSUS_LEVELS) - 1:
                data["consensus_level"] = _CONSENSUS_LEVELS[idx + 1]

        # Add routing exclusions
        new_exclusions = mods.get("add_routing_exclusions", [])
        if new_exclusions:
            routing_data = data.get("routing") or {"use_when": [], "do_not_use_when": []}
            if isinstance(routing_data, dict):
                existing = routing_data.get("do_not_use_when", [])
                for exc in new_exclusions:
                    existing.append(exc)
                routing_data["do_not_use_when"] = existing
                data["routing"] = routing_data

        # Add escalation rule
        escalation_rule = mods.get("add_escalation_rule")
        if escalation_rule:
            eh_data = data.get("error_handling") or {"escalation": [], "self_correction": []}
            if isinstance(eh_data, dict):
                eh_data.setdefault("escalation", []).append(escalation_rule)
                data["error_handling"] = eh_data

        return SkillDefinition(**data)

    @staticmethod
    def bump_version(version: str, bump: str = "patch") -> str:
        """Bump a semver version string.

        Args:
            version: Current version (e.g. "1.2.3").
            bump: One of "major", "minor", "patch".

        Returns:
            Bumped version string.
        """
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
        if not match:
            return "1.0.1"

        major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))

        if bump == "major":
            return f"{major + 1}.0.0"
        elif bump == "minor":
            return f"{major}.{minor + 1}.0"
        else:
            return f"{major}.{minor}.{patch + 1}"
