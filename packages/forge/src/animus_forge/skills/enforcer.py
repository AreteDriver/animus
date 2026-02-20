"""Runtime enforcement of skill constraints on agent outputs."""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .library import SkillLibrary
    from .models import SkillDefinition

logger = logging.getLogger(__name__)

# Regex to extract path-like strings from output text
_PATH_RE = re.compile(r'(?:^|\s)([~/][^\s\'"`,;]+)', re.MULTILINE)


class EnforcementAction(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


class ViolationType(str, Enum):
    PROTECTED_PATH = "protected_path"
    BLOCKED_PATTERN = "blocked_pattern"


@dataclass
class Violation:
    type: ViolationType
    severity: str  # low, medium, high, critical
    message: str
    matched_text: str
    skill: str = ""


@dataclass
class EnforcementResult:
    action: EnforcementAction
    violations: list[Violation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.action == EnforcementAction.ALLOW

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "passed": self.passed,
            "violations": [
                {
                    "type": v.type.value,
                    "severity": v.severity,
                    "message": v.message,
                    "matched_text": v.matched_text,
                    "skill": v.skill,
                }
                for v in self.violations
            ],
        }


class SkillEnforcer:
    """Validate agent output against skill constraints."""

    def __init__(
        self,
        library: SkillLibrary,
        role_skill_agents: dict[str, list[str]] | None = None,
    ) -> None:
        self._library = library
        self._role_skill_agents = role_skill_agents or {}
        self._pattern_cache: dict[str, re.Pattern] = {}

    def check_output(self, role: str, output: str) -> EnforcementResult:
        """Check agent output for skill constraint violations.

        Args:
            role: The agent role (e.g. 'builder', 'devops').
            output: The text output from the agent.

        Returns:
            EnforcementResult with action and any violations found.
        """
        if not output:
            return EnforcementResult(action=EnforcementAction.ALLOW)

        agent_names = self._role_skill_agents.get(role, [])
        if not agent_names:
            return EnforcementResult(action=EnforcementAction.ALLOW)

        violations: list[Violation] = []
        for agent_name in agent_names:
            skills = self._library.get_skills_for_agent(agent_name)
            for skill in skills:
                violations.extend(self._check_protected_paths(skill, output))
                violations.extend(self._check_blocked_patterns(skill, output))

        if not violations:
            return EnforcementResult(action=EnforcementAction.ALLOW)

        # Determine action from worst violation
        has_block = any(v.type == ViolationType.BLOCKED_PATTERN for v in violations)
        action = EnforcementAction.BLOCK if has_block else EnforcementAction.WARN
        return EnforcementResult(action=action, violations=violations)

    def _check_protected_paths(self, skill: SkillDefinition, output: str) -> list[Violation]:
        if not skill.protected_paths:
            return []

        violations: list[Violation] = []
        paths_in_output = _PATH_RE.findall(output)

        for path_str in paths_in_output:
            for protected in skill.protected_paths:
                if fnmatch.fnmatch(path_str, protected) or path_str == protected:
                    violations.append(
                        Violation(
                            type=ViolationType.PROTECTED_PATH,
                            severity="high",
                            message=f"Output references protected path matching '{protected}'",
                            matched_text=path_str,
                            skill=skill.name,
                        )
                    )
                    break  # One violation per path is enough

        return violations

    def _check_blocked_patterns(self, skill: SkillDefinition, output: str) -> list[Violation]:
        if not skill.blocked_patterns:
            return []

        violations: list[Violation] = []
        for pattern_str in skill.blocked_patterns:
            compiled = self._get_compiled_pattern(pattern_str)
            if compiled is None:
                continue
            match = compiled.search(output)
            if match:
                violations.append(
                    Violation(
                        type=ViolationType.BLOCKED_PATTERN,
                        severity="critical",
                        message=f"Output contains blocked pattern '{pattern_str}'",
                        matched_text=match.group(),
                        skill=skill.name,
                    )
                )

        return violations

    def _get_compiled_pattern(self, pattern: str) -> re.Pattern | None:
        if pattern not in self._pattern_cache:
            try:
                self._pattern_cache[pattern] = re.compile(pattern)
            except re.error:
                logger.warning("Invalid blocked pattern regex: %s", pattern)
                self._pattern_cache[pattern] = None  # type: ignore[assignment]
        return self._pattern_cache[pattern]
