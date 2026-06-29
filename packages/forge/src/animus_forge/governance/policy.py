"""Policy engine — rules, conditions, and policy containers.

A Policy is a named guardrail. Each Policy contains PolicyRules.
Each rule has a condition (evaluated against context) and an effect
(ALLOW, DENY, or REQUIRE_APPROVAL).

Rules are evaluated in order; the first matching rule wins.
If no rule matches, the default effect applies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuleEffect(str, Enum):
    """Outcome of a single rule evaluation."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class PolicyRule:
    """A single rule within a policy.

    Attributes:
        name: Human-readable rule identifier
        effect: ALLOW, DENY, or REQUIRE_APPROVAL
        condition: Dict of {field: pattern_or_value} that must ALL match
        description: Why this rule exists
        priority: Lower number = higher priority (evaluated first)
    """

    name: str
    effect: RuleEffect
    condition: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    priority: int = 100

    def matches(self, context: dict[str, Any]) -> bool:
        r"""Check if the rule's condition is satisfied by context.

        Supports:
        - Exact value equality: {"action": "file.write"}
        - Regex patterns: {"path": r"re:.*\\.py$"}
        - List containment: {"tags": ["sensitive"]}
        - Range checks: {"size": (0, 1024)}
        - Callable predicates: {"custom": lambda ctx: ...}
        """
        for key, expected in self.condition.items():
            actual = context.get(key)

            # Callable predicate
            if callable(expected):
                if not expected(context):
                    return False
                continue

            # Regex pattern (strings starting with "re:")
            if isinstance(expected, str) and expected.startswith("re:"):
                pattern = expected[3:]
                if actual is None or not re.search(pattern, str(actual)):
                    return False
                continue

            # Range tuple (min, max)
            if isinstance(expected, tuple) and len(expected) == 2:
                min_val, max_val = expected
                if actual is None or not (min_val <= actual <= max_val):
                    return False
                continue

            # List containment (expected is list → actual must contain all)
            if isinstance(expected, list):
                if actual is None:
                    return False
                actual_set = set(actual) if isinstance(actual, (list, tuple, set)) else {actual}
                if not all(e in actual_set for e in expected):
                    return False
                continue

            # Exact equality
            if actual != expected:
                return False

        return True


@dataclass
class Policy:
    """A named collection of rules for a specific domain.

    Attributes:
        name: Policy identifier (e.g. "file_access", "api_calls")
        rules: Ordered list of rules (sorted by priority ascending)
        default_effect: Effect when no rule matches
        description: Human-readable purpose
        version: SemVer string for tracking policy evolution
    """

    name: str
    rules: list[PolicyRule] = field(default_factory=list)
    default_effect: RuleEffect = RuleEffect.DENY
    description: str = ""
    version: str = "1.0.0"

    def evaluate(self, context: dict[str, Any]) -> tuple[RuleEffect, PolicyRule | None, str]:
        """Evaluate context against ordered rules.

        Returns:
            (effect, matching_rule, explanation)
        """
        sorted_rules = sorted(self.rules, key=lambda r: r.priority)
        for rule in sorted_rules:
            if rule.matches(context):
                return rule.effect, rule, f"Rule '{rule.name}' matched"
        return (
            self.default_effect,
            None,
            f"No rule matched; default effect = {self.default_effect.value}",
        )

    def add_rule(self, rule: PolicyRule) -> None:
        """Add a rule to the policy."""
        self.rules.append(rule)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Policy":
        """Load policy from a dictionary (e.g. parsed YAML)."""
        rules = []
        for r_data in data.get("rules", []):
            effect = RuleEffect(r_data.get("effect", "deny"))
            rules.append(
                PolicyRule(
                    name=r_data["name"],
                    effect=effect,
                    condition=r_data.get("condition", {}),
                    description=r_data.get("description", ""),
                    priority=r_data.get("priority", 100),
                )
            )
        return cls(
            name=data["name"],
            rules=rules,
            default_effect=RuleEffect(data.get("default_effect", "deny")),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "default_effect": self.default_effect.value,
            "rules": [
                {
                    "name": r.name,
                    "effect": r.effect.value,
                    "condition": r.condition,
                    "description": r.description,
                    "priority": r.priority,
                }
                for r in sorted(self.rules, key=lambda x: x.priority)
            ],
        }


class PolicyEngine:
    """Loads, caches, and queries policies from disk or memory."""

    def __init__(self):
        self._policies: dict[str, Policy] = {}

    def load(self, policy: Policy) -> None:
        """Register a policy in memory."""
        self._policies[policy.name] = policy

    def load_from_file(self, path: str) -> None:
        """Load policy from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        policy = Policy.from_dict(data)
        self.load(policy)

    def get(self, name: str) -> Policy | None:
        """Retrieve a policy by name."""
        return self._policies.get(name)

    def evaluate(self, policy_name: str, context: dict[str, Any]) -> tuple[RuleEffect, PolicyRule | None, str]:
        """Evaluate context against a named policy."""
        policy = self._policies.get(policy_name)
        if policy is None:
            return RuleEffect.DENY, None, f"Policy '{policy_name}' not found"
        return policy.evaluate(context)

    def list_policies(self) -> list[str]:
        """List all loaded policy names."""
        return list(self._policies.keys())
