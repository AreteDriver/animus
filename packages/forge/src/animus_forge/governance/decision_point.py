"""Policy Decision Point (PDP) — the runtime guard between intent and execution.

Every action with external impact flows through the PDP:
    1. Classify the action (what domain? what sensitivity?)
    2. Load relevant policies from the PolicyEngine
    3. Evaluate rules against the action context
    4. Produce a Decision (allowed/denied/needs_approval + rationale)
    5. Log to AuditTrail

The PDP is the single place where governance decisions are made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .audit import AuditTrail
from .policy import PolicyEngine, RuleEffect


@dataclass
class Decision:
    """Outcome of a policy evaluation.

    Attributes:
        allowed: True if the action may proceed
        action: The action being evaluated (e.g. "file.write", "api.call")
        context: Full context that was evaluated
        reason: Human-readable explanation
        policy: Name of the policy that produced this decision
        rule: Specific rule that matched (if any)
        effect: Raw rule effect (allow/deny/require_approval)
        timestamp: When the decision was made
        request_id: Correlation ID for distributed tracing
        metadata: Extra fields (risk_score, suggested_alternatives, etc.)
    """

    allowed: bool
    action: str
    context: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    policy: str = ""
    rule: str | None = None
    effect: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_approval(self) -> bool:
        """True if the decision requires human sign-off."""
        return self.effect == RuleEffect.REQUIRE_APPROVAL.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "context": self.context,
            "reason": self.reason,
            "policy": self.policy,
            "rule": self.rule,
            "effect": self.effect,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "metadata": self.metadata,
        }


class DecisionPoint:
    """Policy Decision Point for the Animus governance plane.

    Evaluates actions against loaded policies and produces audited Decisions.

    Usage:
        pdp = DecisionPoint()
        pdp.engine.load_from_file("policies/file_access.yaml")

        decision = pdp.evaluate("file.write", {"path": "/tmp/test.py", "size": 1024})
        if decision.allowed:
            proceed()
        else:
            deny(decision.reason)
    """

    def __init__(
        self,
        engine: PolicyEngine | None = None,
        audit: AuditTrail | None = None,
    ):
        self.engine = engine or PolicyEngine()
        self.audit = audit or AuditTrail()

    def evaluate(
        self,
        action: str,
        context: dict[str, Any],
        request_id: str = "",
    ) -> Decision:
        """Evaluate an action against relevant policies.

        The action string is used to select which policy to apply.
        For example, "file.write" maps to the "file_access" policy.
        """
        # Map action to policy name (simple convention)
        policy_name = self._action_to_policy(action)

        effect, rule, reason = self.engine.evaluate(policy_name, context)

        allowed = effect == RuleEffect.ALLOW

        decision = Decision(
            allowed=allowed,
            action=action,
            context=context,
            reason=reason,
            policy=policy_name,
            rule=rule.name if rule else None,
            effect=effect.value,
            request_id=request_id,
        )

        self.audit.record(decision)
        return decision

    def _action_to_policy(self, action: str) -> str:
        """Map action string to policy name.

        Override this method for custom action → policy mappings.
        """
        mapping = {
            "file.write": "file_access",
            "file.delete": "file_access",
            "file.read": "file_access",
            "api.call": "api_usage",
            "api.external": "api_usage",
            "mcp.invoke": "mcp_tools",
            "mcp.read": "mcp_tools",
            "shell.exec": "command_execution",
            "git.push": "git_operations",
            "git.commit": "git_operations",
            "memory.store": "memory_operations",
            "memory.delete": "memory_operations",
        }
        return mapping.get(action, "default")

    @classmethod
    def from_file(cls, path: str) -> "DecisionPoint":
        """Create a DecisionPoint with policies loaded from a YAML file or directory.

        If path is a directory, loads all *.yaml files inside it.
        """
        import os

        pdp = cls()
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".yaml") or fname.endswith(".yml"):
                    pdp.engine.load_from_file(os.path.join(path, fname))
        else:
            pdp.engine.load_from_file(path)
        return pdp
