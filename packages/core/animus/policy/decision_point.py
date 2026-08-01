"""Deterministic Policy Decision Point (PDP).

Evaluates action requests against capability grants and policy rules.
Default deny. No model involvement.

Kernel-native policy layer (v2.3, 2026-07-06).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from animus.logging import get_logger

from .capability_store import CapabilityGrantStore

logger = get_logger("policy.decision_point")


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"


class DenialReason(str, Enum):
    MISSING_SCOPE = "missing_scope"
    STALE_APPROVAL = "stale_approval"
    UNKNOWN_SCHEMA = "unknown_schema"
    INVALID_TRANSITION = "invalid_transition"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CAPABILITY_REVOKED = "capability_revoked"
    POLICY_CONFLICT = "policy_conflict"
    ESCALATION_REQUIRED = "escalation_required"


@dataclass(frozen=True)
class PolicyResult:
    """Immutable result of a policy evaluation."""

    decision: Decision
    rule_id: str
    reason: str
    denial_reason_code: DenialReason | None = None
    confidence: float = 1.0
    obligations: list[dict[str, Any]] | None = None


class PolicyDecisionPoint:
    """Deterministic policy enforcement.

    Usage::

        pdp = PolicyDecisionPoint(capability_store)
        result = pdp.evaluate(
            principal="agent-researcher",
            action="delete",
            resource="mem-001",
            workspace_id="ws-test",
        )
        if result.decision != Decision.ALLOW:
            raise PermissionError(result.reason)
    """

    def __init__(self, capability_store: CapabilityGrantStore | None = None) -> None:
        self._store = capability_store or CapabilityGrantStore()

    def evaluate(
        self,
        principal: str,
        action: str,
        resource: str,
        workspace_id: str,
        schema_id: str | None = None,
        payload_summary: str | None = None,
    ) -> PolicyResult:
        """Evaluate a request. Returns a PolicyResult (never raises)."""
        now = datetime.now(timezone.utc)

        # 1. Collect grants for this principal
        grants = self._store.find_grants(principal, workspace_id)
        if not grants:
            return PolicyResult(
                decision=Decision.DENY,
                rule_id="rule-default-deny",
                reason="No capability grants found for principal",
                denial_reason_code=DenialReason.MISSING_SCOPE,
            )

        # 2. Check expiry and revocation
        active_grants = [
            g
            for g in grants
            if (g.expires_at is None or g.expires_at > now) and g.revoked_at is None
        ]
        if not active_grants:
            return PolicyResult(
                decision=Decision.DENY,
                rule_id="rule-expired-or-revoked",
                reason="All grants expired or revoked",
                denial_reason_code=DenialReason.CAPABILITY_REVOKED,
            )

        # 3. Check action permission
        action_allowed = any(action in g.action for g in active_grants)
        if not action_allowed:
            return PolicyResult(
                decision=Decision.DENY,
                rule_id="rule-action-not-permitted",
                reason=f"Action '{action}' not permitted by any active grant",
                denial_reason_code=DenialReason.MISSING_SCOPE,
            )

        # 4. Check budget (simple call counter)
        for g in active_grants:
            if g.budget and g.budget.get("max_calls"):
                # In a real system, this would query a counter service.
                # For the scaffold, we assume budgets are enforced elsewhere.
                pass

        # 5. Check schema restrictions
        if schema_id:
            schema_allowed = any(
                schema_id in (g.conditions.get("allowed_schemas", []) if g.conditions else [])
                for g in active_grants
            )
            if not schema_allowed and any(
                g.conditions and g.conditions.get("allowed_schemas") for g in active_grants
            ):
                return PolicyResult(
                    decision=Decision.DENY,
                    rule_id="rule-schema-restricted",
                    reason=f"Schema '{schema_id}' not in allowed_schemas",
                    denial_reason_code=DenialReason.UNKNOWN_SCHEMA,
                )

        # 6. High-risk actions escalate
        high_risk = {"delete", "execute", "delegate", "export"}
        if action in high_risk:
            return PolicyResult(
                decision=Decision.ESCALATE,
                rule_id="rule-high-risk-escalation",
                reason=f"Action '{action}' requires explicit approval",
                denial_reason_code=DenialReason.ESCALATION_REQUIRED,
                obligations=[
                    {
                        "obligation_type": "approve",
                        "description": f"Approve {action} on {resource}",
                        "target": "owner",
                    }
                ],
            )

        # 7. Allow
        return PolicyResult(
            decision=Decision.ALLOW,
            rule_id="rule-capability-grant",
            reason="Capability grant permits this action",
            confidence=1.0,
        )
