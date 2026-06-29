"""Governance plane — policy engine, decision point, and audit trail.

The governance plane sits between intent and execution. Every action that
has external impact (file writes, API calls, MCP tool invocations) is
subjected to a Policy Decision Point (PDP) before proceeding.

Architecture:
    PolicyRule      → atomic guard (regex, threshold, allowlist)
    Policy          → named collection of rules for a domain
    PolicyEngine    → loads and caches policies
    DecisionPoint   → evaluates actions against policies
    AuditTrail      → records every decision with rationale

Usage:
    from animus_forge.governance import DecisionPoint, Policy

    pdp = DecisionPoint.from_file("policies/execution.yaml")
    decision = pdp.evaluate(
        action="file.write",
        context={"path": "/etc/passwd", "size": 1024},
    )
    if not decision.allowed:
        raise PermissionError(decision.reason)
"""

from __future__ import annotations

from .audit import AuditTrail, AuditEntry
from .decision_point import Decision, DecisionPoint
from .policy import Policy, PolicyEngine, PolicyRule, RuleEffect

__all__ = [
    "AuditEntry",
    "AuditTrail",
    "Decision",
    "DecisionPoint",
    "Policy",
    "PolicyEngine",
    "PolicyRule",
    "RuleEffect",
]
