"""Tests for the Policy Decision Point."""

from __future__ import annotations

from pathlib import Path

import pytest

from animus_forge.governance import DecisionPoint, Policy, PolicyRule, RuleEffect
from animus_forge.governance.audit import AuditTrail


class TestDecisionPoint:
    def test_allow_action(self):
        pdp = DecisionPoint()
        pdp.engine.load(
            Policy(
                name="file_access",
                rules=[
                    PolicyRule("allow_tmp", RuleEffect.ALLOW, {"path": "re:^/tmp/"}, priority=1),
                ],
                default_effect=RuleEffect.DENY,
            )
        )

        decision = pdp.evaluate("file.write", {"path": "/tmp/test.py", "size": 1024})
        assert decision.allowed is True
        assert decision.policy == "file_access"
        assert decision.rule == "allow_tmp"

    def test_deny_action(self):
        pdp = DecisionPoint()
        pdp.engine.load(
            Policy(
                name="file_access",
                rules=[
                    PolicyRule("deny_etc", RuleEffect.DENY, {"path": "re:^/etc/"}, priority=1),
                ],
                default_effect=RuleEffect.ALLOW,
            )
        )

        decision = pdp.evaluate("file.write", {"path": "/etc/passwd"})
        assert decision.allowed is False
        assert decision.policy == "file_access"

    def test_require_approval(self):
        pdp = DecisionPoint()
        pdp.engine.load(
            Policy(
                name="api_usage",
                rules=[
                    PolicyRule(
                        name="approve_external",
                        effect=RuleEffect.REQUIRE_APPROVAL,
                        condition={"endpoint": r"re:api\.external\.com"},
                        priority=1,
                    ),
                ],
                default_effect=RuleEffect.ALLOW,
            )
        )

        decision = pdp.evaluate("api.call", {"endpoint": "api.external.com/v1"})
        assert decision.needs_approval is True
        assert decision.allowed is False  # REQUIRE_APPROVAL is not ALLOW

    def test_audit_trail_records_decisions(self):
        pdp = DecisionPoint()
        pdp.engine.load(
            Policy(
                name="file_access",
                rules=[
                    PolicyRule("allow_tmp", RuleEffect.ALLOW, {"path": "re:^/tmp/"}, priority=1),
                ],
                default_effect=RuleEffect.DENY,
            )
        )

        pdp.evaluate("file.write", {"path": "/tmp/test.py"})
        pdp.evaluate("file.write", {"path": "/etc/passwd"})

        entries = pdp.audit.entries()
        assert len(entries) == 2
        assert entries[0].action == "file.write"
        assert entries[1].action == "file.write"

    def test_action_to_policy_mapping(self):
        pdp = DecisionPoint()
        assert pdp._action_to_policy("file.write") == "file_access"
        assert pdp._action_to_policy("api.call") == "api_usage"
        assert pdp._action_to_policy("mcp.invoke") == "mcp_tools"
        assert pdp._action_to_policy("unknown.action") == "default"

    def test_request_id_propagation(self):
        pdp = DecisionPoint()
        pdp.engine.load(
            Policy(
                name="file_access",
                rules=[
                    PolicyRule("allow_tmp", RuleEffect.ALLOW, {"path": "re:^/tmp/"}, priority=1),
                ],
                default_effect=RuleEffect.DENY,
            )
        )

        decision = pdp.evaluate("file.write", {"path": "/tmp/test.py"}, request_id="req-123")
        assert decision.request_id == "req-123"
        assert pdp.audit.entries()[0].request_id == "req-123"


class TestDecisionPointFromFile:
    def test_loads_yaml_policy(self, tmp_path: Path):
        yaml_content = """
name: file_access
description: Test policy
version: "1.0.0"
default_effect: deny
rules:
  - name: allow_tmp
    effect: allow
    condition:
      path: "re:^/tmp/"
    priority: 1
"""
        policy_file = tmp_path / "file_access.yaml"
        policy_file.write_text(yaml_content)

        pdp = DecisionPoint.from_file(str(policy_file))
        decision = pdp.evaluate("file.write", {"path": "/tmp/test.py"})
        assert decision.allowed is True
