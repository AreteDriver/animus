"""Tests for the governance policy engine."""

from __future__ import annotations

import pytest

from animus_forge.governance.policy import Policy, PolicyEngine, PolicyRule, RuleEffect


class TestPolicyRule:
    def test_exact_match(self):
        rule = PolicyRule(
            name="block_sensitive",
            effect=RuleEffect.DENY,
            condition={"path": "/etc/passwd"},
        )
        assert rule.matches({"path": "/etc/passwd", "action": "read"})
        assert not rule.matches({"path": "/tmp/file", "action": "read"})

    def test_regex_match(self):
        rule = PolicyRule(
            name="block_etc",
            effect=RuleEffect.DENY,
            condition={"path": "re:^/etc/.*"},
        )
        assert rule.matches({"path": "/etc/shadow"})
        assert not rule.matches({"path": "/tmp/file"})

    def test_range_match(self):
        rule = PolicyRule(
            name="size_limit",
            effect=RuleEffect.DENY,
            condition={"size": (0, 1024)},
        )
        assert rule.matches({"size": 500})
        assert not rule.matches({"size": 2048})

    def test_list_containment(self):
        rule = PolicyRule(
            name="require_tag",
            effect=RuleEffect.ALLOW,
            condition={"tags": ["approved"]},
        )
        assert rule.matches({"tags": ["approved", "draft"]})
        assert not rule.matches({"tags": ["draft"]})

    def test_callable_predicate(self):
        rule = PolicyRule(
            name="custom_check",
            effect=RuleEffect.DENY,
            condition={"custom": lambda ctx: ctx.get("risk_score", 0) > 0.8},
        )
        assert rule.matches({"risk_score": 0.9})
        assert not rule.matches({"risk_score": 0.5})

    def test_multiple_conditions_all_must_match(self):
        rule = PolicyRule(
            name="multi",
            effect=RuleEffect.ALLOW,
            condition={"action": "write", "path": r"re:\.py$"},
        )
        assert rule.matches({"action": "write", "path": "/tmp/test.py"})
        assert not rule.matches({"action": "read", "path": "/tmp/test.py"})


class TestPolicy:
    def test_evaluate_first_matching_rule_wins(self):
        policy = Policy(
            name="file_access",
            rules=[
                PolicyRule(
                    name="allow_tmp",
                    effect=RuleEffect.ALLOW,
                    condition={"path": "re:^/tmp/"},
                    priority=1,
                ),
                PolicyRule(
                    name="deny_etc",
                    effect=RuleEffect.DENY,
                    condition={"path": "re:^/etc/"},
                    priority=2,
                ),
            ],
            default_effect=RuleEffect.DENY,
        )

        effect, rule, reason = policy.evaluate({"path": "/tmp/test.py"})
        assert effect == RuleEffect.ALLOW
        assert rule.name == "allow_tmp"

        effect, rule, reason = policy.evaluate({"path": "/etc/passwd"})
        assert effect == RuleEffect.DENY
        assert rule.name == "deny_etc"

    def test_default_effect_when_no_rule_matches(self):
        policy = Policy(
            name="default_deny",
            default_effect=RuleEffect.DENY,
            rules=[],
        )
        effect, rule, reason = policy.evaluate({"path": "/unknown"})
        assert effect == RuleEffect.DENY
        assert rule is None

    def test_serialize_roundtrip(self):
        policy = Policy(
            name="test",
            rules=[
                PolicyRule("r1", RuleEffect.ALLOW, {"action": "read"}, priority=1),
                PolicyRule("r2", RuleEffect.DENY, {"action": "delete"}, priority=2),
            ],
            default_effect=RuleEffect.REQUIRE_APPROVAL,
        )

        data = policy.to_dict()
        restored = Policy.from_dict(data)
        assert restored.name == policy.name
        assert restored.default_effect == policy.default_effect
        assert len(restored.rules) == 2


class TestPolicyEngine:
    def test_load_and_evaluate(self):
        engine = PolicyEngine()
        policy = Policy(
            name="file_access",
            rules=[
                PolicyRule("allow_tmp", RuleEffect.ALLOW, {"path": "re:^/tmp/"}, priority=1),
            ],
            default_effect=RuleEffect.DENY,
        )
        engine.load(policy)

        effect, rule, reason = engine.evaluate("file_access", {"path": "/tmp/test.py"})
        assert effect == RuleEffect.ALLOW

    def test_missing_policy_returns_deny(self):
        engine = PolicyEngine()
        effect, rule, reason = engine.evaluate("nonexistent", {"path": "/tmp"})
        assert effect == RuleEffect.DENY
        assert "not found" in reason

    def test_list_policies(self):
        engine = PolicyEngine()
        engine.load(Policy("p1"))
        engine.load(Policy("p2"))
        assert sorted(engine.list_policies()) == ["p1", "p2"]
