"""Tests for SkillTuner."""

from animus_forge.skills.evolver.models import SkillChange, SkillMetrics
from animus_forge.skills.evolver.tuner import SkillTuner
from animus_forge.skills.models import (
    SkillCapability,
    SkillDefinition,
)


def _make_metrics(**overrides):
    defaults = dict(
        skill_name="test_skill",
        skill_version="1.0.0",
        period_start="2025-01-01",
        period_end="2025-01-31",
        total_invocations=100,
        success_count=50,
        failure_count=50,
        success_rate=0.50,
        avg_quality_score=0.6,
        avg_cost_usd=0.05,
        avg_latency_ms=500,
        total_cost_usd=5.0,
        computed_at="now",
    )
    defaults.update(overrides)
    return SkillMetrics(**defaults)


def _make_skill(**overrides):
    defaults = dict(
        name="test_skill",
        version="1.0.0",
        agent="builder",
        description="Test",
        consensus_level="any",
        capabilities=[SkillCapability(name="cap1")],
        capability_names=["cap1"],
    )
    defaults.update(overrides)
    return SkillDefinition(**defaults)


class TestTuneDecliningSkill:
    def test_no_tuning_when_high_success(self):
        tuner = SkillTuner()
        metrics = _make_metrics(success_rate=0.95, failure_count=0)
        assert tuner.tune_declining_skill("s", metrics) is None

    def test_consensus_escalation(self):
        tuner = SkillTuner()
        metrics = _make_metrics(success_rate=0.5, failure_count=50)
        change = tuner.tune_declining_skill("s", metrics)
        assert change is not None
        assert change.change_type == "tune"
        assert "consensus_escalation" in change.modifications

    def test_routing_exclusions_from_contexts(self):
        tuner = SkillTuner()
        metrics = _make_metrics(success_rate=0.5, failure_count=10)
        change = tuner.tune_declining_skill("s", metrics, failure_contexts=["timeout", "auth fail"])
        assert change is not None
        assert "add_routing_exclusions" in change.modifications
        assert len(change.modifications["add_routing_exclusions"]) == 2

    def test_escalation_rule_on_failures(self):
        tuner = SkillTuner()
        metrics = _make_metrics(success_rate=0.8, failure_count=5)
        change = tuner.tune_declining_skill("s", metrics)
        assert change is not None
        assert "add_escalation_rule" in change.modifications

    def test_version_bump(self):
        tuner = SkillTuner()
        metrics = _make_metrics(success_rate=0.5, failure_count=10)
        change = tuner.tune_declining_skill("s", metrics)
        assert change is not None
        assert change.new_version == "1.0.1"


class TestTuneCostAnomaly:
    def test_no_tuning_low_ratio(self):
        tuner = SkillTuner()
        assert tuner.tune_cost_anomaly("s", 1.2) is None

    def test_cost_flag(self):
        tuner = SkillTuner()
        change = tuner.tune_cost_anomaly("s", 3.0)
        assert change is not None
        assert "cost_flag" in change.modifications
        assert change.modifications["cost_flag"]["ratio"] == 3.0

    def test_with_peer_skills(self):
        tuner = SkillTuner()
        change = tuner.tune_cost_anomaly("s", 3.0, peer_skills=["cheap_skill"])
        assert change is not None
        assert "add_routing_exclusions" in change.modifications


class TestApplyChangeToDefinition:
    def test_consensus_escalation(self):
        tuner = SkillTuner()
        skill = _make_skill(consensus_level="any")
        change = SkillChange(
            skill_name="test_skill",
            old_version="1.0.0",
            new_version="1.0.1",
            change_type="tune",
            modifications={"consensus_escalation": True},
        )
        new_skill = tuner.apply_change_to_definition(skill, change)
        assert new_skill.version == "1.0.1"
        assert new_skill.consensus_level == "majority"

    def test_escalation_to_unanimous(self):
        tuner = SkillTuner()
        skill = _make_skill(consensus_level="majority")
        change = SkillChange(
            skill_name="test_skill",
            old_version="1.0.0",
            new_version="1.0.1",
            change_type="tune",
            modifications={"consensus_escalation": True},
        )
        new_skill = tuner.apply_change_to_definition(skill, change)
        assert new_skill.consensus_level == "unanimous"

    def test_add_routing_exclusions(self):
        tuner = SkillTuner()
        skill = _make_skill()
        change = SkillChange(
            skill_name="test_skill",
            old_version="1.0.0",
            new_version="1.0.1",
            change_type="tune",
            modifications={
                "add_routing_exclusions": [{"condition": "timeout scenario", "reason": "observed"}]
            },
        )
        new_skill = tuner.apply_change_to_definition(skill, change)
        assert new_skill.routing is not None

    def test_add_escalation_rule(self):
        tuner = SkillTuner()
        skill = _make_skill()
        change = SkillChange(
            skill_name="test_skill",
            old_version="1.0.0",
            new_version="1.0.1",
            change_type="tune",
            modifications={
                "add_escalation_rule": {
                    "error_class": "timeout",
                    "action": "retry",
                    "max_retries": 3,
                }
            },
        )
        new_skill = tuner.apply_change_to_definition(skill, change)
        assert new_skill.error_handling is not None


class TestBumpVersion:
    def test_patch(self):
        assert SkillTuner.bump_version("1.0.0", "patch") == "1.0.1"

    def test_minor(self):
        assert SkillTuner.bump_version("1.2.3", "minor") == "1.3.0"

    def test_major(self):
        assert SkillTuner.bump_version("1.2.3", "major") == "2.0.0"

    def test_invalid_format(self):
        assert SkillTuner.bump_version("bad", "patch") == "1.0.1"

    def test_default_is_patch(self):
        assert SkillTuner.bump_version("1.0.0") == "1.0.1"
