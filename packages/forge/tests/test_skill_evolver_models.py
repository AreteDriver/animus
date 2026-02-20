"""Tests for skill evolver frozen dataclass models."""

from animus_forge.skills.evolver.models import (
    CapabilityGap,
    DeprecationRecord,
    ExperimentConfig,
    ExperimentResult,
    SkillChange,
    SkillMetrics,
)


class TestSkillMetrics:
    def test_defaults(self):
        m = SkillMetrics(skill_name="s", skill_version="1.0.0", period_start="a", period_end="b")
        assert m.total_invocations == 0
        assert m.success_rate == 0.0
        assert m.trend == "stable"
        assert m.computed_at  # auto-set

    def test_all_fields(self):
        m = SkillMetrics(
            skill_name="test",
            skill_version="2.0.0",
            period_start="2025-01-01",
            period_end="2025-01-31",
            total_invocations=100,
            success_count=90,
            failure_count=10,
            success_rate=0.9,
            avg_quality_score=0.85,
            avg_cost_usd=0.05,
            avg_latency_ms=500,
            total_cost_usd=5.0,
            trend="improving",
            computed_at="2025-02-01",
        )
        assert m.skill_name == "test"
        assert m.total_invocations == 100
        assert m.trend == "improving"
        assert m.computed_at == "2025-02-01"

    def test_frozen(self):
        m = SkillMetrics(skill_name="s", skill_version="1", period_start="a", period_end="b")
        try:
            m.skill_name = "x"
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_equality(self):
        a = SkillMetrics(
            skill_name="s", skill_version="1", period_start="a", period_end="b", computed_at="c"
        )
        b = SkillMetrics(
            skill_name="s", skill_version="1", period_start="a", period_end="b", computed_at="c"
        )
        assert a == b


class TestSkillChange:
    def test_defaults(self):
        c = SkillChange(
            skill_name="s", old_version="1.0.0", new_version="1.0.1", change_type="tune"
        )
        assert c.description == ""
        assert c.diff == ""
        assert c.modifications == {}

    def test_with_modifications(self):
        mods = {"consensus_escalation": True}
        c = SkillChange(
            skill_name="s",
            old_version="1.0.0",
            new_version="1.0.1",
            change_type="tune",
            description="test",
            modifications=mods,
        )
        assert c.modifications["consensus_escalation"] is True

    def test_frozen(self):
        c = SkillChange(skill_name="s", old_version="1", new_version="2", change_type="tune")
        try:
            c.skill_name = "x"
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestExperimentConfig:
    def test_defaults(self):
        ec = ExperimentConfig(
            experiment_id="e1",
            skill_name="s",
            control_version="1.0.0",
            variant_version="1.0.1",
        )
        assert ec.traffic_split == 0.5
        assert ec.min_invocations == 100
        assert ec.start_date  # auto-set

    def test_explicit_start_date(self):
        ec = ExperimentConfig(
            experiment_id="e1",
            skill_name="s",
            control_version="1",
            variant_version="2",
            start_date="2025-01-01",
        )
        assert ec.start_date == "2025-01-01"


class TestExperimentResult:
    def test_defaults(self):
        er = ExperimentResult(
            experiment_id="e1",
            skill_name="s",
            control_version="1",
            variant_version="2",
        )
        assert er.winner == ""
        assert er.control_metrics is None
        assert er.statistical_significance == 0.0

    def test_with_metrics(self):
        m = SkillMetrics(skill_name="s", skill_version="1", period_start="a", period_end="b")
        er = ExperimentResult(
            experiment_id="e1",
            skill_name="s",
            control_version="1",
            variant_version="2",
            control_metrics=m,
            winner="1",
        )
        assert er.control_metrics is not None
        assert er.winner == "1"


class TestDeprecationRecord:
    def test_defaults(self):
        dr = DeprecationRecord(skill_name="s")
        assert dr.status == "flagged"
        assert dr.flagged_at  # auto-set
        assert dr.reason == ""

    def test_all_fields(self):
        dr = DeprecationRecord(
            skill_name="s",
            status="deprecated",
            flagged_at="a",
            deprecated_at="b",
            reason="low quality",
            success_rate_at_flag=0.3,
            invocations_at_flag=50,
        )
        assert dr.deprecated_at == "b"
        assert dr.invocations_at_flag == 50


class TestCapabilityGap:
    def test_defaults(self):
        g = CapabilityGap(description="missing skill")
        assert g.failure_contexts == []
        assert g.confidence == 0.0

    def test_with_context(self):
        g = CapabilityGap(
            description="test",
            failure_contexts=["w1", "w2"],
            suggested_agent="builder",
            confidence=0.8,
        )
        assert len(g.failure_contexts) == 2
        assert g.suggested_agent == "builder"
