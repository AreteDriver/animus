"""Tests for SkillGenerator."""

from animus_forge.skills.evolver.generator import SkillGenerator
from animus_forge.skills.evolver.models import CapabilityGap


class TestGenerateFromGap:
    def test_basic(self):
        gen = SkillGenerator()
        gap = CapabilityGap(
            description="Missing code review capability",
            suggested_agent="reviewer",
            confidence=0.8,
        )
        skill = gen.generate_from_gap(gap)
        assert skill.status == "experimental"
        assert skill.agent == "reviewer"
        assert skill.risk_level == "medium"
        assert skill.consensus_level == "majority"
        assert skill.trust == "supervised"
        assert len(skill.capabilities) == 1

    def test_uses_suggested_agent(self):
        gen = SkillGenerator()
        gap = CapabilityGap(description="test", suggested_agent="deployer")
        skill = gen.generate_from_gap(gap)
        assert skill.agent == "deployer"

    def test_defaults_to_system_agent(self):
        gen = SkillGenerator()
        gap = CapabilityGap(description="test")
        skill = gen.generate_from_gap(gap)
        assert skill.agent == "system"

    def test_version_is_experimental(self):
        gen = SkillGenerator()
        gap = CapabilityGap(description="test")
        skill = gen.generate_from_gap(gap)
        assert skill.version == "0.1.0"

    def test_routing_from_gap_description(self):
        gen = SkillGenerator()
        gap = CapabilityGap(description="Handle database migrations")
        skill = gen.generate_from_gap(gap)
        assert skill.routing is not None
        assert "Handle database migrations" in skill.routing.use_when

    def test_category_from_gap(self):
        gen = SkillGenerator()
        gap = CapabilityGap(description="test", suggested_category="infra")
        skill = gen.generate_from_gap(gap)
        assert skill.category == "infra"

    def test_default_category(self):
        gen = SkillGenerator()
        gap = CapabilityGap(description="test")
        skill = gen.generate_from_gap(gap)
        assert skill.category == "generated"


class TestGenerateFromTemplate:
    def test_basic(self):
        gen = SkillGenerator()
        skill = gen.generate_from_template(
            name="custom_skill",
            agent="builder",
            category="system",
            capabilities=["build", "test"],
            description="Custom skill",
        )
        assert skill.name == "custom_skill"
        assert skill.agent == "builder"
        assert len(skill.capabilities) == 2
        assert skill.status == "experimental"

    def test_capability_names(self):
        gen = SkillGenerator()
        skill = gen.generate_from_template(
            name="s",
            agent="a",
            category="c",
            capabilities=["x", "y"],
        )
        assert skill.capability_names == ["x", "y"]

    def test_default_description(self):
        gen = SkillGenerator()
        skill = gen.generate_from_template(
            name="my_skill",
            agent="a",
            category="c",
            capabilities=["x"],
        )
        assert "my_skill" in skill.description


class TestGapToName:
    def test_with_agent(self):
        gap = CapabilityGap(description="test", suggested_agent="Builder")
        name = SkillGenerator._gap_to_name(gap)
        assert name == "builder_auto"

    def test_from_description(self):
        gap = CapabilityGap(description="Handle code review tasks")
        name = SkillGenerator._gap_to_name(gap)
        assert "auto" in name
        assert "handle" in name.lower()

    def test_fallback(self):
        gap = CapabilityGap(description="ab cd")  # short words
        name = SkillGenerator._gap_to_name(gap)
        assert name == "generated_skill_auto"
