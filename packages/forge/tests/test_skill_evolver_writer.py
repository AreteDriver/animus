"""Tests for SkillWriter."""

import yaml

from animus_forge.skills.evolver.writer import SkillWriter
from animus_forge.skills.models import (
    RoutingExclusion,
    SkillCapability,
    SkillDefinition,
    SkillRouting,
)


def _make_skill(**overrides):
    defaults = dict(
        name="test_skill",
        version="1.0.0",
        agent="builder",
        description="A test skill",
        category="system",
        capabilities=[
            SkillCapability(name="cap1", description="Capability 1"),
        ],
        capability_names=["cap1"],
        status="active",
    )
    defaults.update(overrides)
    return SkillDefinition(**defaults)


class TestSkillToYaml:
    def test_basic_serialization(self, tmp_path):
        writer = SkillWriter(tmp_path)
        skill = _make_skill()
        yaml_str = writer.skill_to_yaml(skill)
        data = yaml.safe_load(yaml_str)
        assert data["name"] == "test_skill"
        assert data["version"] == "1.0.0"
        assert data["agent"] == "builder"

    def test_sorted_keys(self, tmp_path):
        writer = SkillWriter(tmp_path)
        skill = _make_skill()
        yaml_str = writer.skill_to_yaml(skill)
        keys = list(yaml.safe_load(yaml_str).keys())
        assert keys == sorted(keys)

    def test_includes_capabilities(self, tmp_path):
        writer = SkillWriter(tmp_path)
        skill = _make_skill()
        yaml_str = writer.skill_to_yaml(skill)
        data = yaml.safe_load(yaml_str)
        assert len(data["capabilities"]) == 1
        assert data["capabilities"][0]["name"] == "cap1"

    def test_includes_routing(self, tmp_path):
        routing = SkillRouting(
            use_when=["code review"],
            do_not_use_when=[RoutingExclusion(condition="trivial changes")],
        )
        writer = SkillWriter(tmp_path)
        skill = _make_skill(routing=routing)
        yaml_str = writer.skill_to_yaml(skill)
        data = yaml.safe_load(yaml_str)
        assert "routing" in data
        assert data["routing"]["use_when"] == ["code review"]

    def test_round_trip(self, tmp_path):
        writer = SkillWriter(tmp_path)
        skill = _make_skill()
        yaml_str = writer.skill_to_yaml(skill)
        data = yaml.safe_load(yaml_str)
        assert data["name"] == skill.name
        assert data["version"] == skill.version


class TestWriteSkill:
    def test_creates_directory(self, tmp_path):
        writer = SkillWriter(tmp_path)
        skill = _make_skill()
        path = writer.write_skill(skill, "system")
        assert path.exists()
        assert path.name == "schema.yaml"
        assert "system" in str(path)

    def test_file_content(self, tmp_path):
        writer = SkillWriter(tmp_path)
        skill = _make_skill()
        path = writer.write_skill(skill, "system")
        data = yaml.safe_load(path.read_text())
        assert data["name"] == "test_skill"

    def test_overwrite_existing(self, tmp_path):
        writer = SkillWriter(tmp_path)
        skill_v1 = _make_skill(version="1.0.0")
        writer.write_skill(skill_v1, "system")

        skill_v2 = _make_skill(version="2.0.0")
        path = writer.write_skill(skill_v2, "system")
        data = yaml.safe_load(path.read_text())
        assert data["version"] == "2.0.0"


class TestUpdateRegistry:
    def test_creates_registry(self, tmp_path):
        writer = SkillWriter(tmp_path)
        skill = _make_skill()
        writer.update_registry(skill, "system")
        registry_path = tmp_path / "registry.yaml"
        assert registry_path.exists()
        data = yaml.safe_load(registry_path.read_text())
        assert len(data["skills"]) == 1
        assert data["skills"][0]["name"] == "test_skill"

    def test_updates_existing(self, tmp_path):
        writer = SkillWriter(tmp_path)
        skill = _make_skill(version="1.0.0")
        writer.update_registry(skill, "system")

        skill_v2 = _make_skill(version="2.0.0")
        writer.update_registry(skill_v2, "system")

        data = yaml.safe_load((tmp_path / "registry.yaml").read_text())
        assert len(data["skills"]) == 1
        assert data["skills"][0]["version"] == "2.0.0"

    def test_appends_new_skill(self, tmp_path):
        writer = SkillWriter(tmp_path)
        writer.update_registry(_make_skill(name="a"), "system")
        writer.update_registry(_make_skill(name="b"), "system")

        data = yaml.safe_load((tmp_path / "registry.yaml").read_text())
        assert len(data["skills"]) == 2


class TestReadRawYaml:
    def test_reads_existing(self, tmp_path):
        writer = SkillWriter(tmp_path)
        skill = _make_skill()
        writer.write_skill(skill, "system")
        raw = writer.read_raw_yaml("test_skill")
        assert "test_skill" in raw

    def test_returns_empty_for_missing(self, tmp_path):
        writer = SkillWriter(tmp_path)
        assert writer.read_raw_yaml("nonexistent") == ""
