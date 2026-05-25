"""Tests for skill_registry — spec §4.7 + R6 + A3 + C4."""

from pathlib import Path

from animus_forge.agent.skill_registry import (
    DEFAULT_SKILLS_DIR,
    SkillRegistry,
    _first_heading,
    _parse_skill,
    load_skills,
)
from animus_forge.agent.types import Skill


def _write_skill(parent: Path, name: str, body: str, filename: str = "SKILL.md") -> Path:
    """Create a skill dir under `parent` with the given body."""
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / filename
    p.write_text(body, encoding="utf-8")
    return p


class TestParseSkillFrontmatter:
    def test_full_frontmatter(self, tmp_path: Path):
        path = _write_skill(
            tmp_path,
            "demo",
            "---\nname: demo-skill\ndescription: a demo\ninvokable: false\n---\n# Body\nHello\n",
        )
        skill = _parse_skill(path)
        assert skill.name == "demo-skill"
        assert skill.description == "a demo"
        assert skill.invokable is False
        assert "# Body" in skill.body

    def test_frontmatter_default_invokable_true(self, tmp_path: Path):
        path = _write_skill(tmp_path, "demo", "---\nname: x\ndescription: y\n---\nbody\n")
        assert _parse_skill(path).invokable is True

    def test_no_frontmatter_falls_back_to_dirname(self, tmp_path: Path):
        path = _write_skill(tmp_path, "my-skill", "# My Skill\n\nbody only\n")
        skill = _parse_skill(path)
        assert skill.name == "my-skill"
        assert skill.description == "My Skill"  # first heading
        assert skill.invokable is True

    def test_no_frontmatter_no_heading_uses_dirname_for_desc(self, tmp_path: Path):
        path = _write_skill(tmp_path, "bare", "just some content\nno headings\n")
        skill = _parse_skill(path)
        assert skill.name == "bare"
        assert skill.description == "bare"

    def test_invalid_yaml_falls_back(self, tmp_path: Path):
        """Malformed YAML must not crash — fall back to dirname/heading."""
        path = _write_skill(tmp_path, "broken", "---\nname: [unclosed\n---\n# Broken\nbody\n")
        skill = _parse_skill(path)
        assert skill.name == "broken"  # falls back to dirname

    def test_non_dict_yaml_falls_back(self, tmp_path: Path):
        """YAML scalar/list frontmatter must fall back gracefully."""
        path = _write_skill(tmp_path, "list", "---\n- a\n- b\n---\n# List Skill\n")
        skill = _parse_skill(path)
        assert skill.name == "list"
        assert skill.description == "List Skill"

    def test_empty_frontmatter_uses_dirname(self, tmp_path: Path):
        path = _write_skill(tmp_path, "empty", "---\n\n---\n# Empty Front\n")
        skill = _parse_skill(path)
        assert skill.name == "empty"


class TestFirstHeading:
    def test_first_heading_extracted(self):
        assert _first_heading("# Title\n\nbody") == "Title"

    def test_h2_also_works(self):
        assert _first_heading("body\n\n## Sub\n") == "Sub"

    def test_no_heading_returns_empty(self):
        assert _first_heading("no headings here") == ""

    def test_first_of_many(self):
        assert _first_heading("# First\n# Second\n") == "First"


class TestLoadSkills:
    def test_empty_dir_returns_empty(self, tmp_path: Path):
        registry, warnings = load_skills(tmp_path)
        assert registry == {}
        assert warnings == []

    def test_missing_dir_emits_warning(self, tmp_path: Path):
        nope = tmp_path / "missing"
        registry, warnings = load_skills(nope)
        assert registry == {}
        assert any("not found" in w for w in warnings)

    def test_loads_multiple_skills(self, tmp_path: Path):
        _write_skill(tmp_path, "a", "---\nname: a\ndescription: A\n---\nbody\n")
        _write_skill(tmp_path, "b", "---\nname: b\ndescription: B\n---\nbody\n")
        registry, warnings = load_skills(tmp_path)
        assert set(registry.keys()) == {"a", "b"}
        assert warnings == []

    def test_skill_md_lowercase_filename_accepted(self, tmp_path: Path):
        _write_skill(
            tmp_path,
            "lower",
            "---\nname: lower\ndescription: x\n---\nbody\n",
            filename="skill.md",
        )
        registry, _ = load_skills(tmp_path)
        assert "lower" in registry

    def test_dir_without_skill_md_warned(self, tmp_path: Path):
        (tmp_path / "no-skill").mkdir()
        (tmp_path / "no-skill" / "README.md").write_text("not a skill")
        registry, warnings = load_skills(tmp_path)
        assert registry == {}
        assert any("no SKILL.md" in w for w in warnings)

    def test_non_directory_entries_ignored(self, tmp_path: Path):
        (tmp_path / "loose.md").write_text("# Loose\n")
        registry, _ = load_skills(tmp_path)
        assert registry == {}

    def test_duplicate_name_later_wins_with_warning(self, tmp_path: Path):
        _write_skill(tmp_path, "first", "---\nname: dup\ndescription: A\n---\nbody1\n")
        _write_skill(tmp_path, "second", "---\nname: dup\ndescription: B\n---\nbody2\n")
        registry, warnings = load_skills(tmp_path)
        assert "dup" in registry
        # Sorted iteration → "first" loaded first, "second" overwrites
        assert registry["dup"].source_path.parent.name == "second"
        assert any("duplicate" in w for w in warnings)


class TestSkillRegistry:
    def test_list_is_sorted(self, tmp_path: Path):
        for name in ("zebra", "alpha", "mango"):
            _write_skill(
                tmp_path,
                name,
                f"---\nname: {name}\ndescription: x\n---\nbody\n",
            )
        reg = SkillRegistry(tmp_path)
        assert reg.list() == ["alpha", "mango", "zebra"]

    def test_get_returns_skill_or_none(self, tmp_path: Path):
        _write_skill(tmp_path, "x", "---\nname: x\ndescription: y\n---\nb\n")
        reg = SkillRegistry(tmp_path)
        assert isinstance(reg.get("x"), Skill)
        assert reg.get("missing") is None

    def test_len_and_contains(self, tmp_path: Path):
        _write_skill(tmp_path, "x", "---\nname: x\ndescription: y\n---\nb\n")
        reg = SkillRegistry(tmp_path)
        assert len(reg) == 1
        assert "x" in reg
        assert "missing" not in reg
        # __contains__ rejects non-str via isinstance guard
        assert 42 not in reg

    def test_warnings_property_returns_copy(self, tmp_path: Path):
        (tmp_path / "no-skill").mkdir()
        reg = SkillRegistry(tmp_path)
        w = reg.warnings
        w.append("mutated")
        assert "mutated" not in reg.warnings


class TestLiveSkillsDir:
    """Run against real `~/.claude/skills/` to satisfy C4 / A3."""

    def test_loads_all_installed_skills(self):
        """Per C4: registry MUST list >= 127 (snapshot 2026-05-23)."""
        if not DEFAULT_SKILLS_DIR.exists():
            import pytest

            pytest.skip(f"no skills dir at {DEFAULT_SKILLS_DIR}")
        reg = SkillRegistry()
        assert len(reg) >= 127, f"only loaded {len(reg)} skills (warnings: {reg.warnings[:5]})"

    def test_no_load_warnings_on_real_dir(self):
        """If this fails, a real skill regressed — investigate before raising."""
        if not DEFAULT_SKILLS_DIR.exists():
            import pytest

            pytest.skip(f"no skills dir at {DEFAULT_SKILLS_DIR}")
        reg = SkillRegistry()
        assert reg.warnings == [], f"unexpected warnings: {reg.warnings}"
