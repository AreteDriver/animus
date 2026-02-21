"""Tests for IdentityFileManager."""

from __future__ import annotations

import pytest

from animus_bootstrap.identity.manager import IdentityFileManager


@pytest.fixture()
def identity_dir(tmp_path):
    """Return a temp identity directory."""
    return tmp_path / "identity"


@pytest.fixture()
def manager(identity_dir):
    """Return an IdentityFileManager backed by tmp_path."""
    return IdentityFileManager(identity_dir)


class TestInit:
    """Initialization tests."""

    def test_creates_directory(self, identity_dir):
        assert not identity_dir.exists()
        IdentityFileManager(identity_dir)
        assert identity_dir.is_dir()

    def test_accepts_string_path(self, tmp_path):
        mgr = IdentityFileManager(str(tmp_path / "id"))
        assert mgr.identity_dir.exists()


class TestReadWrite:
    """Read/write operation tests."""

    def test_read_missing_returns_empty(self, manager):
        assert manager.read("IDENTITY.md") == ""

    def test_write_and_read(self, manager):
        manager.write("IDENTITY.md", "hello")
        assert manager.read("IDENTITY.md") == "hello"

    def test_write_locked_file_raises(self, manager):
        with pytest.raises(PermissionError, match="immutable"):
            manager.write("CORE_VALUES.md", "nope")

    def test_write_locked_method_works(self, manager):
        manager.write_locked("CORE_VALUES.md", "values here")
        assert manager.read("CORE_VALUES.md") == "values here"

    def test_write_unknown_file_raises(self, manager):
        with pytest.raises(ValueError, match="Unknown identity file"):
            manager.write("RANDOM.md", "bad")

    def test_read_unknown_file_raises(self, manager):
        with pytest.raises(ValueError, match="Unknown identity file"):
            manager.read("RANDOM.md")


class TestExists:
    """Existence check tests."""

    def test_exists_false_initially(self, manager):
        assert manager.exists("IDENTITY.md") is False

    def test_exists_after_write(self, manager):
        manager.write("IDENTITY.md", "content")
        assert manager.exists("IDENTITY.md") is True

    def test_exists_unknown_raises(self, manager):
        with pytest.raises(ValueError):
            manager.exists("BAD.md")


class TestReadAll:
    """read_all tests."""

    def test_read_all_empty(self, manager):
        result = manager.read_all()
        assert len(result) == len(IdentityFileManager.ALL_FILES)
        assert all(v == "" for v in result.values())

    def test_read_all_with_content(self, manager):
        manager.write("IDENTITY.md", "id content")
        manager.write_locked("CORE_VALUES.md", "values")
        result = manager.read_all()
        assert result["IDENTITY.md"] == "id content"
        assert result["CORE_VALUES.md"] == "values"


class TestAppendToLearned:
    """append_to_learned tests."""

    def test_append_creates_file(self, manager):
        manager.append_to_learned("Patterns", "User prefers concise answers")
        content = manager.read("LEARNED.md")
        assert "## Patterns" in content
        assert "User prefers concise answers" in content

    def test_append_to_existing_section(self, manager):
        manager.append_to_learned("Patterns", "first entry")
        manager.append_to_learned("Patterns", "second entry")
        content = manager.read("LEARNED.md")
        assert "first entry" in content
        assert "second entry" in content
        # Section header should appear only once
        assert content.count("## Patterns") == 1

    def test_append_new_section(self, manager):
        manager.append_to_learned("Patterns", "entry1")
        manager.append_to_learned("Preferences", "entry2")
        content = manager.read("LEARNED.md")
        assert "## Patterns" in content
        assert "## Preferences" in content

    def test_append_includes_timestamp(self, manager):
        manager.append_to_learned("Test", "timestamped")
        content = manager.read("LEARNED.md")
        # Format: [YYYY-MM-DD HH:MM UTC]
        assert "UTC]" in content

    def test_append_respects_line_cap(self, manager, identity_dir):
        # Pre-fill with many lines
        learned_path = identity_dir / "LEARNED.md"
        learned_path.write_text("## Old\n" + "- line\n" * 600, encoding="utf-8")
        manager.append_to_learned("Old", "new entry")
        content = manager.read("LEARNED.md")
        assert len(content.splitlines()) <= 500


class TestGetIdentityPrompt:
    """get_identity_prompt assembly tests."""

    def test_empty_prompt(self, manager):
        assert manager.get_identity_prompt() == ""

    def test_full_assembly_order(self, manager):
        manager.write_locked("CORE_VALUES.md", "VALUES")
        manager.write("IDENTITY.md", "WHO")
        manager.write("CONTEXT.md", "CTX")
        manager.write("GOALS.md", "GOALS")
        manager.write("PREFERENCES.md", "PREFS")
        manager.write("LEARNED.md", "LEARNED")

        prompt = manager.get_identity_prompt()
        # Verify order
        idx_values = prompt.index("VALUES")
        idx_who = prompt.index("Who I'm Talking To")
        idx_ctx = prompt.index("Current Context")
        idx_goals = prompt.index("Goals")
        idx_prefs = prompt.index("Communication Preferences")
        idx_learned = prompt.index("What I've Learned")

        assert idx_values < idx_who < idx_ctx < idx_goals < idx_prefs < idx_learned

    def test_memory_context_appended(self, manager):
        manager.write_locked("CORE_VALUES.md", "values")
        prompt = manager.get_identity_prompt(memory_context="some memory")
        assert "Relevant Memory" in prompt
        assert "some memory" in prompt

    def test_partial_files(self, manager):
        manager.write_locked("CORE_VALUES.md", "only values")
        prompt = manager.get_identity_prompt()
        assert "only values" in prompt
        assert "Who I'm Talking To" not in prompt


class TestGenerateFromTemplates:
    """Template generation tests."""

    def test_generates_all_files(self, manager):
        manager.generate_from_templates({"name": "TestUser", "timezone": "UTC"})
        for f in IdentityFileManager.ALL_FILES:
            assert manager.exists(f), f"{f} was not generated"

    def test_core_values_contains_name(self, manager):
        manager.generate_from_templates({"name": "Alice"})
        content = manager.read("CORE_VALUES.md")
        assert "Alice" in content

    def test_identity_contains_name(self, manager):
        manager.generate_from_templates({"name": "Bob", "timezone": "US/Eastern"})
        content = manager.read("IDENTITY.md")
        assert "Bob" in content
        assert "US/Eastern" in content

    def test_skips_existing_files(self, manager):
        manager.write("IDENTITY.md", "custom content")
        manager.generate_from_templates({"name": "Override"})
        assert manager.read("IDENTITY.md") == "custom content"

    def test_locked_file_via_template(self, manager):
        manager.generate_from_templates({"name": "User"})
        # CORE_VALUES.md should be written via write_locked, not write
        content = manager.read("CORE_VALUES.md")
        assert "Sovereignty" in content


class TestConstants:
    """Verify class-level constants."""

    def test_locked_files(self):
        assert "CORE_VALUES.md" in IdentityFileManager.LOCKED_FILES

    def test_editable_files(self):
        expected = {"IDENTITY.md", "CONTEXT.md", "GOALS.md", "PREFERENCES.md", "LEARNED.md"}
        assert set(IdentityFileManager.EDITABLE_FILES) == expected

    def test_all_files_is_union(self):
        all_set = set(IdentityFileManager.ALL_FILES)
        expected = set(IdentityFileManager.LOCKED_FILES) | set(IdentityFileManager.EDITABLE_FILES)
        assert all_set == expected
