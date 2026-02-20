"""Tests for SkillVersioner."""

from animus_forge.skills.evolver.versioner import SkillVersioner
from animus_forge.state.backends import SQLiteBackend


def _make_backend():
    backend = SQLiteBackend(":memory:")
    backend.executescript("""
        CREATE TABLE IF NOT EXISTS skill_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skill_name TEXT NOT NULL,
            version TEXT NOT NULL,
            previous_version TEXT,
            change_type TEXT NOT NULL,
            change_description TEXT,
            schema_snapshot TEXT NOT NULL,
            diff_summary TEXT,
            approval_id TEXT,
            created_at TEXT NOT NULL,
            created_by TEXT DEFAULT 'skill_evolver',
            UNIQUE(skill_name, version)
        );
    """)
    return backend


class TestRecordVersion:
    def test_basic(self):
        backend = _make_backend()
        v = SkillVersioner(backend)
        v.record_version("skill_a", "1.0.0", None, "manual", "Initial", "name: skill_a")
        rows = backend.fetchall("SELECT * FROM skill_versions", ())
        assert len(rows) == 1
        assert rows[0]["skill_name"] == "skill_a"

    def test_with_all_fields(self):
        backend = _make_backend()
        v = SkillVersioner(backend)
        v.record_version(
            "skill_a",
            "1.0.1",
            "1.0.0",
            "tune",
            "Tightened consensus",
            "name: skill_a\nconsensus: majority",
            approval_id="abc",
            diff_summary="+consensus: majority",
        )
        rows = backend.fetchall("SELECT * FROM skill_versions", ())
        assert rows[0]["approval_id"] == "abc"
        assert rows[0]["diff_summary"] == "+consensus: majority"

    def test_replace_on_conflict(self):
        backend = _make_backend()
        v = SkillVersioner(backend)
        v.record_version("s", "1.0.0", None, "manual", "v1", "yaml1")
        v.record_version("s", "1.0.0", None, "manual", "v1-updated", "yaml2")
        rows = backend.fetchall("SELECT * FROM skill_versions WHERE skill_name='s'", ())
        assert len(rows) == 1
        assert rows[0]["schema_snapshot"] == "yaml2"


class TestGetVersionHistory:
    def test_empty(self):
        backend = _make_backend()
        v = SkillVersioner(backend)
        assert v.get_version_history("nonexistent") == []

    def test_ordered_by_date(self):
        backend = _make_backend()
        v = SkillVersioner(backend)
        v.record_version("s", "1.0.0", None, "manual", "v1", "y1")
        v.record_version("s", "1.0.1", "1.0.0", "tune", "v2", "y2")
        history = v.get_version_history("s")
        assert len(history) == 2
        assert history[0]["version"] == "1.0.1"  # newest first


class TestGetVersionSnapshot:
    def test_found(self):
        backend = _make_backend()
        v = SkillVersioner(backend)
        v.record_version("s", "1.0.0", None, "manual", "v1", "yaml_content")
        assert v.get_version_snapshot("s", "1.0.0") == "yaml_content"

    def test_not_found(self):
        backend = _make_backend()
        v = SkillVersioner(backend)
        assert v.get_version_snapshot("s", "9.9.9") == ""


class TestComputeDiff:
    def test_diff(self):
        backend = _make_backend()
        v = SkillVersioner(backend)
        v.record_version("s", "1.0.0", None, "manual", "v1", "name: s\nversion: 1.0.0\n")
        v.record_version("s", "1.0.1", "1.0.0", "tune", "v2", "name: s\nversion: 1.0.1\n")
        diff = v.compute_diff("s", "1.0.0", "1.0.1")
        assert "1.0.0" in diff
        assert "1.0.1" in diff

    def test_both_empty(self):
        backend = _make_backend()
        v = SkillVersioner(backend)
        assert v.compute_diff("s", "a", "b") == ""


class TestGetLatestVersion:
    def test_found(self):
        backend = _make_backend()
        v = SkillVersioner(backend)
        v.record_version("s", "1.0.0", None, "manual", "v1", "y1")
        v.record_version("s", "1.0.1", "1.0.0", "tune", "v2", "y2")
        assert v.get_latest_version("s") == "1.0.1"

    def test_not_found(self):
        backend = _make_backend()
        v = SkillVersioner(backend)
        assert v.get_latest_version("nonexistent") == ""
