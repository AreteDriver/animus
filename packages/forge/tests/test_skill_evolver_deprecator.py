"""Tests for SkillDeprecator."""

from animus_forge.self_improve.approval import ApprovalGate
from animus_forge.skills.evolver.deprecator import SkillDeprecator
from animus_forge.skills.evolver.models import SkillMetrics
from animus_forge.state.backends import SQLiteBackend


def _make_backend():
    backend = SQLiteBackend(":memory:")
    backend.executescript("""
        CREATE TABLE IF NOT EXISTS skill_deprecations (
            skill_name TEXT PRIMARY KEY,
            status TEXT DEFAULT 'flagged',
            flagged_at TEXT NOT NULL,
            deprecated_at TEXT,
            retired_at TEXT,
            reason TEXT,
            success_rate_at_flag REAL,
            invocations_at_flag INTEGER,
            replacement_skill TEXT,
            approval_id TEXT
        );
    """)
    return backend


def _make_metrics(**overrides):
    defaults = dict(
        skill_name="test_skill",
        skill_version="1.0.0",
        period_start="a",
        period_end="b",
        total_invocations=50,
        success_rate=0.3,
        computed_at="now",
    )
    defaults.update(overrides)
    return SkillMetrics(**defaults)


class TestFlagForDeprecation:
    def test_basic(self):
        backend = _make_backend()
        dep = SkillDeprecator(backend)
        record = dep.flag_for_deprecation("s", "low quality")
        assert record.skill_name == "s"
        assert record.status == "flagged"
        assert record.reason == "low quality"

    def test_with_metrics(self):
        backend = _make_backend()
        dep = SkillDeprecator(backend)
        metrics = _make_metrics(success_rate=0.3, total_invocations=50)
        record = dep.flag_for_deprecation("s", "bad", metrics)
        assert record.success_rate_at_flag == 0.3
        assert record.invocations_at_flag == 50

    def test_persisted(self):
        backend = _make_backend()
        dep = SkillDeprecator(backend)
        dep.flag_for_deprecation("s", "test")
        rows = backend.fetchall("SELECT * FROM skill_deprecations", ())
        assert len(rows) == 1
        assert rows[0]["skill_name"] == "s"


class TestDeprecate:
    def test_from_flagged(self):
        backend = _make_backend()
        dep = SkillDeprecator(backend)
        dep.flag_for_deprecation("s", "test")
        assert dep.deprecate("s") is True

    def test_cannot_deprecate_non_flagged(self):
        backend = _make_backend()
        dep = SkillDeprecator(backend)
        assert dep.deprecate("nonexistent") is False


class TestRetirement:
    def test_retire_without_gate(self):
        backend = _make_backend()
        dep = SkillDeprecator(backend)
        dep.flag_for_deprecation("s", "test")
        dep.deprecate("s")
        assert dep.retire("s") is True

    def test_retire_with_approved_gate(self):
        backend = _make_backend()
        gate = ApprovalGate()
        dep = SkillDeprecator(backend, gate)
        dep.flag_for_deprecation("s", "test")
        dep.deprecate("s")

        approval_id = dep.request_retirement("s")
        assert approval_id
        gate.auto_approve_for_testing(approval_id)
        assert dep.retire("s", approval_id) is True

    def test_retire_with_unapproved_gate(self):
        backend = _make_backend()
        gate = ApprovalGate()
        dep = SkillDeprecator(backend, gate)
        dep.flag_for_deprecation("s", "test")
        dep.deprecate("s")

        assert dep.retire("s", "fake_id") is False

    def test_request_retirement_non_deprecated(self):
        backend = _make_backend()
        gate = ApprovalGate()
        dep = SkillDeprecator(backend, gate)
        dep.flag_for_deprecation("s", "test")
        # Still "flagged", not "deprecated"
        assert dep.request_retirement("s") == ""


class TestQueryByStatus:
    def test_get_flagged(self):
        backend = _make_backend()
        dep = SkillDeprecator(backend)
        dep.flag_for_deprecation("a", "test")
        dep.flag_for_deprecation("b", "test")
        flagged = dep.get_flagged_skills()
        assert len(flagged) == 2

    def test_get_deprecated(self):
        backend = _make_backend()
        dep = SkillDeprecator(backend)
        dep.flag_for_deprecation("a", "test")
        dep.deprecate("a")
        deprecated = dep.get_deprecated_skills()
        assert len(deprecated) == 1
        assert deprecated[0].skill_name == "a"

    def test_empty(self):
        backend = _make_backend()
        dep = SkillDeprecator(backend)
        assert dep.get_flagged_skills() == []
        assert dep.get_deprecated_skills() == []
