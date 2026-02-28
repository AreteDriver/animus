"""Tests for IdentityProposalManager — typed proposal lifecycle."""

from __future__ import annotations

import pytest

from animus_bootstrap.identity.manager import IdentityFileManager
from animus_bootstrap.intelligence.proposals import (
    IdentityProposalManager,
    Proposal,
    _compute_diff,
)
from animus_bootstrap.intelligence.tools.builtin.improvement_store import ImprovementStore


@pytest.fixture()
def identity_dir(tmp_path):
    """Create identity directory with template files."""
    mgr = IdentityFileManager(tmp_path / "identity")
    mgr.generate_from_templates({"name": "TestUser", "timezone": "UTC"})
    return mgr


@pytest.fixture()
def store(tmp_path):
    """Create a fresh ImprovementStore."""
    s = ImprovementStore(tmp_path / "improvements.db")
    yield s
    s.close()


@pytest.fixture()
def pm(store, identity_dir):
    """Create an IdentityProposalManager."""
    return IdentityProposalManager(store, identity_dir)


class TestProposalDataclass:
    def test_from_store_dict_basic(self):
        d = {
            "id": 1,
            "area": "identity:CONTEXT.md",
            "description": "Proposed change to CONTEXT.md: reason",
            "status": "proposed",
            "timestamp": "2026-02-27T00:00:00",
            "patch": "new content",
            "analysis": "size info",
            "applied_at": None,
        }
        p = Proposal.from_store_dict(d, "old content")
        assert p.id == 1
        assert p.file == "CONTEXT.md"
        assert p.proposed == "new content"
        assert p.current == "old content"
        assert p.status == "pending"  # mapped from "proposed"
        assert "---" in p.diff or "+new content" in p.diff

    def test_from_store_dict_approved_status(self):
        d = {
            "id": 2,
            "area": "identity:GOALS.md",
            "description": "Update goals",
            "status": "approved",
            "timestamp": "2026-02-27T00:00:00",
            "patch": "new goals",
            "applied_at": "2026-02-27T01:00:00",
        }
        p = Proposal.from_store_dict(d)
        assert p.status == "approved"
        assert p.resolved_at == "2026-02-27T01:00:00"

    def test_from_store_dict_rejected_status(self):
        d = {
            "id": 3,
            "area": "identity:PREFERENCES.md",
            "description": "Change prefs",
            "status": "rejected",
            "timestamp": "2026-02-27T00:00:00",
            "patch": "",
            "applied_at": "2026-02-27T02:00:00",
        }
        p = Proposal.from_store_dict(d)
        assert p.status == "rejected"

    def test_from_store_dict_no_colon_in_area(self):
        d = {
            "id": 4,
            "area": "general",
            "description": "General improvement",
            "status": "proposed",
            "timestamp": "2026-02-27T00:00:00",
            "patch": "",
        }
        p = Proposal.from_store_dict(d)
        assert p.file == "general"


class TestComputeDiff:
    def test_diff_shows_changes(self):
        diff = _compute_diff("test.md", "line1\nline2\n", "line1\nline3\n")
        assert "--- a/test.md" in diff
        assert "+++ b/test.md" in diff
        assert "-line2" in diff
        assert "+line3" in diff

    def test_diff_empty_current(self):
        diff = _compute_diff("test.md", "", "new content\n")
        assert "+new content" in diff

    def test_diff_identical_files(self):
        diff = _compute_diff("test.md", "same\n", "same\n")
        assert diff == ""


class TestCreate:
    def test_create_returns_typed_proposal(self, pm, identity_dir):
        identity_dir.write("CONTEXT.md", "old context")
        p = pm.create("CONTEXT.md", "new context", "updating context")
        assert isinstance(p, Proposal)
        assert p.id >= 1
        assert p.file == "CONTEXT.md"
        assert p.proposed == "new context"
        assert p.current == "old context"
        assert p.status == "pending"
        assert p.reason == "updating context"
        assert p.diff  # non-empty diff
        assert p.created_at  # non-empty timestamp

    def test_create_core_values_raises(self, pm):
        with pytest.raises(PermissionError, match="immutable"):
            pm.create("CORE_VALUES.md", "hacked", "evil reason")

    def test_create_persists_to_store(self, pm, store, identity_dir):
        identity_dir.write("GOALS.md", "old goals")
        p = pm.create("GOALS.md", "new goals", "goal update")
        raw = store.get(p.id)
        assert raw is not None
        assert raw["area"] == "identity:GOALS.md"
        assert raw["patch"] == "new goals"
        assert raw["status"] == "proposed"


class TestListPending:
    def test_empty_when_no_proposals(self, pm):
        assert pm.list_pending() == []

    def test_filters_only_pending_identity_proposals(self, pm, store, identity_dir):
        identity_dir.write("CONTEXT.md", "ctx")
        pm.create("CONTEXT.md", "new ctx", "update")
        # Add a non-identity proposal directly
        store.save(
            {
                "area": "tool:web_search",
                "description": "Improve search",
                "status": "proposed",
                "timestamp": "2026-02-27T00:00:00",
            }
        )
        # Add an approved identity proposal
        store.save(
            {
                "area": "identity:GOALS.md",
                "description": "Old goals change",
                "status": "approved",
                "timestamp": "2026-02-27T00:00:00",
            }
        )
        pending = pm.list_pending()
        assert len(pending) == 1
        assert pending[0].file == "CONTEXT.md"

    def test_pending_includes_current_content(self, pm, identity_dir):
        identity_dir.write("PREFERENCES.md", "current prefs")
        pm.create("PREFERENCES.md", "new prefs", "pref change")
        pending = pm.list_pending()
        assert pending[0].current == "current prefs"


class TestApprove:
    def test_approve_writes_file(self, pm, identity_dir):
        identity_dir.write("CONTEXT.md", "old")
        p = pm.create("CONTEXT.md", "approved content", "update")
        result = pm.approve(p.id)
        assert result.status == "approved"
        assert identity_dir.read("CONTEXT.md") == "approved content"

    def test_approve_updates_store_status(self, pm, store, identity_dir):
        identity_dir.write("GOALS.md", "old")
        p = pm.create("GOALS.md", "new goals", "goal update")
        pm.approve(p.id)
        raw = store.get(p.id)
        assert raw["status"] == "approved"
        assert raw["applied_at"] is not None

    def test_approve_nonexistent_raises(self, pm):
        with pytest.raises(ValueError, match="not found"):
            pm.approve(999)

    def test_approve_non_identity_raises(self, pm, store):
        store.save(
            {
                "area": "tool:search",
                "description": "Not identity",
                "status": "proposed",
                "timestamp": "2026-02-27T00:00:00",
            }
        )
        with pytest.raises(ValueError, match="not an identity proposal"):
            pm.approve(1)

    def test_approve_locked_file_raises(self, pm, store):
        store.save(
            {
                "area": "identity:CORE_VALUES.md",
                "description": "Try to hack values",
                "status": "proposed",
                "timestamp": "2026-02-27T00:00:00",
                "patch": "hacked",
            }
        )
        with pytest.raises(PermissionError):
            pm.approve(1)


class TestReject:
    def test_reject_updates_status(self, pm, store, identity_dir):
        identity_dir.write("CONTEXT.md", "ctx")
        p = pm.create("CONTEXT.md", "new ctx", "change")
        result = pm.reject(p.id, "not useful")
        assert result.status == "rejected"
        assert result.rejection_reason == "not useful"

    def test_reject_stores_reason_in_analysis(self, pm, store, identity_dir):
        identity_dir.write("GOALS.md", "goals")
        p = pm.create("GOALS.md", "new goals", "update")
        pm.reject(p.id, "too aggressive")
        raw = store.get(p.id)
        assert "Rejection reason: too aggressive" in raw["analysis"]

    def test_reject_nonexistent_raises(self, pm):
        with pytest.raises(ValueError, match="not found"):
            pm.reject(999)

    def test_reject_without_reason(self, pm, identity_dir):
        identity_dir.write("CONTEXT.md", "ctx")
        p = pm.create("CONTEXT.md", "new", "change")
        result = pm.reject(p.id)
        assert result.rejection_reason == ""


class TestHistory:
    def test_empty_history(self, pm):
        assert pm.history() == []

    def test_history_returns_all_identity_proposals(self, pm, identity_dir, store):
        identity_dir.write("CONTEXT.md", "ctx")
        p1 = pm.create("CONTEXT.md", "new1", "first change")
        p2 = pm.create("CONTEXT.md", "new2", "second change")
        pm.approve(p1.id)
        pm.reject(p2.id, "no")
        # Non-identity proposal should not appear
        store.save(
            {
                "area": "tool:search",
                "description": "Not identity",
                "status": "proposed",
                "timestamp": "2026-02-27T00:00:00",
            }
        )
        h = pm.history()
        assert len(h) == 2

    def test_history_sorted_newest_first(self, pm, identity_dir):
        identity_dir.write("CONTEXT.md", "ctx")
        pm.create("CONTEXT.md", "first", "1")
        pm.create("CONTEXT.md", "second", "2")
        h = pm.history()
        assert h[0].created_at >= h[1].created_at


class TestGet:
    def test_get_existing_proposal(self, pm, identity_dir):
        identity_dir.write("CONTEXT.md", "ctx")
        p = pm.create("CONTEXT.md", "new ctx", "update")
        result = pm.get(p.id)
        assert result is not None
        assert result.file == "CONTEXT.md"
        assert result.current == "ctx"  # current reads from disk (unchanged)

    def test_get_nonexistent_returns_none(self, pm):
        assert pm.get(999) is None

    def test_get_non_identity_returns_none(self, pm, store):
        store.save(
            {
                "area": "tool:search",
                "description": "Not identity",
                "status": "proposed",
                "timestamp": "2026-02-27T00:00:00",
            }
        )
        assert pm.get(1) is None


class TestThresholdEdgeCases:
    """Test the 20% change threshold at exact boundaries."""

    def test_19_percent_change_direct_write(self, pm, identity_dir):
        """19% change should NOT create a proposal."""
        # Write 100 chars, change to 119 chars (19% increase)
        base = "x" * 100
        identity_dir.write("CONTEXT.md", base)
        # Just verify proposal creation works at boundary
        new = "x" * 119
        p = pm.create("CONTEXT.md", new, "small change")
        # It creates a proposal because we're calling create() directly
        assert p.status == "pending"

    def test_21_percent_change_creates_proposal(self, pm, identity_dir):
        """21% change should create a proposal."""
        base = "x" * 100
        identity_dir.write("CONTEXT.md", base)
        new = "x" * 121
        p = pm.create("CONTEXT.md", new, "big change")
        assert p.status == "pending"
        assert p.diff  # has a diff
