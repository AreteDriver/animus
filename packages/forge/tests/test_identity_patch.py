"""Tests for the identity patch approval gate."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from animus_forge.coordination.identity_patch import IdentityPatch, IdentityPatchGate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_patches_path(tmp_path: Path) -> Path:
    return tmp_path / "identity_patches.json"


@pytest.fixture()
def tmp_audit_path(tmp_path: Path) -> Path:
    return tmp_path / "forge_audit.jsonl"


@pytest.fixture()
def gate(tmp_patches_path: Path, tmp_audit_path: Path) -> IdentityPatchGate:
    return IdentityPatchGate(
        patches_path=tmp_patches_path,
        audit_log_path=tmp_audit_path,
    )


@pytest.fixture()
def sample_patch() -> IdentityPatch:
    return IdentityPatch(
        proposed_changes={"CONTEXT.md": "Updated context for new project"},
        reasoning="Adding new project context",
        impact_assessment="Low impact — additive change only",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPropose:
    """Tests for IdentityPatchGate.propose()."""

    def test_propose_returns_patch_id(
        self, gate: IdentityPatchGate, sample_patch: IdentityPatch
    ) -> None:
        patch_id = gate.propose(sample_patch)
        assert patch_id == sample_patch.patch_id
        assert len(patch_id) == 8

    def test_propose_adds_to_pending(
        self, gate: IdentityPatchGate, sample_patch: IdentityPatch
    ) -> None:
        gate.propose(sample_patch)
        pending = gate.list_pending()
        assert len(pending) == 1
        assert pending[0].patch_id == sample_patch.patch_id

    def test_propose_persists_to_disk(
        self,
        gate: IdentityPatchGate,
        sample_patch: IdentityPatch,
        tmp_patches_path: Path,
    ) -> None:
        gate.propose(sample_patch)
        assert tmp_patches_path.exists()
        data = json.loads(tmp_patches_path.read_text())
        assert len(data) == 1
        assert data[0]["patch_id"] == sample_patch.patch_id

    def test_propose_appends_audit_log(
        self,
        gate: IdentityPatchGate,
        sample_patch: IdentityPatch,
        tmp_audit_path: Path,
    ) -> None:
        gate.propose(sample_patch)
        assert tmp_audit_path.exists()
        lines = tmp_audit_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event_type"] == "identity_patch_proposed"
        assert record["patch_id"] == sample_patch.patch_id


class TestApprove:
    """Tests for IdentityPatchGate.approve()."""

    def test_approve_existing_patch(
        self, gate: IdentityPatchGate, sample_patch: IdentityPatch
    ) -> None:
        patch_id = gate.propose(sample_patch)
        result = gate.approve(patch_id)
        assert result is True
        assert gate.list_pending() == []

    def test_approve_nonexistent_patch(self, gate: IdentityPatchGate) -> None:
        result = gate.approve("nonexistent")
        assert result is False

    def test_approve_writes_staging_file(
        self,
        gate: IdentityPatchGate,
        sample_patch: IdentityPatch,
        tmp_patches_path: Path,
    ) -> None:
        patch_id = gate.propose(sample_patch)
        gate.approve(patch_id)
        staging = tmp_patches_path.parent / f"staged_{patch_id}.json"
        assert staging.exists()
        data = json.loads(staging.read_text())
        assert data["patch_id"] == patch_id

    def test_approve_uses_proposal_manager(
        self, tmp_patches_path: Path, tmp_audit_path: Path
    ) -> None:
        mgr = MagicMock()
        gate = IdentityPatchGate(
            patches_path=tmp_patches_path,
            audit_log_path=tmp_audit_path,
            proposal_manager=mgr,
        )
        patch = IdentityPatch(
            proposed_changes={"LEARNED.md": "new content"},
            reasoning="test",
            impact_assessment="none",
        )
        patch_id = gate.propose(patch)
        gate.approve(patch_id)
        mgr.create.assert_called_once_with(
            file="LEARNED.md",
            proposed_content="new content",
            reason="test",
        )

    def test_approve_appends_audit_log(
        self,
        gate: IdentityPatchGate,
        sample_patch: IdentityPatch,
        tmp_audit_path: Path,
    ) -> None:
        patch_id = gate.propose(sample_patch)
        gate.approve(patch_id)
        lines = tmp_audit_path.read_text().strip().split("\n")
        assert len(lines) == 2  # propose + approve
        record = json.loads(lines[1])
        assert record["event_type"] == "identity_patch_approved"


class TestReject:
    """Tests for IdentityPatchGate.reject()."""

    def test_reject_existing_patch(
        self, gate: IdentityPatchGate, sample_patch: IdentityPatch
    ) -> None:
        patch_id = gate.propose(sample_patch)
        result = gate.reject(patch_id, reason="Too risky")
        assert result is True
        assert gate.list_pending() == []

    def test_reject_nonexistent_patch(self, gate: IdentityPatchGate) -> None:
        result = gate.reject("nonexistent", reason="Gone")
        assert result is False

    def test_reject_appends_audit_log(
        self,
        gate: IdentityPatchGate,
        sample_patch: IdentityPatch,
        tmp_audit_path: Path,
    ) -> None:
        patch_id = gate.propose(sample_patch)
        gate.reject(patch_id, reason="Violates P4")
        lines = tmp_audit_path.read_text().strip().split("\n")
        record = json.loads(lines[-1])
        assert record["event_type"] == "identity_patch_rejected"
        assert record["reason"] == "Violates P4"


class TestListPending:
    """Tests for IdentityPatchGate.list_pending()."""

    def test_empty_initially(self, gate: IdentityPatchGate) -> None:
        assert gate.list_pending() == []

    def test_multiple_pending(self, gate: IdentityPatchGate) -> None:
        for i in range(3):
            patch = IdentityPatch(
                proposed_changes={f"file_{i}.md": f"content {i}"},
                reasoning=f"reason {i}",
                impact_assessment="low",
            )
            gate.propose(patch)
        assert len(gate.list_pending()) == 3


class TestPersistence:
    """Tests for patch persistence across instances."""

    def test_load_patches_from_disk(
        self, tmp_patches_path: Path, tmp_audit_path: Path
    ) -> None:
        gate1 = IdentityPatchGate(
            patches_path=tmp_patches_path,
            audit_log_path=tmp_audit_path,
        )
        patch = IdentityPatch(
            proposed_changes={"test.md": "content"},
            reasoning="persist test",
            impact_assessment="none",
        )
        gate1.propose(patch)

        # Create a new gate — should load persisted patches
        gate2 = IdentityPatchGate(
            patches_path=tmp_patches_path,
            audit_log_path=tmp_audit_path,
        )
        pending = gate2.list_pending()
        assert len(pending) == 1
        assert pending[0].reasoning == "persist test"


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_proposed_changes(self, gate: IdentityPatchGate) -> None:
        patch = IdentityPatch(
            proposed_changes={},
            reasoning="empty",
            impact_assessment="none",
        )
        patch_id = gate.propose(patch)
        assert gate.approve(patch_id) is True

    def test_double_approve_fails(
        self, gate: IdentityPatchGate, sample_patch: IdentityPatch
    ) -> None:
        patch_id = gate.propose(sample_patch)
        assert gate.approve(patch_id) is True
        assert gate.approve(patch_id) is False

    def test_approve_then_reject_fails(
        self, gate: IdentityPatchGate, sample_patch: IdentityPatch
    ) -> None:
        patch_id = gate.propose(sample_patch)
        gate.approve(patch_id)
        assert gate.reject(patch_id) is False
