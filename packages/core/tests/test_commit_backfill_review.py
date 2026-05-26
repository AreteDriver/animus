"""Tests for animus.scripts.commit_backfill_review (Stage 2.E tooling)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from animus.memory.layer import MemoryLayer
from animus.memory.types import Sensitivity
from animus.scripts.commit_backfill_review import _resolve_target, commit_review


class TestResolveTarget:
    """The review_state verb resolution."""

    def test_pending_returns_none(self):
        assert _resolve_target("pending", Sensitivity.SECRET) is None

    def test_applied_returns_none(self):
        assert _resolve_target("applied", Sensitivity.SECRET) is None

    def test_confirm_keeps_tier(self):
        assert _resolve_target("confirm", Sensitivity.CONFIDENTIAL) is Sensitivity.CONFIDENTIAL

    def test_downgrade_secret_to_confidential(self):
        assert _resolve_target("downgrade", Sensitivity.SECRET) is Sensitivity.CONFIDENTIAL

    def test_downgrade_confidential_to_personal(self):
        assert _resolve_target("downgrade", Sensitivity.CONFIDENTIAL) is Sensitivity.PERSONAL

    def test_downgrade_personal_to_public(self):
        assert _resolve_target("downgrade", Sensitivity.PERSONAL) is Sensitivity.PUBLIC

    def test_downgrade_public_clamps(self):
        """Can't downgrade below PUBLIC."""
        assert _resolve_target("downgrade", Sensitivity.PUBLIC) is Sensitivity.PUBLIC

    def test_upgrade_public_to_personal(self):
        assert _resolve_target("upgrade", Sensitivity.PUBLIC) is Sensitivity.PERSONAL

    def test_upgrade_secret_clamps(self):
        """Can't upgrade above SECRET."""
        assert _resolve_target("upgrade", Sensitivity.SECRET) is Sensitivity.SECRET

    def test_explicit_tier_name(self):
        assert _resolve_target("public", Sensitivity.SECRET) is Sensitivity.PUBLIC
        assert _resolve_target("personal", Sensitivity.SECRET) is Sensitivity.PERSONAL
        assert _resolve_target("confidential", Sensitivity.PUBLIC) is Sensitivity.CONFIDENTIAL

    def test_case_insensitive(self):
        assert _resolve_target("DOWNGRADE", Sensitivity.SECRET) is Sensitivity.CONFIDENTIAL
        assert _resolve_target(" Confirm ", Sensitivity.PERSONAL) is Sensitivity.PERSONAL

    def test_unrecognized_raises(self):
        with pytest.raises(ValueError, match="Unrecognized review_state"):
            _resolve_target("maybe", Sensitivity.SECRET)


class TestCommitReview:
    """End-to-end against a real LocalMemoryStore + JSONL file."""

    @pytest.fixture
    def seeded(self, tmp_path: Path):
        """Seed the store + write a review file with mixed verbs."""
        layer = MemoryLayer(tmp_path, backend="local")
        mems = {
            "drop": layer.remember(
                content="false-positive credential thesis chunk",
                tags=["credentialing-thesis"],
                sensitivity=Sensitivity.SECRET,  # heuristic mis-classified
            ),
            "keep": layer.remember(
                content="legit toyota litigation note",
                tags=["toyota"],
                sensitivity=Sensitivity.CONFIDENTIAL,  # heuristic was right
            ),
            "bump": layer.remember(
                content="actually a credential",
                tags=["api-key"],
                sensitivity=Sensitivity.PERSONAL,  # under-classified
            ),
            "skip": layer.remember(
                content="not reviewed yet",
                tags=["legal"],
                sensitivity=Sensitivity.CONFIDENTIAL,
            ),
        }

        audit_path = tmp_path / "audit" / "backfill-review.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "id": mems["drop"].id,
                "tier_assigned": "secret",
                "review_state": "public",  # explicit override
                "reason": "test",
                "content_preview": "...",
                "tags": ["x"],
                "source": "stated",
                "created_at": "2026-01-01T00:00:00",
            },
            {
                "id": mems["keep"].id,
                "tier_assigned": "confidential",
                "review_state": "confirm",
                "reason": "test",
                "content_preview": "...",
                "tags": ["x"],
                "source": "stated",
                "created_at": "2026-01-01T00:00:00",
            },
            {
                "id": mems["bump"].id,
                "tier_assigned": "personal",
                "review_state": "upgrade",
                "reason": "test",
                "content_preview": "...",
                "tags": ["x"],
                "source": "stated",
                "created_at": "2026-01-01T00:00:00",
            },
            {
                "id": mems["skip"].id,
                "tier_assigned": "confidential",
                "review_state": "pending",
                "reason": "test",
                "content_preview": "...",
                "tags": ["x"],
                "source": "stated",
                "created_at": "2026-01-01T00:00:00",
            },
        ]
        audit_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return tmp_path, mems

    def test_dry_run_reports_counts(self, seeded):
        tmp_path, _ = seeded
        stats = commit_review(dry_run=True, data_dir=tmp_path)
        assert stats["total"] == 4
        assert stats["applied"] == 3
        assert stats["skipped_pending"] == 1
        assert stats["skipped_applied"] == 0
        assert stats["errors"] == 0

    def test_dry_run_does_not_persist(self, seeded):
        tmp_path, mems = seeded
        commit_review(dry_run=True, data_dir=tmp_path)
        # Original tiers preserved
        reloaded = MemoryLayer(tmp_path, backend="local")
        assert reloaded.store.retrieve(mems["drop"].id).sensitivity is Sensitivity.SECRET
        assert reloaded.store.retrieve(mems["bump"].id).sensitivity is Sensitivity.PERSONAL

    def test_apply_persists_tier_changes(self, seeded):
        tmp_path, mems = seeded
        commit_review(dry_run=False, data_dir=tmp_path)
        reloaded = MemoryLayer(tmp_path, backend="local")
        assert reloaded.store.retrieve(mems["drop"].id).sensitivity is Sensitivity.PUBLIC
        assert reloaded.store.retrieve(mems["keep"].id).sensitivity is Sensitivity.CONFIDENTIAL
        assert reloaded.store.retrieve(mems["bump"].id).sensitivity is Sensitivity.CONFIDENTIAL
        # Pending row unchanged
        assert reloaded.store.retrieve(mems["skip"].id).sensitivity is Sensitivity.CONFIDENTIAL

    def test_apply_marks_rows_idempotent(self, seeded):
        tmp_path, _ = seeded
        commit_review(dry_run=False, data_dir=tmp_path)
        audit_path = tmp_path / "audit" / "backfill-review.jsonl"
        rows = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        applied = [r for r in rows if r["review_state"] == "applied"]
        assert len(applied) == 3
        for r in applied:
            assert "final_tier" in r
            assert "committed_at" in r

    def test_second_run_is_no_op(self, seeded):
        tmp_path, _ = seeded
        commit_review(dry_run=False, data_dir=tmp_path)
        stats = commit_review(dry_run=False, data_dir=tmp_path)
        assert stats["applied"] == 0
        assert stats["skipped_applied"] == 3
        assert stats["skipped_pending"] == 1

    def test_missing_review_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No review file"):
            commit_review(dry_run=False, data_dir=tmp_path)

    def test_session_boundary_lines_preserved(self, tmp_path):
        """Boundary markers from re-running migrate_sensitivity_tiers stay
        in the file across commit operations."""
        layer = MemoryLayer(tmp_path, backend="local")
        m = layer.remember(content="x", sensitivity=Sensitivity.SECRET)

        audit_path = tmp_path / "audit" / "backfill-review.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps({"_session_boundary": "2026-05-01T00:00:00"})
            + "\n"
            + json.dumps(
                {
                    "id": m.id,
                    "tier_assigned": "secret",
                    "review_state": "confirm",
                    "reason": "x",
                    "content_preview": "x",
                    "tags": [],
                    "source": "stated",
                    "created_at": "x",
                }
            )
            + "\n"
        )
        commit_review(dry_run=False, data_dir=tmp_path)
        final = audit_path.read_text()
        assert "_session_boundary" in final
