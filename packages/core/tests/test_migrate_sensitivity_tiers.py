"""Tests for animus.scripts.migrate_sensitivity_tiers (Stage 2.D)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from animus.memory.layer import MemoryLayer
from animus.memory.types import Sensitivity
from animus.scripts.migrate_sensitivity_tiers import (
    CONFIDENTIAL_TAG_NEEDLES,
    SECRET_TAG_NEEDLES,
    classify,
    migrate,
)


def _make_memory(content: str, tags: list[str] | None = None, source: str = "stated"):
    """Helper — make a Memory through MemoryLayer.remember so it's in the store."""
    from animus.memory.types import Memory

    return Memory.create(content=content, tags=tags or [], source=source)


class TestClassifyHeuristic:
    """Tag-substring heuristic rules."""

    def test_default_is_personal(self):
        mem = _make_memory("random fact")
        tier, _ = classify(mem)
        assert tier is Sensitivity.PERSONAL

    def test_secret_tag_match(self):
        mem = _make_memory("x", tags=["api-key", "anthropic"])
        tier, reason = classify(mem)
        assert tier is Sensitivity.SECRET
        assert "secret" in reason

    def test_confidential_tag_match(self):
        for needle in CONFIDENTIAL_TAG_NEEDLES:
            mem = _make_memory("x", tags=[f"{needle}-meeting"])
            tier, _ = classify(mem)
            assert tier is Sensitivity.CONFIDENTIAL, (
                f"expected confidential for tag '{needle}-meeting'"
            )

    def test_secret_beats_confidential(self):
        mem = _make_memory("x", tags=["tiaid-client", "api-key"])
        tier, _ = classify(mem)
        assert tier is Sensitivity.SECRET

    def test_harvest_source_is_public(self):
        mem = _make_memory("public repo content", source="harvest")
        tier, reason = classify(mem)
        assert tier is Sensitivity.PUBLIC
        assert "harvest" in reason

    def test_tiaid_tag_overrides_source(self):
        mem = _make_memory("client work", tags=["tiaid"], source="stated")
        tier, _ = classify(mem)
        assert tier is Sensitivity.CONFIDENTIAL

    def test_case_insensitive_match(self):
        mem = _make_memory("x", tags=["TOYOTA-Internal"])
        tier, _ = classify(mem)
        assert tier is Sensitivity.CONFIDENTIAL

    def test_known_personal_tags_classify(self):
        # SECRET_TAG_NEEDLES iterations
        for needle in SECRET_TAG_NEEDLES:
            mem = _make_memory("x", tags=[f"a-{needle}-z"])
            tier, _ = classify(mem)
            assert tier is Sensitivity.SECRET, f"expected secret for tag '{needle}'"


class TestMigrateIntegration:
    """End-to-end migration against the LocalMemoryStore."""

    @pytest.fixture
    def seeded_layer(self, tmp_path: Path):
        layer = MemoryLayer(tmp_path, backend="local")
        # Seed varied content
        layer.remember(content="TRIM vs secure-erase facts", tags=["ssd", "linux"])
        layer.remember(content="meeting with client", tags=["tiaid", "kickoff"])
        layer.remember(content="ssh key rotated", tags=["api-key", "rotation"])
        layer.remember(content="public repo discovery", tags=["harvest"], source="harvest")
        return layer

    def test_dry_run_reports_correct_counts(self, seeded_layer, tmp_path):
        stats = migrate(dry_run=True, data_dir=tmp_path)
        assert stats["total"] == 4
        # The PUBLIC-default writes get reclassified by the heuristic
        assert stats["classified_public"] == 1  # harvest source
        assert stats["classified_personal"] == 1  # generic
        assert stats["classified_confidential"] == 1  # tiaid tag
        assert stats["classified_secret"] == 1  # api-key tag
        assert stats["review_queued"] == 2  # confidential + secret

    def test_dry_run_does_not_persist_changes(self, seeded_layer, tmp_path):
        migrate(dry_run=True, data_dir=tmp_path)
        # Reload layer, verify sensitivity unchanged (still PUBLIC default)
        reloaded = MemoryLayer(tmp_path, backend="local")
        for m in reloaded.store.list_all():
            assert m.sensitivity is Sensitivity.PUBLIC

    def test_dry_run_does_not_write_review_jsonl(self, seeded_layer, tmp_path):
        migrate(dry_run=True, data_dir=tmp_path)
        audit_path = tmp_path / "audit" / "backfill-review.jsonl"
        assert not audit_path.exists()

    def test_apply_reclassifies_in_place(self, seeded_layer, tmp_path):
        migrate(dry_run=False, data_dir=tmp_path)
        # Reload — verify sensitivity persisted via store.update()
        reloaded = MemoryLayer(tmp_path, backend="local")
        tiers = {m.sensitivity for m in reloaded.store.list_all()}
        assert Sensitivity.CONFIDENTIAL in tiers
        assert Sensitivity.SECRET in tiers
        assert Sensitivity.PUBLIC in tiers
        assert Sensitivity.PERSONAL in tiers

    def test_apply_writes_review_jsonl(self, seeded_layer, tmp_path):
        migrate(dry_run=False, data_dir=tmp_path)
        audit_path = tmp_path / "audit" / "backfill-review.jsonl"
        assert audit_path.exists()

        rows = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        # 2 candidate rows (confidential + secret), no session boundary on first run
        assert len(rows) == 2
        tiers = {row["tier_assigned"] for row in rows}
        assert tiers == {"confidential", "secret"}
        for row in rows:
            assert row["review_state"] == "pending"
            assert "content_preview" in row
            assert "reason" in row

    def test_idempotent_already_classified(self, seeded_layer, tmp_path):
        # First run — classifies everything
        migrate(dry_run=False, data_dir=tmp_path)
        # Second run — non-PUBLIC are recognized as already-classified.
        # PUBLIC tier re-evaluates each run (no marker distinguishes
        # "default PUBLIC" from "explicit PUBLIC"); the heuristic is a
        # no-op for them so the data stays idempotent even if the count
        # differs from first run.
        stats = migrate(dry_run=False, data_dir=tmp_path)
        assert stats["already_classified"] == 3  # personal + confidential + secret
        assert stats["classified_public"] == 1  # harvest source re-evaluates → PUBLIC
        assert stats["classified_personal"] == 0
        assert stats["classified_confidential"] == 0
        assert stats["classified_secret"] == 0
        # Verify no new review-queue rows on the second run (data didn't change)
        # — only the session boundary marker gets appended.

    def test_session_boundary_added_on_rerun(self, seeded_layer, tmp_path):
        # First run writes the audit file
        migrate(dry_run=False, data_dir=tmp_path)
        # Add a new PUBLIC-default memory and re-run
        seeded_layer.remember(content="another tiaid task", tags=["tiaid", "task"])
        migrate(dry_run=False, data_dir=tmp_path)
        audit_path = tmp_path / "audit" / "backfill-review.jsonl"
        rows = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        # Original 2 + boundary + 1 new = 4 rows
        assert any("_session_boundary" in row for row in rows)

    def test_explicit_classification_preserved(self, tmp_path):
        """Memories already tagged via Stage 2.C writers are not reclassified."""
        layer = MemoryLayer(tmp_path, backend="local")
        layer.remember(
            content="my own decisions",
            tags=["decision"],
            sensitivity=Sensitivity.CONFIDENTIAL,
        )
        stats = migrate(dry_run=False, data_dir=tmp_path)
        assert stats["already_classified"] == 1
        # And the value is preserved
        reloaded = MemoryLayer(tmp_path, backend="local")
        results = reloaded.store.list_all()
        assert results[0].sensitivity is Sensitivity.CONFIDENTIAL
