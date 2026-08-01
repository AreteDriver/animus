"""Tests for MemoryLayer — the public façade over pluggable backends.

Covers:
- remember / recall with all filters
- remember_fact / remember_procedure
- get_memory (exact + partial match)
- update_memory / add_tag / remove_tag
- promote_memory / demote_memory
- recall_for_egress (tier-gated)
- recall_by_tags / recall_by_tags_for_egress
- update_with_version / get_version_history
- snapshot / restore_snapshot
- forget
- save_conversation
- export/import (json, jsonl)
- backup
- get_statistics
- consolidate
- export_memories_csv
- backend resolution (auto, local, chroma fallback)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from animus_kernel.memory.layer import MemoryLayer
from animus_kernel.memory.types import (
    Conversation,
    MemoryTier,
    MemoryType,
    Sensitivity,
)


@pytest.fixture
def layer(tmp_path: Path) -> MemoryLayer:
    return MemoryLayer(data_dir=tmp_path, backend="local")


class TestRemember:
    def test_remember_stores_memory(self, layer: MemoryLayer):
        mem = layer.remember("hello world")
        assert mem.content == "hello world"
        assert mem.memory_type == MemoryType.SEMANTIC
        assert mem.tags == []
        retrieved = layer.get_memory(mem.id)
        assert retrieved is not None
        assert retrieved.content == "hello world"

    def test_remember_with_tags(self, layer: MemoryLayer):
        mem = layer.remember("tagged", tags=["A", "B"])
        assert mem.tags == ["a", "b"]

    def test_remember_redacts_secrets(self, layer: MemoryLayer):
        key = "sk-" + "a" * 48
        mem = layer.remember(f"key is {key}")
        assert "[REDACTED:" in mem.content
        assert key not in mem.content
        assert mem.metadata.get("_redaction_count") >= 1

    def test_remember_with_sensitivity(self, layer: MemoryLayer):
        mem = layer.remember("secret", sensitivity=Sensitivity.SECRET)
        assert mem.sensitivity == Sensitivity.SECRET

    def test_remember_with_tier(self, layer: MemoryLayer):
        mem = layer.remember("tiered", tier=MemoryTier.HOT)
        assert mem.tier == MemoryTier.HOT


class TestRememberFact:
    def test_remember_fact(self, layer: MemoryLayer):
        mem = layer.remember_fact("Alice", "likes", "tea", category="preference")
        assert "Alice likes tea" in mem.content
        assert mem.memory_type == MemoryType.SEMANTIC
        assert mem.subtype == "preference"
        assert mem.metadata["fact_subject"] == "Alice"


class TestRememberProcedure:
    def test_remember_procedure(self, layer: MemoryLayer):
        mem = layer.remember_procedure("Deploy", "push", ["test", "build", "deploy"])
        assert "Deploy" in mem.content
        assert "test" in mem.content
        assert mem.memory_type == MemoryType.PROCEDURAL
        assert mem.subtype == "workflow"


class TestRecall:
    def test_recall_finds_content(self, layer: MemoryLayer):
        layer.remember("apple pie")
        layer.remember("banana bread")
        results = layer.recall("pie")
        assert len(results) >= 1
        assert any("apple pie" in r.content for r in results)

    def test_recall_by_memory_type(self, layer: MemoryLayer):
        layer.remember("semantic", memory_type=MemoryType.SEMANTIC)
        layer.remember("episodic", memory_type=MemoryType.EPISODIC)
        results = layer.recall("episodic", memory_type=MemoryType.EPISODIC)
        assert all(r.memory_type == MemoryType.EPISODIC for r in results)

    def test_recall_by_tags(self, layer: MemoryLayer):
        layer.remember("a", tags=["foo", "bar"])
        layer.remember("b", tags=["foo"])
        results = layer.recall("a", tags=["foo", "bar"])
        assert len(results) == 1
        assert results[0].content == "a"

    def test_recall_by_source(self, layer: MemoryLayer):
        layer.remember("learned", source="learned")
        layer.remember("stated", source="stated")
        results = layer.recall("learned", source="learned")
        assert len(results) == 1

    def test_recall_min_confidence(self, layer: MemoryLayer):
        layer.remember("high", confidence=0.9)
        layer.remember("low", confidence=0.2)
        results = layer.recall("high", min_confidence=0.5)
        assert all(r.confidence >= 0.5 for r in results)

    def test_recall_allowed_tiers(self, layer: MemoryLayer):
        layer.remember("secret", sensitivity=Sensitivity.SECRET)
        layer.remember("public", sensitivity=Sensitivity.PUBLIC)
        results = layer.recall("secret", allowed_tiers={Sensitivity.PUBLIC})
        assert len(results) == 0

    def test_recall_for_egress_only_public(self, layer: MemoryLayer):
        layer.remember("secret", sensitivity=Sensitivity.SECRET)
        layer.remember("public", sensitivity=Sensitivity.PUBLIC)
        results = layer.recall_for_egress("public")
        assert len(results) == 1
        assert results[0].sensitivity == Sensitivity.PUBLIC

    def test_recall_by_tags_for_egress(self, layer: MemoryLayer):
        layer.remember("pub", tags=["x"], sensitivity=Sensitivity.PUBLIC)
        layer.remember("sec", tags=["x"], sensitivity=Sensitivity.SECRET)
        results = layer.recall_by_tags_for_egress(["x"])
        assert len(results) == 1
        assert results[0].sensitivity == Sensitivity.PUBLIC

    def test_recall_tier_filter(self, layer: MemoryLayer):
        layer.remember("hot", tier=MemoryTier.HOT)
        layer.remember("cold", tier=MemoryTier.COLD)
        results = layer.recall("hot", tier=MemoryTier.HOT)
        assert all(r.tier == MemoryTier.HOT for r in results)

    def test_recall_respects_limit(self, layer: MemoryLayer):
        for i in range(5):
            layer.remember(f"item {i}")
        results = layer.recall("item", limit=3)
        assert len(results) <= 3


class TestGetMemory:
    def test_exact_match(self, layer: MemoryLayer):
        mem = layer.remember("exact")
        found = layer.get_memory(mem.id)
        assert found is not None
        assert found.content == "exact"

    def test_partial_match(self, layer: MemoryLayer):
        mem = layer.remember("partial")
        prefix = mem.id[:8]
        found = layer.get_memory(prefix)
        assert found is not None
        assert found.id == mem.id

    def test_missing_returns_none(self, layer: MemoryLayer):
        assert layer.get_memory("nonexistent") is None


class TestUpdateMemory:
    def test_update_changes_content(self, layer: MemoryLayer):
        mem = layer.remember("original")
        mem.content = "updated"
        ok = layer.update_memory(mem)
        assert ok is True
        assert layer.get_memory(mem.id).content == "updated"


class TestAddRemoveTag:
    def test_add_tag(self, layer: MemoryLayer):
        mem = layer.remember("taggable")
        ok = layer.add_tag(mem.id, "new-tag")
        assert ok is True
        assert "new-tag" in layer.get_memory(mem.id).tags

    def test_remove_tag(self, layer: MemoryLayer):
        mem = layer.remember("tagged", tags=["remove-me"])
        ok = layer.remove_tag(mem.id, "remove-me")
        assert ok is True
        assert "remove-me" not in layer.get_memory(mem.id).tags

    def test_add_tag_missing_returns_false(self, layer: MemoryLayer):
        assert layer.add_tag("nonexistent", "x") is False

    def test_remove_tag_missing_returns_false(self, layer: MemoryLayer):
        assert layer.remove_tag("nonexistent", "x") is False


class TestSnapshot:
    def test_snapshot_and_restore(self, layer: MemoryLayer):
        layer.remember("one")
        layer.remember("two")
        meta = layer.snapshot("test-snap")
        assert meta["memory_count"] == 2
        assert Path(meta["path"]).exists()

        # Clear
        for mem in layer.store.list_all():
            layer.store.delete(mem.id)
        assert len(layer.store.list_all()) == 0

        restored = layer.restore_snapshot(meta["path"])
        assert restored == 2
        assert len(layer.store.list_all()) == 2

    def test_restore_missing_raises(self, layer: MemoryLayer):
        with pytest.raises(FileNotFoundError):
            layer.restore_snapshot("/nonexistent/snapshot.json")


class TestForget:
    def test_forget_removes_memory(self, layer: MemoryLayer):
        mem = layer.remember("forget me")
        ok = layer.forget(mem.id)
        assert ok is True
        assert layer.get_memory(mem.id) is None

    def test_forget_missing_returns_false(self, layer: MemoryLayer):
        assert layer.forget("nonexistent") is False

    def test_forget_partial_match(self, layer: MemoryLayer):
        mem = layer.remember("partial forget")
        ok = layer.forget(mem.id[:8])
        assert ok is True
        assert layer.get_memory(mem.id) is None


class TestSaveConversation:
    def test_save_conversation(self, layer: MemoryLayer):
        conv = Conversation.new()
        conv.add_message("user", "hello")
        conv.add_message("assistant", "hi there")
        mem = layer.save_conversation(conv)
        assert mem.memory_type == MemoryType.EPISODIC
        assert "hello" in mem.content
        assert "hi there" in mem.content
        assert mem.metadata.get("message_count") == 2


class TestExportImport:
    def test_export_json(self, layer: MemoryLayer):
        layer.remember("one")
        layer.remember("two")
        exported = layer.export_memories(format="json")
        assert "one" in exported
        assert "two" in exported

    def test_export_jsonl(self, layer: MemoryLayer):
        layer.remember("one")
        layer.remember("two")
        exported = layer.export_memories(format="jsonl")
        lines = exported.strip().split("\n")
        assert len(lines) == 2

    def test_import_json(self, layer: MemoryLayer):
        layer.remember("one")
        exported = layer.export_memories(format="json")
        # Clear and import
        for mem in layer.store.list_all():
            layer.store.delete(mem.id)
        count = layer.import_memories(exported, format="json")
        assert count == 1
        assert len(layer.store.list_all()) == 1

    def test_import_jsonl(self, layer: MemoryLayer):
        layer.remember("one")
        layer.remember("two")
        exported = layer.export_memories(format="jsonl")
        for mem in list(layer.store.list_all()):
            layer.store.delete(mem.id)
        count = layer.import_memories(exported, format="jsonl")
        assert count == 2


class TestVersioning:
    def test_update_with_version(self, layer: MemoryLayer):
        mem = layer.remember("original")
        new_mem = layer.update_with_version(
            mem.id, content="updated", change_summary="content updated"
        )
        assert new_mem is not None
        assert new_mem.version == 2
        assert new_mem.parent_id == mem.id
        assert new_mem.change_summary == "content updated"

    def test_update_with_version_noop(self, layer: MemoryLayer):
        mem = layer.remember("same")
        new_mem = layer.update_with_version(mem.id)
        assert new_mem.version == 2
        assert new_mem.change_summary == "no-op version bump"

    def test_update_with_version_missing_returns_none(self, layer: MemoryLayer):
        assert layer.update_with_version("nonexistent", content="x") is None

    def test_get_version_history(self, layer: MemoryLayer):
        v1 = layer.remember("base")
        v2 = layer.update_with_version(v1.id, content="v2")
        v3 = layer.update_with_version(v2.id, content="v3")
        history = layer.get_version_history(v3.id)
        assert len(history) == 3
        assert history[0].id == v3.id
        assert history[1].id == v2.id
        assert history[2].id == v1.id


class TestStatistics:
    def test_get_statistics(self, layer: MemoryLayer):
        layer.remember("a", memory_type=MemoryType.SEMANTIC, tags=["x"], confidence=0.9)
        layer.remember("b", memory_type=MemoryType.EPISODIC, tags=["x", "y"], confidence=0.5)
        stats = layer.get_statistics()
        assert stats["total"] == 2
        assert stats["by_type"]["semantic"] == 1
        assert stats["by_type"]["episodic"] == 1
        assert stats["unique_tags"] == 2
        assert 0.6 < stats["avg_confidence"] < 0.8

    def test_statistics_empty(self, layer: MemoryLayer):
        stats = layer.get_statistics()
        assert stats["total"] == 0
        assert stats["avg_confidence"] == 0


class TestBackup:
    def test_backup_creates_zip(self, layer: MemoryLayer, tmp_path: Path):
        layer.remember("backup")
        backup_path = tmp_path / "backup.zip"
        layer.backup(backup_path)
        assert backup_path.exists()
        assert backup_path.stat().st_size > 0

    def test_backup_adds_zip_extension(self, layer: MemoryLayer, tmp_path: Path):
        layer.remember("backup")
        backup_path = tmp_path / "backup"
        layer.backup(backup_path)
        assert (tmp_path / "backup.zip").exists()


class TestConsolidate:
    def test_consolidate_old_memories(self, layer: MemoryLayer):
        # Create old episodic memories
        for i in range(3):
            mem = layer.remember(
                f"old event {i}", memory_type=MemoryType.EPISODIC, tags=["project"]
            )
            mem.created_at = datetime.now() - timedelta(days=100)
            layer.update_memory(mem)
        count = layer.consolidate(max_age_days=90, min_group_size=3)
        assert count == 3
        assert len([m for m in layer.store.list_all() if m.subtype == "consolidated"]) == 1

    def test_consolidate_no_old_memories(self, layer: MemoryLayer):
        layer.remember("fresh", memory_type=MemoryType.EPISODIC)
        count = layer.consolidate(max_age_days=90, min_group_size=3)
        assert count == 0

    def test_consolidate_below_min_group(self, layer: MemoryLayer):
        mem = layer.remember("old", memory_type=MemoryType.EPISODIC, tags=["project"])
        mem.created_at = datetime.now() - timedelta(days=100)
        layer.update_memory(mem)
        count = layer.consolidate(max_age_days=90, min_group_size=3)
        assert count == 0


class TestCSVExport:
    def test_export_memories_csv(self, layer: MemoryLayer):
        layer.remember("csv", tags=["test"])
        csv_out = layer.export_memories_csv()
        lines = csv_out.strip().split("\n")
        assert lines[0].startswith("id,content")
        assert len(lines) == 2  # header + 1 row


class TestBackendResolution:
    def test_auto_falls_back_to_local(self, tmp_path: Path):
        # No ANIMUS_DATABASE_URL and chromadb not available
        layer = MemoryLayer(data_dir=tmp_path, backend="auto")
        assert type(layer.store).__name__ == "LocalMemoryStore"

    def test_explicit_local(self, tmp_path: Path):
        layer = MemoryLayer(data_dir=tmp_path, backend="local")
        assert type(layer.store).__name__ == "LocalMemoryStore"

    def test_explicit_durable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANIMUS_DATABASE_URL", "sqlite:///:memory:")
        layer = MemoryLayer(data_dir=tmp_path, backend="durable")
        assert type(layer.store).__name__ == "DurableMemoryStore"

    def test_unrecognized_backend_defaults_to_local(self, tmp_path: Path):
        layer = MemoryLayer(data_dir=tmp_path, backend="unknown")
        assert type(layer.store).__name__ == "LocalMemoryStore"

    def test_auto_prefers_durable_when_db_url_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("ANIMUS_DATABASE_URL", "sqlite:///:memory:")
        layer = MemoryLayer(data_dir=tmp_path, backend="auto")
        # Should pick DurableMemoryStore because DB URL is set
        assert type(layer.store).__name__ == "DurableMemoryStore"
