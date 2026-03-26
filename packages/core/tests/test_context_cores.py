"""Tests for Context Core versioning features.

Covers Memory version fields, update_with_version(), get_version_history(),
snapshot/restore, and provenance tagging.
"""

import json
from pathlib import Path

import pytest

from animus.memory import (
    Memory,
    MemoryLayer,
)


@pytest.fixture
def memory_layer(tmp_data_dir: Path) -> MemoryLayer:
    """MemoryLayer backed by LocalMemoryStore for fast, isolated tests."""
    return MemoryLayer(data_dir=tmp_data_dir, backend="local")


# ── 1. Memory version fields ───────────────────────────────────────


class TestMemoryVersionFields:
    """New Memory objects carry sensible version defaults."""

    def test_default_version_fields(self):
        mem = Memory.create(content="hello world")
        assert mem.version == 1
        assert mem.parent_id is None
        assert mem.provenance == "direct"
        assert mem.change_summary is None

    def test_to_dict_includes_version_fields(self):
        mem = Memory.create(
            content="versioned",
            version=3,
            parent_id="abc-123",
            change_summary="content updated",
            provenance="sync",
        )
        d = mem.to_dict()
        assert d["version"] == 3
        assert d["parent_id"] == "abc-123"
        assert d["change_summary"] == "content updated"
        assert d["provenance"] == "sync"

    def test_from_dict_missing_version_fields_uses_defaults(self):
        """Backward compat: older serialised dicts lack version keys."""
        minimal = {
            "id": "old-mem",
            "content": "legacy entry",
            "memory_type": "semantic",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        }
        mem = Memory.from_dict(minimal)
        assert mem.version == 1
        assert mem.parent_id is None
        assert mem.change_summary is None
        assert mem.provenance == "direct"


# ── 2. update_with_version() ───────────────────────────────────────


class TestUpdateWithVersion:
    """Versioned updates create new memories linked to the parent."""

    def test_basic_version_bump(self, memory_layer: MemoryLayer):
        original = memory_layer.remember("alpha content", tags=["test"])

        updated = memory_layer.update_with_version(original.id, content="beta content")

        assert updated is not None
        assert updated.version == 2
        assert updated.parent_id == original.id
        assert updated.content == "beta content"

    def test_auto_generated_change_summary(self, memory_layer: MemoryLayer):
        original = memory_layer.remember("before")
        updated = memory_layer.update_with_version(original.id, content="after")

        assert updated is not None
        assert "content updated" in updated.change_summary

    def test_explicit_change_summary(self, memory_layer: MemoryLayer):
        original = memory_layer.remember("v1")
        updated = memory_layer.update_with_version(
            original.id, content="v2", change_summary="manual note"
        )

        assert updated is not None
        assert updated.change_summary == "manual note"

    def test_parent_unchanged_after_update(self, memory_layer: MemoryLayer):
        original = memory_layer.remember("immutable content")
        memory_layer.update_with_version(original.id, content="new content")

        parent = memory_layer.get_memory(original.id)
        assert parent is not None
        assert parent.content == "immutable content"
        assert parent.version == 1

    def test_update_nonexistent_returns_none(self, memory_layer: MemoryLayer):
        result = memory_layer.update_with_version("does-not-exist", content="whatever")
        assert result is None

    def test_no_op_version_bump(self, memory_layer: MemoryLayer):
        original = memory_layer.remember("same")
        updated = memory_layer.update_with_version(original.id)

        assert updated is not None
        assert updated.version == 2
        assert updated.change_summary == "no-op version bump"
        assert updated.content == "same"

    def test_tags_change_detected(self, memory_layer: MemoryLayer):
        original = memory_layer.remember("tagged", tags=["a"])
        updated = memory_layer.update_with_version(original.id, tags=["a", "b"])

        assert updated is not None
        assert "tags changed" in updated.change_summary

    def test_provenance_carried_through(self, memory_layer: MemoryLayer):
        original = memory_layer.remember("data")
        updated = memory_layer.update_with_version(
            original.id, content="data v2", provenance="sync"
        )

        assert updated is not None
        assert updated.provenance == "sync"


# ── 3. get_version_history() ──────────────────────────────────────


class TestGetVersionHistory:
    """Walking the parent chain returns ordered version history."""

    def test_three_version_chain(self, memory_layer: MemoryLayer):
        v1 = memory_layer.remember("v1")
        v2 = memory_layer.update_with_version(v1.id, content="v2")
        v3 = memory_layer.update_with_version(v2.id, content="v3")

        history = memory_layer.get_version_history(v3.id)
        assert len(history) == 3
        assert history[0].id == v3.id
        assert history[1].id == v2.id
        assert history[2].id == v1.id
        assert [h.version for h in history] == [3, 2, 1]

    def test_limit_parameter(self, memory_layer: MemoryLayer):
        v1 = memory_layer.remember("v1")
        v2 = memory_layer.update_with_version(v1.id, content="v2")
        v3 = memory_layer.update_with_version(v2.id, content="v3")

        history = memory_layer.get_version_history(v3.id, limit=2)
        assert len(history) == 2
        assert history[0].id == v3.id
        assert history[1].id == v2.id

    def test_single_memory_returns_one(self, memory_layer: MemoryLayer):
        mem = memory_layer.remember("solo")
        history = memory_layer.get_version_history(mem.id)
        assert len(history) == 1
        assert history[0].id == mem.id

    def test_nonexistent_returns_empty(self, memory_layer: MemoryLayer):
        history = memory_layer.get_version_history("no-such-id")
        assert history == []


# ── 4. snapshot() and restore_snapshot() ──────────────────────────


class TestSnapshotRestore:
    """Snapshot exports and restores round-trip correctly."""

    def test_snapshot_creates_file(self, memory_layer: MemoryLayer, tmp_data_dir: Path):
        memory_layer.remember("mem-a")
        memory_layer.remember("mem-b")
        memory_layer.remember("mem-c")

        result = memory_layer.snapshot("test_snap")

        assert result["label"] == "test_snap"
        assert result["memory_count"] == 3
        assert "timestamp" in result
        assert Path(result["path"]).exists()

    def test_snapshot_file_in_snapshots_dir(self, memory_layer: MemoryLayer, tmp_data_dir: Path):
        memory_layer.remember("x")
        result = memory_layer.snapshot("check_dir")
        snap_path = Path(result["path"])
        assert snap_path.parent == tmp_data_dir / "snapshots"

    def test_restore_returns_to_original_state(self, memory_layer: MemoryLayer):
        memory_layer.remember("alpha")
        memory_layer.remember("beta")
        memory_layer.remember("gamma")

        snap = memory_layer.snapshot("before_extras")

        memory_layer.remember("delta")
        memory_layer.remember("epsilon")
        assert len(memory_layer.store.list_all()) == 5

        restored_count = memory_layer.restore_snapshot(snap["path"])
        assert restored_count == 3
        assert len(memory_layer.store.list_all()) == 3

    def test_restore_nonexistent_raises(self, memory_layer: MemoryLayer):
        with pytest.raises(FileNotFoundError):
            memory_layer.restore_snapshot("/no/such/snapshot.json")

    def test_snapshot_content_valid_json(self, memory_layer: MemoryLayer):
        memory_layer.remember("json check")
        snap = memory_layer.snapshot("json_test")
        data = json.loads(Path(snap["path"]).read_text())
        assert data["label"] == "json_test"
        assert len(data["memories"]) == 1


# ── 5. Provenance tagging ────────────────────────────────────────


class TestProvenanceTagging:
    """Provenance field is stored and retrievable."""

    def test_sync_provenance(self, memory_layer: MemoryLayer):
        mem = memory_layer.remember("synced data", provenance="sync")
        retrieved = memory_layer.get_memory(mem.id)
        assert retrieved is not None
        assert retrieved.provenance == "sync"

    def test_mcp_provenance(self, memory_layer: MemoryLayer):
        mem = memory_layer.remember("mcp data", provenance="mcp")
        retrieved = memory_layer.get_memory(mem.id)
        assert retrieved is not None
        assert retrieved.provenance == "mcp"

    def test_default_provenance_is_direct(self, memory_layer: MemoryLayer):
        mem = memory_layer.remember("plain data")
        assert mem.provenance == "direct"
        retrieved = memory_layer.get_memory(mem.id)
        assert retrieved is not None
        assert retrieved.provenance == "direct"


# ── 6. snapshot() creates snapshots directory ─────────────────────


class TestSnapshotDirectoryCreation:
    """snapshot() must create the snapshots/ subdirectory if missing."""

    def test_creates_snapshots_dir_from_scratch(self, tmp_path: Path):
        fresh_dir = tmp_path / "fresh_data"
        fresh_dir.mkdir()
        # No snapshots/ subdir exists yet
        assert not (fresh_dir / "snapshots").exists()

        layer = MemoryLayer(data_dir=fresh_dir, backend="local")
        layer.remember("seed")
        result = layer.snapshot("first_ever")

        assert (fresh_dir / "snapshots").is_dir()
        assert Path(result["path"]).exists()
        assert result["memory_count"] == 1
