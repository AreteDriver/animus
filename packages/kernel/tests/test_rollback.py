"""Tests for sandbox rollback management."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from animus_kernel.sandbox.rollback import RollbackManager, Snapshot


# ═══════════════════════════════════════════════════════════════════
# RollbackManager tests
# ═══════════════════════════════════════════════════════════════════


class TestRollbackManager:
    def test_create_snapshot_reads_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            codebase = Path(tmpdir) / "codebase"
            codebase.mkdir()
            (codebase / "main.py").write_text("print('hello')")
            (codebase / "utils.py").write_text("def helper(): pass")

            storage = Path(tmpdir) / "snapshots"
            manager = RollbackManager(storage_path=storage, max_snapshots=5)

            snapshot = manager.create_snapshot(
                files=["main.py", "utils.py"],
                description="Before test change",
                codebase_path=codebase,
            )

            assert isinstance(snapshot, Snapshot)
            assert snapshot.description == "Before test change"
            assert len(snapshot.id) == 8
            assert "main.py" in snapshot.files
            assert "utils.py" in snapshot.files
            assert snapshot.files["main.py"] == "print('hello')"

    def test_create_snapshot_missing_file_graceful(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            codebase = Path(tmpdir) / "codebase"
            codebase.mkdir()
            (codebase / "main.py").write_text("print('hello')")

            storage = Path(tmpdir) / "snapshots"
            manager = RollbackManager(storage_path=storage, max_snapshots=5)

            snapshot = manager.create_snapshot(
                files=["main.py", "missing.py"],
                description="With missing",
                codebase_path=codebase,
            )

            assert "main.py" in snapshot.files
            assert "missing.py" not in snapshot.files

    def test_get_snapshot_loads_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            codebase = Path(tmpdir) / "codebase"
            codebase.mkdir()
            (codebase / "main.py").write_text("original")

            storage = Path(tmpdir) / "snapshots"
            manager = RollbackManager(storage_path=storage, max_snapshots=5)

            created = manager.create_snapshot(
                files=["main.py"],
                description="test",
                codebase_path=codebase,
            )

            # Simulate fresh manager instance loading from disk
            manager2 = RollbackManager(storage_path=storage, max_snapshots=5)
            loaded = manager2.get_snapshot(created.id)

            assert loaded is not None
            assert loaded.id == created.id
            assert loaded.files["main.py"] == "original"

    def test_get_snapshot_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir) / "snapshots"
            manager = RollbackManager(storage_path=storage, max_snapshots=5)
            assert manager.get_snapshot("nonexistent") is None

    def test_rollback_restores_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            codebase = Path(tmpdir) / "codebase"
            codebase.mkdir()
            (codebase / "main.py").write_text("original")

            storage = Path(tmpdir) / "snapshots"
            manager = RollbackManager(storage_path=storage, max_snapshots=5)

            snapshot = manager.create_snapshot(
                files=["main.py"],
                description="Before modification",
                codebase_path=codebase,
            )

            # Modify the file
            (codebase / "main.py").write_text("modified")
            assert (codebase / "main.py").read_text() == "modified"

            # Rollback
            result = manager.rollback(snapshot.id, codebase_path=codebase)
            assert result is True
            assert (codebase / "main.py").read_text() == "original"

    def test_rollback_missing_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir) / "snapshots"
            manager = RollbackManager(storage_path=storage, max_snapshots=5)
            assert manager.rollback("nonexistent") is False

    def test_list_snapshots_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            codebase = Path(tmpdir) / "codebase"
            codebase.mkdir()
            (codebase / "main.py").write_text("v1")

            storage = Path(tmpdir) / "snapshots"
            manager = RollbackManager(storage_path=storage, max_snapshots=10)

            ids = []
            for i in range(3):
                (codebase / "main.py").write_text(f"v{i + 1}")
                snap = manager.create_snapshot(
                    files=["main.py"],
                    description=f"snapshot {i + 1}",
                    codebase_path=codebase,
                )
                ids.append(snap.id)

            recent = manager.list_snapshots(limit=10)
            # Most recent first
            assert recent[0].description == "snapshot 3"
            assert recent[2].description == "snapshot 1"

    def test_cleanup_old_snapshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            codebase = Path(tmpdir) / "codebase"
            codebase.mkdir()
            (codebase / "main.py").write_text("x")

            storage = Path(tmpdir) / "snapshots"
            manager = RollbackManager(storage_path=storage, max_snapshots=2)

            snap1 = manager.create_snapshot(
                files=["main.py"], description="snap1", codebase_path=codebase
            )
            snap2 = manager.create_snapshot(
                files=["main.py"], description="snap2", codebase_path=codebase
            )
            snap3 = manager.create_snapshot(
                files=["main.py"], description="snap3", codebase_path=codebase
            )

            # Oldest should be removed
            assert manager.get_snapshot(snap1.id) is None
            assert manager.get_snapshot(snap2.id) is not None
            assert manager.get_snapshot(snap3.id) is not None

            # Only 2 snapshots remain
            assert len(manager.list_snapshots()) == 2

    def test_delete_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            codebase = Path(tmpdir) / "codebase"
            codebase.mkdir()
            (codebase / "main.py").write_text("x")

            storage = Path(tmpdir) / "snapshots"
            manager = RollbackManager(storage_path=storage, max_snapshots=5)

            snap = manager.create_snapshot(
                files=["main.py"], description="to_delete", codebase_path=codebase
            )

            assert manager.get_snapshot(snap.id) is not None
            result = manager.delete_snapshot(snap.id)
            assert result is True
            assert manager.get_snapshot(snap.id) is None

    def test_delete_snapshot_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir) / "snapshots"
            manager = RollbackManager(storage_path=storage, max_snapshots=5)
            assert manager.delete_snapshot("nonexistent") is False
