"""Tests for ``animus.workflows.code_ingest``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from animus.memory import MemoryLayer
from animus.workflows.code_ingest import (
    CodeIngestResult,
    IngestError,
    _hash_file,
    ingest_codebase,
)


class TestIngestCodebase:
    def test_ingests_python_file(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("def hello():\n    pass\n")
        memory = MemoryLayer(tmp_path / "data", backend="local")

        result = ingest_codebase(tmp_path, memory=memory, tags=["test"])
        assert result.stored_count >= 1
        assert result.success
        assert len(result.errors) == 0

    def test_manifest_written(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("def hello(): pass\n")
        memory = MemoryLayer(tmp_path / "data", backend="local")

        result = ingest_codebase(tmp_path, memory=memory, write_manifest=True)
        assert result.manifest_path is not None
        assert result.manifest_path.exists()
        data = json.loads(result.manifest_path.read_text())
        assert data["version"] == "1.0"
        assert "files" in data

    def test_skip_existing(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("def hello(): pass\n")
        memory = MemoryLayer(tmp_path / "data", backend="local")

        r1 = ingest_codebase(tmp_path, memory=memory)
        assert r1.stored_count >= 1

        r2 = ingest_codebase(tmp_path, memory=memory, skip_existing=True)
        assert r2.skipped_count >= 1
        assert r2.stored_count == 0

    def test_not_a_directory(self, tmp_path: Path):
        memory = MemoryLayer(tmp_path / "data", backend="local")
        result = ingest_codebase(tmp_path / "nope", memory=memory)
        assert not result.success
        assert any(e.stage == "fatal" for e in result.errors)

    def test_excluded_files_not_ingested(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("def main(): pass\n")
        (tmp_path / "test_utils.py").write_text("def test_utils(): pass\n")
        memory = MemoryLayer(tmp_path / "data", backend="local")

        result = ingest_codebase(tmp_path, memory=memory, exclude=["test_*"])
        assert "test_utils.py" not in {
            err.path.split(":")[0] for err in result.errors
        }
        # Should still have main.py chunks
        assert result.stored_count >= 1


class TestHashFile:
    def test_hash_stable(self, tmp_path: Path):
        p = tmp_path / "stable.txt"
        p.write_text("hello")
        h1 = _hash_file(p)
        h2 = _hash_file(p)
        assert h1 == h2
        assert len(h1) == 16

    def test_hash_changes_with_content(self, tmp_path: Path):
        p = tmp_path / "a.txt"
        p.write_text("a")
        h1 = _hash_file(p)
        p.write_text("b")
        h2 = _hash_file(p)
        assert h1 != h2

    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert _hash_file(tmp_path / "nope") == ""
