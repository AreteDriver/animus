"""Tests for animus.ogma.grounding — only-read-paths-are-cited invariant."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from animus.ogma.grounding import (
    GroundingError,
    verify_animus_gap,
)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A minimal repo skeleton with three known files in packages/."""
    (tmp_path / ".git").mkdir()  # satisfy the "is a git repo" check
    pkg = tmp_path / "packages" / "core" / "animus"
    pkg.mkdir(parents=True)
    (pkg / "memory.py").write_text(
        "# memory subsystem\n"
        "def retrieve_with_chromadb(query):\n"
        "    # learned test-time memory lives here some day\n"
        "    return None\n",
        encoding="utf-8",
    )
    (pkg / "cognitive.py").write_text(
        "# cognitive layer — unrelated content\nclass CognitiveLayer:\n    pass\n",
        encoding="utf-8",
    )
    (pkg / "unrelated.py").write_text(
        "# nothing useful here\ndef widget():\n    return 42\n",
        encoding="utf-8",
    )
    return tmp_path


class TestVerifyAnimusGap:
    def test_status_none_when_zero_hits(self, fake_repo: Path):
        result = verify_animus_gap("entirely-absent-concept-xyz123", repo_root=fake_repo)
        assert result.status == "NONE"
        assert result.paths_read == []

    def test_partial_status_for_one_match(self, fake_repo: Path):
        # "learned test-time memory" matches memory.py only.
        result = verify_animus_gap("learned test-time memory", repo_root=fake_repo)
        assert result.status == "PARTIAL"
        assert result.paths_read, "must record at least one path when status != NONE"
        assert any("memory.py" in p for p in result.paths_read)
        # critically: the unrelated files MUST NOT appear
        assert not any("unrelated.py" in p for p in result.paths_read)
        assert not any("cognitive.py" in p for p in result.paths_read)

    def test_full_status_for_three_or_more_matches(self, fake_repo: Path):
        # Inject a third file mentioning the same concept so status hits FULL.
        pkg = fake_repo / "packages" / "core" / "animus"
        (pkg / "retrieval.py").write_text(
            "# retrieval module\ndef widget_retrieval():\n    return 'widget'\n",
            encoding="utf-8",
        )
        (pkg / "second.py").write_text(
            "# more widget stuff\ndef widget_two():\n    return 'widget'\n",
            encoding="utf-8",
        )
        result = verify_animus_gap("widget", repo_root=fake_repo)
        assert result.status == "FULL"
        assert len(result.paths_read) >= 3

    def test_paths_read_carries_line_range(self, fake_repo: Path):
        result = verify_animus_gap("retrieve_with_chromadb", repo_root=fake_repo)
        assert result.paths_read
        rel_with_range = result.paths_read[0]
        # shape: "packages/core/animus/memory.py:start-end"
        assert ":" in rel_with_range
        path_part, range_part = rel_with_range.rsplit(":", 1)
        start, end = (int(x) for x in range_part.split("-"))
        assert start >= 1
        assert end >= start

    def test_empty_concept_raises_value_error(self, fake_repo: Path):
        with pytest.raises(ValueError, match="non-empty"):
            verify_animus_gap("", repo_root=fake_repo)
        with pytest.raises(ValueError, match="non-empty"):
            verify_animus_gap("   ", repo_root=fake_repo)

    def test_non_git_repo_raises_grounding_error(self, tmp_path: Path):
        # no .git dir → not a repo
        (tmp_path / "packages").mkdir()
        with pytest.raises(GroundingError, match="not a git repo"):
            verify_animus_gap("anything", repo_root=tmp_path)

    def test_grep_missing_yields_status_none(self, fake_repo: Path, monkeypatch):
        # Simulate grep absent from PATH; subprocess.run raises FileNotFoundError.
        def fake_run(*a, **kw):
            raise FileNotFoundError("no grep")

        monkeypatch.setattr("animus.ogma.grounding.subprocess.run", fake_run)
        result = verify_animus_gap("learned test-time memory", repo_root=fake_repo)
        assert result.status == "NONE"

    def test_grep_timeout_raises(self, fake_repo: Path, monkeypatch):
        def fake_run(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="grep", timeout=30)

        monkeypatch.setattr("animus.ogma.grounding.subprocess.run", fake_run)
        with pytest.raises(GroundingError, match="timeout"):
            verify_animus_gap("learned test-time memory", repo_root=fake_repo)

    def test_max_files_caps_reads(self, fake_repo: Path):
        pkg = fake_repo / "packages" / "core" / "animus"
        for i in range(10):
            (pkg / f"hit_{i}.py").write_text(
                f"# match {i}\ndef widget_{i}(): return 'widget'\n", encoding="utf-8"
            )
        result = verify_animus_gap("widget", repo_root=fake_repo, max_files=3)
        assert result.status == "FULL"
        assert len(result.paths_read) == 3
