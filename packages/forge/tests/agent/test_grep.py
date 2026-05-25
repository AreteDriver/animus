"""Tests for tools.grep — spec §4.6 + A4."""

from pathlib import Path

import pytest

from animus_forge.agent.tools import grep
from animus_forge.agent.types import ToolError


class TestGrepHappyPath:
    def test_finds_pattern_in_file(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("import os\nprint('hello')\n")
        hits = grep("import", tmp_path)
        assert len(hits) == 1
        assert hits[0]["file"] == "a.py"
        assert hits[0]["line"] == 1
        assert "import" in hits[0]["text"]

    def test_finds_across_files(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("foo\n")
        (tmp_path / "b.py").write_text("foo\n")
        hits = grep("foo", tmp_path)
        files = {h["file"] for h in hits}
        assert files == {"a.py", "b.py"}

    def test_glob_filter(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("foo\n")
        (tmp_path / "b.txt").write_text("foo\n")
        hits = grep("foo", tmp_path, glob="*.py")
        assert len(hits) == 1
        assert hits[0]["file"] == "a.py"

    def test_case_insensitive(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("FOO\nfoo\nFoo\n")
        hits = grep("foo", tmp_path, case_insensitive=True)
        assert len(hits) == 3

    def test_case_sensitive_by_default(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("FOO\nfoo\n")
        hits = grep("foo", tmp_path)
        assert len(hits) == 1
        assert hits[0]["line"] == 2

    def test_no_matches_returns_empty(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("nothing here\n")
        assert grep("missing-pattern", tmp_path) == []

    def test_regex_metachars(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("foo.bar\nfoobar\n")
        hits = grep(r"foo\.", tmp_path)
        assert len(hits) == 1
        assert hits[0]["line"] == 1

    def test_subdirectory_search(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.txt").write_text("target\n")
        hits = grep("target", tmp_path, path="sub")
        assert len(hits) == 1
        assert "sub" in hits[0]["file"]

    def test_search_single_file(self, tmp_path: Path):
        f = tmp_path / "single.txt"
        f.write_text("line1\nfoo\nline3\n")
        hits = grep("foo", tmp_path, path="single.txt")
        assert len(hits) == 1
        assert hits[0]["line"] == 2


class TestGrepLimits:
    def test_result_cap_at_100(self, tmp_path: Path):
        """101 matches → 100 results + sentinel."""
        lines = "\n".join(["foo"] * 200)
        (tmp_path / "many.txt").write_text(lines)
        hits = grep("foo", tmp_path)
        # 100 hit dicts + 1 sentinel
        assert sum(1 for h in hits if h.get("_truncated")) == 1
        actual_hits = [h for h in hits if not h.get("_truncated")]
        assert len(actual_hits) == 100

    def test_under_cap_no_sentinel(self, tmp_path: Path):
        lines = "\n".join(["foo"] * 10)
        (tmp_path / "few.txt").write_text(lines)
        hits = grep("foo", tmp_path)
        assert not any(h.get("_truncated") for h in hits)
        assert len(hits) == 10


class TestGrepErrors:
    def test_path_escape_denied(self, tmp_path: Path):
        with pytest.raises(ToolError, match="path escapes project root"):
            grep("anything", tmp_path, path="/etc")

    def test_invalid_regex_raises(self, tmp_path: Path):
        with pytest.raises(ToolError, match="invalid regex"):
            grep("(unclosed", tmp_path)

    def test_nonexistent_path_raises(self, tmp_path: Path):
        with pytest.raises(ToolError, match="path not found"):
            grep("x", tmp_path, path="missing-dir")

    def test_binary_files_skipped_silently(self, tmp_path: Path):
        """Binary files in the glob must not crash; just skip them."""
        (tmp_path / "good.txt").write_text("foo\n")
        (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00foo\x00")
        hits = grep("foo", tmp_path)
        assert len(hits) == 1
        assert hits[0]["file"] == "good.txt"
