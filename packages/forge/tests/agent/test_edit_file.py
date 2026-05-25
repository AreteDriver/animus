"""Tests for tools.edit_file — spec §4.4 + A4."""

from pathlib import Path

import pytest

from animus_forge.agent.tools import edit_file
from animus_forge.agent.types import ToolError


class TestEditFile:
    def test_happy_path_single_replace(self, tmp_path: Path):
        (tmp_path / "f.py").write_text("foo = 1\nbar = 2\n")
        msg = edit_file("f.py", "foo = 1", "foo = 42", tmp_path)
        assert msg == "replaced 1 occurrence"
        assert (tmp_path / "f.py").read_text() == "foo = 42\nbar = 2\n"

    def test_multi_match_rejected(self, tmp_path: Path):
        (tmp_path / "f.py").write_text("x = 1\nx = 1\n")
        with pytest.raises(ToolError, match=r"multiple matches \(2\)"):
            edit_file("f.py", "x = 1", "x = 99", tmp_path)

    def test_zero_match_rejected(self, tmp_path: Path):
        (tmp_path / "f.py").write_text("foo\n")
        with pytest.raises(ToolError, match="string not found"):
            edit_file("f.py", "bar", "baz", tmp_path)

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(ToolError, match="file not found"):
            edit_file("missing.txt", "a", "b", tmp_path)

    def test_directory_target_rejected(self, tmp_path: Path):
        (tmp_path / "d").mkdir()
        with pytest.raises(ToolError, match="not a regular file"):
            edit_file("d", "a", "b", tmp_path)

    def test_binary_file_rejected(self, tmp_path: Path):
        (tmp_path / "b.dat").write_bytes(b"\xff\xfe\x00")
        with pytest.raises(ToolError, match="binary file"):
            edit_file("b.dat", "a", "b", tmp_path)

    def test_path_escape_denied(self, tmp_path: Path):
        with pytest.raises(ToolError, match="path escapes project root"):
            edit_file("/etc/passwd", "a", "b", tmp_path)

    def test_empty_old_string_matches_many(self, tmp_path: Path):
        """Empty `old` would match between every character — must reject."""
        (tmp_path / "f.txt").write_text("ab")
        with pytest.raises(ToolError, match="multiple matches"):
            edit_file("f.txt", "", "x", tmp_path)
