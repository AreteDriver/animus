"""Tests for tools.write_file — spec §4.3 + A4."""

from pathlib import Path

import pytest

from animus_forge.agent.tools import write_file
from animus_forge.agent.types import ToolError


class TestWriteFile:
    def test_happy_path_returns_byte_count_message(self, tmp_path: Path):
        msg = write_file("out.txt", "hello", tmp_path)
        assert msg == "wrote 5 bytes to out.txt"
        assert (tmp_path / "out.txt").read_text() == "hello"

    def test_overwrite_without_confirmation(self, tmp_path: Path):
        (tmp_path / "f.txt").write_text("old")
        write_file("f.txt", "new", tmp_path)
        assert (tmp_path / "f.txt").read_text() == "new"

    def test_creates_parent_dirs(self, tmp_path: Path):
        write_file("a/b/c.txt", "deep", tmp_path)
        assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"

    def test_byte_count_for_unicode(self, tmp_path: Path):
        """Spec says 'bytes'; unicode chars count as their UTF-8 byte length."""
        msg = write_file("u.txt", "héllo", tmp_path)  # 6 bytes in UTF-8
        assert msg == "wrote 6 bytes to u.txt"

    def test_empty_content_writes_empty_file(self, tmp_path: Path):
        msg = write_file("empty.txt", "", tmp_path)
        assert msg == "wrote 0 bytes to empty.txt"
        assert (tmp_path / "empty.txt").read_text() == ""

    def test_path_escape_denied(self, tmp_path: Path):
        with pytest.raises(ToolError, match="path escapes project root"):
            write_file("/tmp/evil.txt", "x", tmp_path)

    def test_dotdot_escape_denied(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        with pytest.raises(ToolError, match="path escapes project root"):
            write_file("../../../etc/evil.txt", "x", sub)
