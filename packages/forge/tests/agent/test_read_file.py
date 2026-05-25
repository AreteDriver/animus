"""Tests for tools.read_file — spec §4.2 + A4."""

from pathlib import Path

import pytest

from animus_forge.agent.tools import read_file
from animus_forge.agent.types import ToolError


class TestReadFile:
    def test_happy_path_full_read(self, tmp_path: Path):
        (tmp_path / "f.txt").write_text("hello world")
        assert read_file("f.txt", tmp_path) == "hello world"

    def test_offset_only(self, tmp_path: Path):
        (tmp_path / "f.txt").write_text("hello world")
        assert read_file("f.txt", tmp_path, offset=6) == "world"

    def test_limit_only(self, tmp_path: Path):
        (tmp_path / "f.txt").write_text("hello world")
        assert read_file("f.txt", tmp_path, limit=5) == "hello"

    def test_offset_and_limit(self, tmp_path: Path):
        (tmp_path / "f.txt").write_text("hello world")
        assert read_file("f.txt", tmp_path, offset=6, limit=3) == "wor"

    def test_limit_beyond_eof_returns_partial(self, tmp_path: Path):
        (tmp_path / "f.txt").write_text("hello")
        assert read_file("f.txt", tmp_path, limit=100) == "hello"

    def test_path_escape_denied(self, tmp_path: Path):
        with pytest.raises(ToolError, match="path escapes project root"):
            read_file("/etc/passwd", tmp_path)

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(ToolError, match="file not found"):
            read_file("nope.txt", tmp_path)

    def test_directory_not_file(self, tmp_path: Path):
        (tmp_path / "dir").mkdir()
        with pytest.raises(ToolError, match="not a regular file"):
            read_file("dir", tmp_path)

    def test_binary_file_raises(self, tmp_path: Path):
        (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\xfd\x00\x01")
        with pytest.raises(ToolError, match="binary file"):
            read_file("bin.dat", tmp_path)

    def test_negative_offset_raises(self, tmp_path: Path):
        (tmp_path / "f.txt").write_text("hi")
        with pytest.raises(ToolError, match="negative offset"):
            read_file("f.txt", tmp_path, offset=-1)

    def test_negative_limit_raises(self, tmp_path: Path):
        (tmp_path / "f.txt").write_text("hi")
        with pytest.raises(ToolError, match="negative limit"):
            read_file("f.txt", tmp_path, limit=-1)
