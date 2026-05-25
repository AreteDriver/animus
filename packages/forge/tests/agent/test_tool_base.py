"""Tests for tools.base — path validation primitive."""

from pathlib import Path

import pytest

from animus_forge.agent.tools.base import resolve_within_root
from animus_forge.agent.types import ToolError


class TestResolveWithinRoot:
    def test_relative_path_inside_root(self, tmp_path: Path):
        target = tmp_path / "file.txt"
        target.touch()
        resolved = resolve_within_root("file.txt", tmp_path)
        assert resolved == target.resolve()

    def test_nested_relative_path(self, tmp_path: Path):
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        target = sub / "c.txt"
        target.touch()
        resolved = resolve_within_root("a/b/c.txt", tmp_path)
        assert resolved == target.resolve()

    def test_absolute_path_inside_root_allowed(self, tmp_path: Path):
        target = tmp_path / "file.txt"
        target.touch()
        resolved = resolve_within_root(str(target), tmp_path)
        assert resolved == target.resolve()

    def test_absolute_path_outside_root_raises(self, tmp_path: Path):
        with pytest.raises(ToolError, match="path escapes project root"):
            resolve_within_root("/etc/passwd", tmp_path)

    def test_dotdot_escape_raises(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        with pytest.raises(ToolError, match="path escapes project root"):
            resolve_within_root("../../etc/passwd", sub)

    def test_symlink_pointing_outside_raises(self, tmp_path: Path):
        outside = tmp_path.parent / "outside.txt"
        outside.touch()
        try:
            link = tmp_path / "link.txt"
            link.symlink_to(outside)
            with pytest.raises(ToolError, match="path escapes project root"):
                resolve_within_root("link.txt", tmp_path)
        finally:
            outside.unlink(missing_ok=True)

    def test_nonexistent_path_returned_if_inside(self, tmp_path: Path):
        """Writers create new files — non-existent paths must resolve cleanly."""
        resolved = resolve_within_root("new_file.txt", tmp_path)
        assert resolved == (tmp_path / "new_file.txt").resolve()

    def test_unresolvable_root_raises(self, tmp_path: Path):
        bogus_root = tmp_path / "does-not-exist"
        with pytest.raises(ToolError):
            resolve_within_root("anything.txt", bogus_root)
