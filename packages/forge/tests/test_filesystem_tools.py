"""Tests for filesystem tools module."""

from pathlib import Path

import pytest

from animus_forge.tools.filesystem import FilesystemTools
from animus_forge.tools.models import (
    EditProposal,
    ProposalStatus,
)
from animus_forge.tools.safety import PathValidator, SecurityError


class TestPathValidator:
    """Tests for PathValidator security checks."""

    def test_validate_path_within_project(self, tmp_path: Path):
        """Test that paths within project are validated."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        validator = PathValidator(tmp_path)
        resolved = validator.validate_path("test.txt")

        assert resolved == test_file

    def test_validate_path_absolute(self, tmp_path: Path):
        """Test that absolute paths within project work."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        validator = PathValidator(tmp_path)
        resolved = validator.validate_path(str(test_file))

        assert resolved == test_file

    def test_validate_path_traversal_blocked(self, tmp_path: Path):
        """Test that path traversal is blocked."""
        validator = PathValidator(tmp_path)

        with pytest.raises(SecurityError, match="outside allowed directories"):
            validator.validate_path("../etc/passwd")

    def test_validate_path_symlink_traversal_blocked(self, tmp_path: Path):
        """Test that symlinks outside project are blocked."""
        # Create a symlink pointing outside project
        symlink = tmp_path / "escape"
        symlink.symlink_to("/etc")

        validator = PathValidator(tmp_path)

        with pytest.raises(SecurityError, match="outside allowed directories"):
            validator.validate_path("escape/passwd")

    def test_validate_path_excluded_pattern(self, tmp_path: Path):
        """Test that excluded patterns are blocked."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        validator = PathValidator(tmp_path)

        with pytest.raises(SecurityError, match="excluded pattern"):
            validator.validate_path(".git")

    def test_validate_path_excluded_nested(self, tmp_path: Path):
        """Test that nested excluded patterns are blocked."""
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        nested = node_modules / "some-package"
        nested.mkdir()

        validator = PathValidator(tmp_path)

        with pytest.raises(SecurityError, match="excluded pattern"):
            validator.validate_path("node_modules/some-package")

    def test_validate_file_for_read_success(self, tmp_path: Path):
        """Test successful file read validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        validator = PathValidator(tmp_path)
        resolved = validator.validate_file_for_read("test.txt")

        assert resolved == test_file

    def test_validate_file_for_read_not_file(self, tmp_path: Path):
        """Test that directories fail file read validation."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        validator = PathValidator(tmp_path)

        with pytest.raises(SecurityError, match="not a file"):
            validator.validate_file_for_read("subdir")

    def test_validate_file_for_read_too_large(self, tmp_path: Path):
        """Test that large files are rejected."""
        test_file = tmp_path / "large.txt"
        # Create a file slightly over 1MB
        test_file.write_bytes(b"x" * (1024 * 1024 + 1))

        validator = PathValidator(tmp_path, max_file_size=1024 * 1024)

        with pytest.raises(SecurityError, match="exceeds size limit"):
            validator.validate_file_for_read("large.txt")

    def test_validate_directory_success(self, tmp_path: Path):
        """Test successful directory validation."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        validator = PathValidator(tmp_path)
        resolved = validator.validate_directory("subdir")

        assert resolved == subdir

    def test_validate_directory_not_dir(self, tmp_path: Path):
        """Test that files fail directory validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        validator = PathValidator(tmp_path)

        with pytest.raises(SecurityError, match="not a directory"):
            validator.validate_directory("test.txt")

    def test_validate_file_for_write_success(self, tmp_path: Path):
        """Test successful file write validation."""
        validator = PathValidator(tmp_path)

        # File doesn't exist yet - that's OK for write
        resolved = validator.validate_file_for_write("new_file.txt")
        assert resolved == tmp_path / "new_file.txt"

    def test_validate_file_for_write_parent_missing(self, tmp_path: Path):
        """Test that write fails if parent doesn't exist."""
        validator = PathValidator(tmp_path)

        with pytest.raises(SecurityError, match="Parent directory does not exist"):
            validator.validate_file_for_write("nonexistent/file.txt")

    def test_allowed_paths(self, tmp_path: Path):
        """Test that additional allowed paths work."""
        project = tmp_path / "project"
        project.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        extra_file = extra / "test.txt"
        extra_file.write_text("content")

        validator = PathValidator(project, allowed_paths=[str(extra)])

        # Should be able to access file in extra path
        resolved = validator.validate_path(str(extra_file))
        assert resolved == extra_file

    def test_is_excluded(self, tmp_path: Path):
        """Test is_excluded helper."""
        validator = PathValidator(tmp_path)

        assert validator.is_excluded(".git")
        assert validator.is_excluded("node_modules/pkg")
        assert validator.is_excluded("__pycache__")
        assert not validator.is_excluded("src/main.py")

    def test_project_path_must_exist(self, tmp_path: Path):
        """Test that project path must be an existing directory."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(SecurityError, match="does not exist"):
            PathValidator(nonexistent)


class TestFilesystemTools:
    """Tests for FilesystemTools operations."""

    @pytest.fixture
    def tools(self, tmp_path: Path) -> FilesystemTools:
        """Create FilesystemTools with test directory."""
        validator = PathValidator(tmp_path)
        return FilesystemTools(validator)

    def test_read_file_basic(self, tmp_path: Path, tools: FilesystemTools):
        """Test basic file reading."""
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3")

        content = tools.read_file("test.py")

        assert content.path == "test.py"
        assert content.line_count == 3
        assert "line1" in content.content
        assert "line2" in content.content

    def test_read_file_with_line_numbers(self, tmp_path: Path, tools: FilesystemTools):
        """Test that file content includes line numbers."""
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3")

        content = tools.read_file("test.py")

        # Should have line numbers in format "     1\tcontent"
        assert "\t" in content.content
        lines = content.content.split("\n")
        assert "1" in lines[0]

    def test_read_file_range(self, tmp_path: Path, tools: FilesystemTools):
        """Test reading a range of lines."""
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\nline4\nline5")

        content = tools.read_file("test.py", start_line=2, end_line=4)

        assert content.truncated is True
        assert "line2" in content.content
        assert "line3" in content.content
        assert "line4" in content.content

    def test_list_files_basic(self, tmp_path: Path, tools: FilesystemTools):
        """Test basic directory listing."""
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.txt").write_text("b")
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        listing = tools.list_files(".")

        assert listing.total_files == 2
        assert listing.total_dirs == 1
        assert len(listing.entries) == 3

    def test_list_files_with_pattern(self, tmp_path: Path, tools: FilesystemTools):
        """Test listing with glob pattern."""
        (tmp_path / "test.py").write_text("a")
        (tmp_path / "test.txt").write_text("b")
        (tmp_path / "other.py").write_text("c")

        listing = tools.list_files(".", pattern="*.py")

        assert listing.total_files == 2
        names = [e.name for e in listing.entries]
        assert "test.py" in names
        assert "other.py" in names

    def test_list_files_recursive(self, tmp_path: Path, tools: FilesystemTools):
        """Test recursive listing."""
        (tmp_path / "top.txt").write_text("a")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("b")

        listing = tools.list_files(".", recursive=True)

        paths = [e.path for e in listing.entries]
        assert "top.txt" in paths
        assert "subdir/nested.txt" in paths or "subdir\\nested.txt" in paths

    def test_list_files_excludes_patterns(self, tmp_path: Path, tools: FilesystemTools):
        """Test that excluded patterns are filtered."""
        (tmp_path / "normal.txt").write_text("a")
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("b")

        listing = tools.list_files(".", recursive=True)

        paths = [e.path for e in listing.entries]
        assert "normal.txt" in paths
        assert ".git" not in paths
        assert ".git/config" not in paths

    def test_search_code_basic(self, tmp_path: Path, tools: FilesystemTools):
        """Test basic code search."""
        (tmp_path / "test.py").write_text("def hello():\n    print('hello')")
        (tmp_path / "other.py").write_text("def world():\n    print('world')")

        result = tools.search_code("hello")

        assert result.pattern == "hello"
        assert result.total_matches >= 2
        assert result.files_searched >= 1

    def test_search_code_regex(self, tmp_path: Path, tools: FilesystemTools):
        """Test regex search."""
        (tmp_path / "test.py").write_text("def func1(): pass\ndef func2(): pass")

        result = tools.search_code(r"def \w+\(\)")

        assert result.total_matches == 2

    def test_search_code_file_pattern(self, tmp_path: Path, tools: FilesystemTools):
        """Test search with file pattern filter."""
        (tmp_path / "test.py").write_text("hello world")
        (tmp_path / "test.txt").write_text("hello world")

        result = tools.search_code("hello", file_pattern="*.py")

        assert result.total_matches == 1
        assert result.matches[0].path == "test.py"

    def test_search_code_case_insensitive(self, tmp_path: Path, tools: FilesystemTools):
        """Test case insensitive search."""
        (tmp_path / "test.py").write_text("Hello\nHELLO\nhello")

        result = tools.search_code("hello", case_sensitive=False)

        assert result.total_matches == 3

    def test_get_structure(self, tmp_path: Path, tools: FilesystemTools):
        """Test project structure overview."""
        (tmp_path / "main.py").write_text("a")
        (tmp_path / "utils.py").write_text("b")
        src = tmp_path / "src"
        src.mkdir()
        (src / "module.py").write_text("c")

        structure = tools.get_structure()

        assert structure.total_files == 3
        assert structure.total_dirs == 1
        assert ".py" in structure.file_types
        assert structure.file_types[".py"] == 3
        assert "src/" in structure.tree

    def test_get_structure_depth_limit(self, tmp_path: Path, tools: FilesystemTools):
        """Test structure depth limiting."""
        # Create deep nesting
        current = tmp_path
        for i in range(10):
            current = current / f"level{i}"
            current.mkdir()
            (current / "file.txt").write_text("a")

        structure = tools.get_structure(max_depth=2)

        # Should only show first few levels
        assert "level0" in structure.tree
        assert "level1" in structure.tree
        # Deep levels should not appear
        assert "level9" not in structure.tree

    def test_glob_files(self, tmp_path: Path, tools: FilesystemTools):
        """Test glob file matching."""
        (tmp_path / "test.py").write_text("a")
        (tmp_path / "other.py").write_text("b")
        src = tmp_path / "src"
        src.mkdir()
        (src / "module.py").write_text("c")

        files = tools.glob_files("**/*.py")

        assert len(files) == 3
        assert "test.py" in files


class TestEditProposalModels:
    """Tests for edit proposal models."""

    def test_proposal_status_enum(self):
        """Test ProposalStatus enum values."""
        assert ProposalStatus.PENDING.value == "pending"
        assert ProposalStatus.APPROVED.value == "approved"
        assert ProposalStatus.REJECTED.value == "rejected"
        assert ProposalStatus.APPLIED.value == "applied"
        assert ProposalStatus.FAILED.value == "failed"

    def test_edit_proposal_creation(self):
        """Test EditProposal model creation."""
        proposal = EditProposal(
            id="test-123",
            session_id="session-456",
            file_path="src/main.py",
            old_content="old code",
            new_content="new code",
            description="Update main function",
        )

        assert proposal.id == "test-123"
        assert proposal.status == ProposalStatus.PENDING
        assert proposal.old_content == "old code"
        assert proposal.new_content == "new code"

    def test_edit_proposal_new_file(self):
        """Test EditProposal for new file (no old_content)."""
        proposal = EditProposal(
            id="test-123",
            session_id="session-456",
            file_path="src/new_file.py",
            new_content="new file content",
        )

        assert proposal.old_content is None
        assert proposal.new_content == "new file content"


class TestFilesystemToolsEdgeCases:
    """Edge case tests for filesystem tools."""

    def test_read_file_binary_fallback(self, tmp_path: Path):
        """Test that non-UTF8 files fall back to latin-1."""
        test_file = tmp_path / "binary.txt"
        # Write bytes that aren't valid UTF-8
        test_file.write_bytes(b"Hello \xff\xfe World")

        validator = PathValidator(tmp_path)
        tools = FilesystemTools(validator)

        content = tools.read_file("binary.txt")
        assert "Hello" in content.content
        assert "World" in content.content

    def test_list_files_max_results(self, tmp_path: Path):
        """Test max results limiting."""
        for i in range(200):
            (tmp_path / f"file{i:03d}.txt").write_text("a")

        validator = PathValidator(tmp_path)
        tools = FilesystemTools(validator, max_results=50)

        listing = tools.list_files(".")

        assert len(listing.entries) == 50
        assert listing.total_files == 200
        assert listing.truncated is True

    def test_search_code_max_results(self, tmp_path: Path):
        """Test search max results."""
        # Create file with many matches
        content = "\n".join([f"match line {i}" for i in range(200)])
        (tmp_path / "many_matches.txt").write_text(content)

        validator = PathValidator(tmp_path)
        tools = FilesystemTools(validator, max_results=50)

        result = tools.search_code("match")

        assert len(result.matches) == 50
        assert result.truncated is True

    def test_search_invalid_regex(self, tmp_path: Path):
        """Test that invalid regex raises SecurityError."""
        (tmp_path / "test.txt").write_text("content")

        validator = PathValidator(tmp_path)
        tools = FilesystemTools(validator)

        with pytest.raises(SecurityError, match="Invalid regex"):
            tools.search_code("[invalid(")

    def test_empty_directory(self, tmp_path: Path):
        """Test operations on empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        validator = PathValidator(tmp_path)
        tools = FilesystemTools(validator)

        listing = tools.list_files("empty")
        assert listing.total_files == 0
        assert listing.total_dirs == 0

        structure = tools.get_structure()
        assert "empty/" in structure.tree
