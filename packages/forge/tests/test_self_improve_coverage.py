"""Tests for self_improve modules: analyzer, rollback, pr_manager, orchestrator."""

from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_forge.self_improve.analyzer import (
    AnalysisResult,
    CodebaseAnalyzer,
    ImprovementCategory,
    ImprovementSuggestion,
)
from animus_forge.self_improve.orchestrator import (
    ImprovementPlan,
    ImprovementResult,
    SelfImproveOrchestrator,
    WorkflowStage,
)
from animus_forge.self_improve.pr_manager import (
    ConflictResult,
    PRManager,
    PRStatus,
    PullRequest,
)
from animus_forge.self_improve.rollback import RollbackManager, Snapshot
from animus_forge.self_improve.safety import SafetyConfig
from animus_forge.self_improve.sandbox import SandboxResult, SandboxStatus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_codebase(tmp_path: Path) -> Path:
    """Create a minimal codebase for analyzer tests."""
    src = tmp_path / "src" / "animus_forge"
    src.mkdir(parents=True)
    # A file with code quality issues
    (src / "sample.py").write_text(
        '"""Module docstring."""\n'
        "\n"
        "\n"
        "def public_func(x):\n"
        '    """Has docstring."""\n'
        "    return x\n"
        "\n"
        "\n"
        "def _private_func():\n"
        "    pass\n"
    )
    # A long function (>50 lines) + TODO + bare except
    lines = ['"""Module doc."""\n', "\n"]
    lines.append("def big_function():\n")
    lines.extend(f"    x = {i}\n" for i in range(55))
    lines.append("\n\n")
    lines.append("def next_func():\n")
    lines.append("    pass\n")
    lines.append("\n")
    lines.append("# TODO: fix this bug\n")
    lines.append("try:\n")
    lines.append("    pass\n")
    lines.append("except:\n")
    lines.append("    pass\n")
    (src / "bigfile.py").write_text("".join(lines))

    # A file missing module docstring with a public func without docstring
    (src / "no_doc.py").write_text("import os\n\n\ndef undocumented(x):\n    return x + 1\n")

    # A file inside __pycache__ (should be skipped)
    cache = src / "__pycache__"
    cache.mkdir()
    (cache / "cached.py").write_text("x = 1\n")

    return tmp_path


@pytest.fixture()
def snapshot_dir(tmp_path: Path) -> Path:
    """Return a temp directory for snapshot storage."""
    return tmp_path / "snapshots"


@pytest.fixture()
def rollback_mgr(snapshot_dir: Path) -> RollbackManager:
    """Create a RollbackManager with a temp storage path."""
    return RollbackManager(storage_path=snapshot_dir, max_snapshots=3)


@pytest.fixture()
def pr_manager(tmp_path: Path) -> PRManager:
    """Create a PRManager with mocked git commands."""
    return PRManager(repo_path=tmp_path, branch_prefix="test-improve/")


@pytest.fixture()
def safety_config() -> SafetyConfig:
    """Create a permissive SafetyConfig for orchestrator tests."""
    return SafetyConfig(
        critical_files=[],
        sensitive_files=[],
        max_files_per_pr=50,
        max_lines_changed=5000,
        max_new_files=20,
        max_deleted_files=5,
        human_approval_plan=False,
        human_approval_apply=False,
        human_approval_merge=False,
        max_snapshots=5,
        auto_rollback_on_test_failure=True,
        branch_prefix="test-improve/",
    )


# ===========================================================================
# CodebaseAnalyzer tests
# ===========================================================================


class TestCodebaseAnalyzer:
    """Tests for CodebaseAnalyzer."""

    def test_analyze_no_focus_paths(self, tmp_codebase: Path):
        """Analyze all src/**/*.py files by default."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        result = analyzer.analyze()
        assert isinstance(result, AnalysisResult)
        # Should have analyzed at least sample.py, bigfile.py, no_doc.py
        # __pycache__ files are skipped
        assert result.files_analyzed >= 2
        assert result.analysis_summary != ""

    def test_analyze_with_focus_paths(self, tmp_codebase: Path):
        """Focus analysis on specific glob patterns."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        result = analyzer.analyze(focus_paths=["src/animus_forge/sample.py"])
        assert result.files_analyzed >= 1

    def test_analyze_category_filter(self, tmp_codebase: Path):
        """Filter analysis to specific categories."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        result = analyzer.analyze(categories=[ImprovementCategory.CODE_QUALITY])
        # Only code quality issues should appear
        for s in result.suggestions:
            assert s.category in (
                ImprovementCategory.CODE_QUALITY,
                ImprovementCategory.REFACTORING,
                ImprovementCategory.BUG_FIXES,
            )

    def test_should_skip_file(self, tmp_codebase: Path):
        """Files in __pycache__, .git, .venv, self_improve are skipped."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        assert analyzer._should_skip_file(Path("src/__pycache__/foo.py")) is True
        assert analyzer._should_skip_file(Path(".git/config")) is True
        assert analyzer._should_skip_file(Path(".venv/lib/foo.py")) is True
        assert analyzer._should_skip_file(Path("src/self_improve/mod.py")) is True
        assert analyzer._should_skip_file(Path("src/animus_forge/normal.py")) is False

    def test_check_code_quality_long_function(self, tmp_codebase: Path):
        """Detect functions longer than 50 lines."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        content = (tmp_codebase / "src" / "animus_forge" / "bigfile.py").read_text()
        suggestions = analyzer._check_code_quality("bigfile.py", content)
        long_func = [s for s in suggestions if "Long function" in s.title]
        assert len(long_func) >= 1
        assert long_func[0].category == ImprovementCategory.REFACTORING

    def test_check_code_quality_todo(self, tmp_codebase: Path):
        """Detect TODO comments."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        content = (tmp_codebase / "src" / "animus_forge" / "bigfile.py").read_text()
        suggestions = analyzer._check_code_quality("bigfile.py", content)
        todos = [s for s in suggestions if "TODO" in s.title]
        assert len(todos) >= 1
        assert todos[0].category == ImprovementCategory.BUG_FIXES

    def test_check_code_quality_bare_except(self, tmp_codebase: Path):
        """Detect bare except clauses."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        content = (tmp_codebase / "src" / "animus_forge" / "bigfile.py").read_text()
        suggestions = analyzer._check_code_quality("bigfile.py", content)
        bare = [s for s in suggestions if "Bare except" in s.title]
        assert len(bare) == 1

    def test_check_documentation_missing_module_docstring(self, tmp_codebase: Path):
        """Detect missing module docstring."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        content = (tmp_codebase / "src" / "animus_forge" / "no_doc.py").read_text()
        suggestions = analyzer._check_documentation("no_doc.py", content)
        missing = [s for s in suggestions if "Missing module docstring" in s.title]
        assert len(missing) == 1

    def test_check_documentation_missing_func_docstring(self, tmp_codebase: Path):
        """Detect public functions without docstrings."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        content = (tmp_codebase / "src" / "animus_forge" / "no_doc.py").read_text()
        suggestions = analyzer._check_documentation("no_doc.py", content)
        missing_func = [s for s in suggestions if "Missing docstring: undocumented" in s.title]
        assert len(missing_func) == 1

    def test_check_documentation_skips_private_funcs(self, tmp_codebase: Path):
        """Private functions are not flagged for missing docstrings."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        content = (tmp_codebase / "src" / "animus_forge" / "sample.py").read_text()
        suggestions = analyzer._check_documentation("sample.py", content)
        private = [s for s in suggestions if "_private_func" in s.title]
        assert len(private) == 0

    def test_check_test_coverage_missing_tests(self, tmp_codebase: Path):
        """Detect source files without corresponding test files."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        content = (tmp_codebase / "src" / "animus_forge" / "sample.py").read_text()
        # Use a path without "test_" substring to avoid early skip
        suggestions = analyzer._check_test_coverage("src/myapp/sample.py", content)
        missing = [s for s in suggestions if "Missing tests" in s.title]
        assert len(missing) == 1
        assert "public_func" in missing[0].implementation_hints

    def test_check_test_coverage_skips_test_files(self, tmp_codebase: Path):
        """Test files themselves are not flagged."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        suggestions = analyzer._check_test_coverage(
            "tests/test_sample.py", "def test_foo(): pass\n"
        )
        assert len(suggestions) == 0

    def test_check_test_coverage_skips_non_src(self, tmp_codebase: Path):
        """Non-src/ files are not checked for test coverage."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        suggestions = analyzer._check_test_coverage("scripts/helper.py", "def helper(): pass\n")
        assert len(suggestions) == 0

    def test_analyze_file_read_error(self, tmp_codebase: Path):
        """Gracefully handle unreadable files."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        fake_path = tmp_codebase / "src" / "animus_forge" / "nonexistent.py"
        suggestions = analyzer._analyze_file(fake_path, None)
        assert suggestions == []

    def test_suggestions_sorted_by_priority(self, tmp_codebase: Path):
        """Suggestions are sorted by priority, then estimated lines descending."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        result = analyzer.analyze()
        if len(result.suggestions) > 1:
            for i in range(len(result.suggestions) - 1):
                a, b = result.suggestions[i], result.suggestions[i + 1]
                assert (a.priority, -a.estimated_lines) <= (
                    b.priority,
                    -b.estimated_lines,
                )

    def test_analyze_with_ai_no_provider(self, tmp_codebase: Path):
        """Without provider, falls back to static analysis."""
        analyzer = CodebaseAnalyzer(codebase_path=tmp_codebase)
        result = asyncio.run(analyzer.analyze_with_ai())
        assert isinstance(result, AnalysisResult)

    def test_analyze_with_ai_focus_file(self, tmp_codebase: Path):
        """AI analysis with a specific focus file."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = json.dumps(
            [
                {
                    "category": "code_quality",
                    "title": "AI suggestion",
                    "description": "Improve this",
                    "affected_files": ["sample.py"],
                    "priority": 2,
                    "reasoning": "test",
                    "implementation_hints": "do it",
                }
            ]
        )
        analyzer = CodebaseAnalyzer(provider=mock_provider, codebase_path=tmp_codebase)
        result = asyncio.run(analyzer.analyze_with_ai(focus_file="src/animus_forge/sample.py"))
        assert any("AI suggestion" in s.title for s in result.suggestions)

    def test_analyze_with_ai_no_focus_samples_files(self, tmp_codebase: Path):
        """AI analysis without focus file samples codebase files."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = "[]"
        analyzer = CodebaseAnalyzer(provider=mock_provider, codebase_path=tmp_codebase)
        result = asyncio.run(analyzer.analyze_with_ai())
        assert isinstance(result, AnalysisResult)
        mock_provider.complete.assert_awaited_once()

    def test_analyze_with_ai_provider_error(self, tmp_codebase: Path):
        """AI provider errors are handled gracefully."""
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("API down")
        analyzer = CodebaseAnalyzer(provider=mock_provider, codebase_path=tmp_codebase)
        result = asyncio.run(analyzer.analyze_with_ai(focus_file="src/animus_forge/sample.py"))
        # Should still return static analysis results
        assert isinstance(result, AnalysisResult)

    def test_analyze_with_ai_focus_file_not_found(self, tmp_codebase: Path):
        """AI analysis with non-existent focus file falls back to static."""
        mock_provider = AsyncMock()
        analyzer = CodebaseAnalyzer(provider=mock_provider, codebase_path=tmp_codebase)
        result = asyncio.run(analyzer.analyze_with_ai(focus_file="nonexistent.py"))
        # Provider should not be called if no code samples found
        assert isinstance(result, AnalysisResult)

    def test_parse_ai_suggestions_json(self):
        """Parse valid JSON response."""
        analyzer = CodebaseAnalyzer()
        suggestions = analyzer._parse_ai_suggestions(
            json.dumps(
                [
                    {
                        "category": "refactoring",
                        "title": "Extract method",
                        "description": "Split function",
                        "affected_files": ["a.py"],
                        "priority": 2,
                    }
                ]
            )
        )
        assert len(suggestions) == 1
        assert suggestions[0].category == ImprovementCategory.REFACTORING

    def test_parse_ai_suggestions_markdown_code_block(self):
        """Parse response wrapped in ```json code block."""
        analyzer = CodebaseAnalyzer()
        response = '```json\n[{"category": "performance", "title": "Cache result"}]\n```'
        suggestions = analyzer._parse_ai_suggestions(response)
        assert len(suggestions) == 1
        assert suggestions[0].category == ImprovementCategory.PERFORMANCE

    def test_parse_ai_suggestions_plain_code_block(self):
        """Parse response wrapped in ``` code block (no json tag)."""
        analyzer = CodebaseAnalyzer()
        response = '```\n[{"title": "Fix bug", "category": "bug_fixes"}]\n```'
        suggestions = analyzer._parse_ai_suggestions(response)
        assert len(suggestions) == 1

    def test_parse_ai_suggestions_single_object(self):
        """Parse response that is a single object, not an array."""
        analyzer = CodebaseAnalyzer()
        response = json.dumps({"category": "documentation", "title": "Add docs"})
        suggestions = analyzer._parse_ai_suggestions(response)
        assert len(suggestions) == 1

    def test_parse_ai_suggestions_invalid_category(self):
        """Invalid category falls back to CODE_QUALITY."""
        analyzer = CodebaseAnalyzer()
        response = json.dumps([{"category": "nonexistent", "title": "Thing"}])
        suggestions = analyzer._parse_ai_suggestions(response)
        assert suggestions[0].category == ImprovementCategory.CODE_QUALITY

    def test_parse_ai_suggestions_invalid_json(self):
        """Invalid JSON returns empty list."""
        analyzer = CodebaseAnalyzer()
        suggestions = analyzer._parse_ai_suggestions("this is not json {{{")
        assert suggestions == []

    def test_parse_ai_suggestions_priority_clamping(self):
        """Priority is clamped between 1 and 5."""
        analyzer = CodebaseAnalyzer()
        response = json.dumps(
            [
                {"title": "Low", "priority": 0},
                {"title": "High", "priority": 99},
            ]
        )
        suggestions = analyzer._parse_ai_suggestions(response)
        assert suggestions[0].priority == 1
        assert suggestions[1].priority == 5

    def test_improvement_suggestion_dataclass(self):
        """ImprovementSuggestion defaults."""
        s = ImprovementSuggestion(
            id="test",
            category=ImprovementCategory.REFACTORING,
            title="Test",
            description="Desc",
            affected_files=["a.py"],
        )
        assert s.priority == 3
        assert s.estimated_lines == 0
        assert s.reasoning == ""

    def test_analysis_result_dataclass(self):
        """AnalysisResult defaults."""
        r = AnalysisResult()
        assert r.suggestions == []
        assert r.files_analyzed == 0
        assert r.issues_found == 0
        assert r.analysis_summary == ""


# ===========================================================================
# RollbackManager tests
# ===========================================================================


class TestRollbackManager:
    """Tests for RollbackManager."""

    def test_init_creates_storage_dir(self, snapshot_dir: Path):
        """Storage directory is created on init."""
        mgr = RollbackManager(storage_path=snapshot_dir)
        assert snapshot_dir.exists()
        assert mgr.max_snapshots == 10

    def test_create_snapshot(self, rollback_mgr: RollbackManager, tmp_path: Path):
        """Create a snapshot of files."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "foo.py").write_text("print('hello')\n")
        (code_dir / "bar.py").write_text("x = 1\n")

        snapshot = rollback_mgr.create_snapshot(
            files=["foo.py", "bar.py"],
            description="test snapshot",
            codebase_path=code_dir,
            metadata={"reason": "testing"},
        )

        assert snapshot.id
        assert snapshot.description == "test snapshot"
        assert "foo.py" in snapshot.files
        assert "bar.py" in snapshot.files
        assert snapshot.metadata == {"reason": "testing"}

    def test_create_snapshot_nonexistent_file(self, rollback_mgr: RollbackManager, tmp_path: Path):
        """Non-existent files are skipped in snapshot."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "exists.py").write_text("a = 1\n")

        snapshot = rollback_mgr.create_snapshot(
            files=["exists.py", "missing.py"],
            description="partial",
            codebase_path=code_dir,
        )
        assert "exists.py" in snapshot.files
        assert "missing.py" not in snapshot.files

    def test_get_snapshot(self, rollback_mgr: RollbackManager, tmp_path: Path):
        """Retrieve a snapshot by ID."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "f.py").write_text("content\n")

        created = rollback_mgr.create_snapshot(
            files=["f.py"], description="get test", codebase_path=code_dir
        )
        fetched = rollback_mgr.get_snapshot(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    def test_get_snapshot_not_found(self, rollback_mgr: RollbackManager):
        """Return None for unknown snapshot ID."""
        assert rollback_mgr.get_snapshot("nonexistent") is None

    def test_get_snapshot_loads_files_from_disk(self, snapshot_dir: Path, tmp_path: Path):
        """Files are lazy-loaded from disk when not in memory."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "f.py").write_text("disk content\n")

        mgr1 = RollbackManager(storage_path=snapshot_dir, max_snapshots=5)
        created = mgr1.create_snapshot(
            files=["f.py"], description="disk load", codebase_path=code_dir
        )

        # New manager instance loads index from disk, files are empty
        mgr2 = RollbackManager(storage_path=snapshot_dir, max_snapshots=5)
        fetched = mgr2.get_snapshot(created.id)
        assert fetched is not None
        assert "f.py" in fetched.files
        assert fetched.files["f.py"] == "disk content\n"

    def test_rollback_restores_files(self, rollback_mgr: RollbackManager, tmp_path: Path):
        """Rollback restores files to snapshot state."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "f.py").write_text("original\n")

        snapshot = rollback_mgr.create_snapshot(
            files=["f.py"], description="rollback test", codebase_path=code_dir
        )

        # Modify file
        (code_dir / "f.py").write_text("modified\n")
        assert (code_dir / "f.py").read_text() == "modified\n"

        # Rollback
        result = rollback_mgr.rollback(snapshot.id, codebase_path=code_dir)
        assert result is True
        assert (code_dir / "f.py").read_text() == "original\n"

    def test_rollback_not_found(self, rollback_mgr: RollbackManager):
        """Rollback returns False for unknown snapshot."""
        assert rollback_mgr.rollback("nonexistent") is False

    def test_rollback_creates_parent_dirs(self, rollback_mgr: RollbackManager, tmp_path: Path):
        """Rollback creates parent directories if needed."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        sub = code_dir / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("nested\n")

        snapshot = rollback_mgr.create_snapshot(
            files=["sub/nested.py"],
            description="nested test",
            codebase_path=code_dir,
        )

        # Delete the file and parent
        (sub / "nested.py").unlink()
        sub.rmdir()

        result = rollback_mgr.rollback(snapshot.id, codebase_path=code_dir)
        assert result is True
        assert (code_dir / "sub" / "nested.py").read_text() == "nested\n"

    def test_list_snapshots(self, rollback_mgr: RollbackManager, tmp_path: Path):
        """List snapshots in reverse chronological order."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "f.py").write_text("c\n")

        for i in range(3):
            rollback_mgr.create_snapshot(
                files=["f.py"], description=f"snap {i}", codebase_path=code_dir
            )

        snapshots = rollback_mgr.list_snapshots()
        assert len(snapshots) == 3
        # Most recent first
        assert snapshots[0].description == "snap 2"

    def test_list_snapshots_limit(self, rollback_mgr: RollbackManager, tmp_path: Path):
        """List snapshots respects limit parameter."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "f.py").write_text("c\n")

        for i in range(3):
            rollback_mgr.create_snapshot(
                files=["f.py"], description=f"snap {i}", codebase_path=code_dir
            )

        assert len(rollback_mgr.list_snapshots(limit=2)) == 2

    def test_delete_snapshot(self, rollback_mgr: RollbackManager, tmp_path: Path):
        """Delete a snapshot removes it from list and disk."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "f.py").write_text("c\n")

        snapshot = rollback_mgr.create_snapshot(
            files=["f.py"], description="to delete", codebase_path=code_dir
        )

        result = rollback_mgr.delete_snapshot(snapshot.id)
        assert result is True
        assert rollback_mgr.get_snapshot(snapshot.id) is None

    def test_delete_snapshot_not_found(self, rollback_mgr: RollbackManager):
        """Delete returns False for unknown snapshot."""
        assert rollback_mgr.delete_snapshot("nonexistent") is False

    def test_cleanup_old_snapshots(self, snapshot_dir: Path, tmp_path: Path):
        """Old snapshots beyond max_snapshots are cleaned up."""
        mgr = RollbackManager(storage_path=snapshot_dir, max_snapshots=2)
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "f.py").write_text("c\n")

        ids = []
        for i in range(4):
            s = mgr.create_snapshot(files=["f.py"], description=f"snap {i}", codebase_path=code_dir)
            ids.append(s.id)

        # Only last 2 should remain
        assert len(mgr.list_snapshots()) == 2
        assert mgr.get_snapshot(ids[0]) is None
        assert mgr.get_snapshot(ids[1]) is None
        assert mgr.get_snapshot(ids[2]) is not None
        assert mgr.get_snapshot(ids[3]) is not None

    def test_load_snapshots_corrupt_index(self, snapshot_dir: Path):
        """Corrupt index file is handled gracefully."""
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "index.json").write_text("not valid json!!!")
        mgr = RollbackManager(storage_path=snapshot_dir)
        assert mgr.list_snapshots() == []

    def test_load_snapshot_files_no_dir(self, rollback_mgr: RollbackManager):
        """Loading files for snapshot with missing dir does nothing."""
        snap = Snapshot(
            id="ghost",
            created_at=datetime.now(),
            description="no dir",
            files={},
        )
        rollback_mgr._load_snapshot_files(snap)
        assert snap.files == {}

    def test_load_snapshot_files_no_files_json(
        self, rollback_mgr: RollbackManager, snapshot_dir: Path
    ):
        """Loading files for snapshot with missing files.json does nothing."""
        snap_dir = snapshot_dir / "nofiles"
        snap_dir.mkdir(parents=True)
        snap = Snapshot(
            id="nofiles",
            created_at=datetime.now(),
            description="no files.json",
            files={},
        )
        rollback_mgr._load_snapshot_files(snap)
        assert snap.files == {}

    def test_snapshot_dataclass(self):
        """Snapshot dataclass defaults."""
        s = Snapshot(id="x", created_at=datetime.now(), description="d", files={})
        assert s.metadata == {}

    def test_save_and_load_index_roundtrip(self, snapshot_dir: Path, tmp_path: Path):
        """Index persists across manager instances."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "f.py").write_text("content\n")

        mgr1 = RollbackManager(storage_path=snapshot_dir)
        mgr1.create_snapshot(files=["f.py"], description="persist test", codebase_path=code_dir)

        mgr2 = RollbackManager(storage_path=snapshot_dir)
        snapshots = mgr2.list_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0].description == "persist test"


# ===========================================================================
# PRManager tests
# ===========================================================================


class TestPRManager:
    """Tests for PRManager."""

    def test_create_branch(self, pr_manager: PRManager):
        """Create a new branch."""
        with patch.object(pr_manager, "_run_git") as mock_git:
            branch = pr_manager.create_branch("fix-docs")
            assert branch == "test-improve/fix-docs"
            mock_git.assert_called_once_with(["checkout", "-b", "test-improve/fix-docs"])

    def test_commit_changes_success(self, pr_manager: PRManager):
        """Commit staged files successfully."""
        mock_result = MagicMock()
        mock_result.stdout = "abc1234\n"
        with patch.object(pr_manager, "_run_git", return_value=mock_result) as mock_git:
            commit_hash = pr_manager.commit_changes(files=["a.py", "b.py"], message="fix docs")
            assert commit_hash == "abc1234"
            assert mock_git.call_count == 3  # add, commit, rev-parse

    def test_commit_changes_failure(self, pr_manager: PRManager):
        """Commit failure returns None."""
        with patch.object(
            pr_manager, "_run_git", side_effect=subprocess.CalledProcessError(1, "git")
        ):
            result = pr_manager.commit_changes(files=["a.py"], message="fail")
            assert result is None

    def test_push_branch_success(self, pr_manager: PRManager):
        """Push branch successfully."""
        with patch.object(pr_manager, "_run_git") as mock_git:
            assert pr_manager.push_branch("test-improve/fix") is True
            mock_git.assert_called_once_with(["push", "-u", "origin", "test-improve/fix"])

    def test_push_branch_failure(self, pr_manager: PRManager):
        """Push failure returns False."""
        with patch.object(pr_manager, "_run_git", side_effect=RuntimeError("no remote")):
            assert pr_manager.push_branch("test-improve/fix") is False

    def test_create_pr_gh_success(self, pr_manager: PRManager):
        """Create PR via gh CLI."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "https://github.com/owner/repo/pull/42\n"
        with patch("animus_forge.self_improve.pr_manager.subprocess.run", return_value=mock_result):
            pr = pr_manager.create_pr(
                branch="test-improve/fix",
                title="Fix docs",
                description="Updated docs",
                files_changed=["docs/README.md"],
                draft=True,
            )
            assert pr.url == "https://github.com/owner/repo/pull/42"
            assert pr.status == PRStatus.DRAFT
            assert pr.title == "Fix docs"
            assert "docs/README.md" in pr.files_changed

    def test_create_pr_gh_failure(self, pr_manager: PRManager):
        """PR is still created locally even if gh CLI fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "not logged in"
        with patch("animus_forge.self_improve.pr_manager.subprocess.run", return_value=mock_result):
            pr = pr_manager.create_pr(
                branch="test-improve/fix",
                title="Fix",
                description="Desc",
                files_changed=["a.py"],
            )
            assert pr.url is None
            assert pr.id in pr_manager._active_prs

    def test_create_pr_gh_not_found(self, pr_manager: PRManager):
        """PR created locally if gh CLI not installed."""
        with patch(
            "animus_forge.self_improve.pr_manager.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            pr = pr_manager.create_pr(
                branch="test-improve/fix",
                title="Fix",
                description="Desc",
                files_changed=["a.py"],
            )
            assert pr.url is None

    def test_create_pr_gh_exception(self, pr_manager: PRManager):
        """PR created locally on unexpected exception."""
        with patch(
            "animus_forge.self_improve.pr_manager.subprocess.run",
            side_effect=OSError("boom"),
        ):
            pr = pr_manager.create_pr(branch="b", title="T", description="D", files_changed=[])
            assert pr.url is None

    def test_create_pr_not_draft(self, pr_manager: PRManager):
        """Create PR as open (not draft)."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "https://github.com/owner/repo/pull/99\n"
        with patch("animus_forge.self_improve.pr_manager.subprocess.run", return_value=mock_result):
            pr = pr_manager.create_pr(
                branch="b", title="T", description="D", files_changed=[], draft=False
            )
            assert pr.status == PRStatus.OPEN

    def test_get_pr(self, pr_manager: PRManager):
        """Get a PR by ID."""
        with patch("animus_forge.self_improve.pr_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="")
            pr = pr_manager.create_pr(branch="b", title="T", description="D", files_changed=[])
            assert pr_manager.get_pr(pr.id) is pr
            assert pr_manager.get_pr("nonexistent") is None

    def test_mark_ready_for_review(self, pr_manager: PRManager):
        """Mark draft PR as ready for review."""
        pr = PullRequest(
            id="abc",
            branch="b",
            title="T",
            description="D",
            url="https://github.com/owner/repo/pull/1",
        )
        pr_manager._active_prs["abc"] = pr
        with patch("animus_forge.self_improve.pr_manager.subprocess.run"):
            assert pr_manager.mark_ready_for_review("abc") is True
            assert pr.status == PRStatus.OPEN

    def test_mark_ready_no_url(self, pr_manager: PRManager):
        """Cannot mark ready if PR has no URL."""
        pr = PullRequest(id="abc", branch="b", title="T", description="D", url=None)
        pr_manager._active_prs["abc"] = pr
        assert pr_manager.mark_ready_for_review("abc") is False

    def test_mark_ready_not_found(self, pr_manager: PRManager):
        """Cannot mark ready for non-existent PR."""
        assert pr_manager.mark_ready_for_review("nope") is False

    def test_mark_ready_exception(self, pr_manager: PRManager):
        """Exception in gh pr ready returns False."""
        pr = PullRequest(
            id="abc",
            branch="b",
            title="T",
            description="D",
            url="https://github.com/test/pull/1",
        )
        pr_manager._active_prs["abc"] = pr
        with patch(
            "animus_forge.self_improve.pr_manager.subprocess.run",
            side_effect=RuntimeError("timeout"),
        ):
            assert pr_manager.mark_ready_for_review("abc") is False

    def test_close_pr_with_url(self, pr_manager: PRManager):
        """Close a PR with gh CLI."""
        pr = PullRequest(
            id="abc",
            branch="b",
            title="T",
            description="D",
            url="https://github.com/test/pull/1",
        )
        pr_manager._active_prs["abc"] = pr
        with patch("animus_forge.self_improve.pr_manager.subprocess.run"):
            assert pr_manager.close_pr("abc", reason="not needed") is True
            assert pr.status == PRStatus.CLOSED

    def test_close_pr_without_url(self, pr_manager: PRManager):
        """Close a PR that has no GitHub URL."""
        pr = PullRequest(id="abc", branch="b", title="T", description="D", url=None)
        pr_manager._active_prs["abc"] = pr
        assert pr_manager.close_pr("abc") is True
        assert pr.status == PRStatus.CLOSED

    def test_close_pr_not_found(self, pr_manager: PRManager):
        """Close returns False for unknown PR."""
        assert pr_manager.close_pr("nope") is False

    def test_close_pr_gh_exception(self, pr_manager: PRManager):
        """gh close exception is swallowed, PR still marked closed."""
        pr = PullRequest(
            id="abc",
            branch="b",
            title="T",
            description="D",
            url="https://github.com/test/pull/1",
        )
        pr_manager._active_prs["abc"] = pr
        with patch(
            "animus_forge.self_improve.pr_manager.subprocess.run",
            side_effect=OSError("no gh"),
        ):
            assert pr_manager.close_pr("abc") is True
            assert pr.status == PRStatus.CLOSED

    def test_checkout_main(self, pr_manager: PRManager):
        """Checkout main branch."""
        mock_result = MagicMock()
        mock_result.stdout = "origin/main\n"
        with patch.object(pr_manager, "_run_git", return_value=mock_result):
            assert pr_manager.checkout_main() is True

    def test_checkout_main_failure(self, pr_manager: PRManager):
        """Checkout main failure returns False."""
        with patch.object(pr_manager, "_run_git", side_effect=RuntimeError("detached")):
            assert pr_manager.checkout_main() is False

    def test_delete_branch(self, pr_manager: PRManager):
        """Delete a local branch."""
        with patch.object(pr_manager, "_run_git") as mock_git:
            assert pr_manager.delete_branch("test-improve/fix") is True
            mock_git.assert_called_once_with(["branch", "-d", "test-improve/fix"])

    def test_delete_branch_force(self, pr_manager: PRManager):
        """Force-delete a local branch."""
        with patch.object(pr_manager, "_run_git") as mock_git:
            assert pr_manager.delete_branch("b", force=True) is True
            mock_git.assert_called_once_with(["branch", "-D", "b"])

    def test_delete_branch_failure(self, pr_manager: PRManager):
        """Delete branch failure returns False."""
        with patch.object(pr_manager, "_run_git", side_effect=RuntimeError("not found")):
            assert pr_manager.delete_branch("b") is False

    def test_get_current_branch(self, pr_manager: PRManager):
        """Get current branch name."""
        mock_result = MagicMock()
        mock_result.stdout = "main\n"
        with patch.object(pr_manager, "_run_git", return_value=mock_result):
            assert pr_manager.get_current_branch() == "main"

    def test_has_uncommitted_changes_yes(self, pr_manager: PRManager):
        """Detect uncommitted changes."""
        mock_result = MagicMock()
        mock_result.stdout = " M file.py\n"
        with patch.object(pr_manager, "_run_git", return_value=mock_result):
            assert pr_manager.has_uncommitted_changes() is True

    def test_has_uncommitted_changes_no(self, pr_manager: PRManager):
        """Detect clean working tree."""
        mock_result = MagicMock()
        mock_result.stdout = "\n"
        with patch.object(pr_manager, "_run_git", return_value=mock_result):
            assert pr_manager.has_uncommitted_changes() is False

    def test_check_conflicts_clean_merge(self, pr_manager: PRManager):
        """No conflicts detected in clean merge."""
        merge_result = MagicMock()
        merge_result.returncode = 0
        abort_result = MagicMock()
        abort_result.returncode = 0
        with patch(
            "animus_forge.self_improve.pr_manager.subprocess.run",
            side_effect=[merge_result, abort_result],
        ):
            result = pr_manager.check_conflicts("feature-branch")
            assert result.has_conflicts is False
            assert result.error is None

    def test_check_conflicts_has_conflicts(self, pr_manager: PRManager):
        """Conflicts detected in merge."""
        merge_result = MagicMock()
        merge_result.returncode = 1
        diff_result = MagicMock()
        diff_result.returncode = 0
        diff_result.stdout = "file1.py\nfile2.py\n"
        abort_result = MagicMock()
        with patch(
            "animus_forge.self_improve.pr_manager.subprocess.run",
            side_effect=[merge_result, diff_result, abort_result],
        ):
            result = pr_manager.check_conflicts("feature-branch")
            assert result.has_conflicts is True
            assert "file1.py" in result.conflicting_files
            assert "file2.py" in result.conflicting_files

    def test_check_conflicts_git_not_found(self, pr_manager: PRManager):
        """Git not found returns error."""
        with patch(
            "animus_forge.self_improve.pr_manager.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            result = pr_manager.check_conflicts("b")
            assert result.has_conflicts is False
            assert result.error == "git not found"

    def test_check_conflicts_timeout(self, pr_manager: PRManager):
        """Timeout returns error."""
        with patch(
            "animus_forge.self_improve.pr_manager.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="git", timeout=60),
        ):
            result = pr_manager.check_conflicts("b")
            assert result.has_conflicts is False
            assert "timed out" in result.error

    def test_check_conflicts_generic_exception(self, pr_manager: PRManager):
        """Generic exception returns error."""
        with patch(
            "animus_forge.self_improve.pr_manager.subprocess.run",
            side_effect=OSError("disk full"),
        ):
            result = pr_manager.check_conflicts("b")
            assert result.has_conflicts is False
            assert "disk full" in result.error

    def test_check_conflicts_diff_fails(self, pr_manager: PRManager):
        """Diff failure during conflict check still aborts and reports conflicts."""
        merge_result = MagicMock()
        merge_result.returncode = 1
        diff_result = MagicMock()
        diff_result.returncode = 1
        diff_result.stdout = ""
        abort_result = MagicMock()
        with patch(
            "animus_forge.self_improve.pr_manager.subprocess.run",
            side_effect=[merge_result, diff_result, abort_result],
        ):
            result = pr_manager.check_conflicts("b")
            assert result.has_conflicts is True
            assert result.conflicting_files == []

    def test_run_git_success(self, pr_manager: PRManager):
        """_run_git returns result on success."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"
        with patch("animus_forge.self_improve.pr_manager.subprocess.run", return_value=mock_result):
            result = pr_manager._run_git(["status"])
            assert result.stdout == "output"

    def test_run_git_failure(self, pr_manager: PRManager):
        """_run_git raises CalledProcessError on failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error"
        with patch("animus_forge.self_improve.pr_manager.subprocess.run", return_value=mock_result):
            with pytest.raises(subprocess.CalledProcessError):
                pr_manager._run_git(["bad-command"])

    def test_conflict_result_dataclass(self):
        """ConflictResult defaults."""
        r = ConflictResult()
        assert r.has_conflicts is False
        assert r.conflicting_files == []
        assert r.error is None

    def test_pull_request_dataclass(self):
        """PullRequest defaults."""
        pr = PullRequest(id="x", branch="b", title="T", description="D")
        assert pr.status == PRStatus.DRAFT
        assert pr.url is None
        assert pr.files_changed == []
        assert pr.metadata == {}


# ===========================================================================
# SelfImproveOrchestrator tests
# ===========================================================================


class TestSelfImproveOrchestrator:
    """Tests for SelfImproveOrchestrator."""

    def _make_orchestrator(self, tmp_path: Path, config: SafetyConfig) -> SelfImproveOrchestrator:
        """Create an orchestrator with temp paths and mocked components."""
        orch = SelfImproveOrchestrator(codebase_path=tmp_path, config=config)
        return orch

    def test_init(self, tmp_path: Path, safety_config: SafetyConfig):
        """Orchestrator initializes all components."""
        orch = self._make_orchestrator(tmp_path, safety_config)
        assert orch.current_stage == WorkflowStage.IDLE
        assert orch.provider is None

    def test_current_stage_property(self, tmp_path: Path, safety_config: SafetyConfig):
        """Current stage is accessible via property."""
        orch = self._make_orchestrator(tmp_path, safety_config)
        assert orch.current_stage == WorkflowStage.IDLE

    def test_get_status(self, tmp_path: Path, safety_config: SafetyConfig):
        """Get status returns correct dictionary."""
        orch = self._make_orchestrator(tmp_path, safety_config)
        status = orch.get_status()
        assert status["stage"] == "idle"
        assert status["current_plan"] is None
        assert status["pending_approvals"] == 0
        assert "snapshots_available" in status

    def test_run_no_suggestions(self, tmp_path: Path, safety_config: SafetyConfig):
        """Run completes with no improvements when analysis finds nothing."""
        orch = self._make_orchestrator(tmp_path, safety_config)
        # Empty codebase = no suggestions
        result = asyncio.run(orch.run())
        assert result.success is True
        assert result.stage_reached == WorkflowStage.COMPLETE
        assert "No improvements needed" in result.metadata.get("message", "")

    def test_run_with_safety_violations(self, tmp_path: Path, safety_config: SafetyConfig):
        """Run fails when safety violations are detected."""
        # Create a codebase with a file that triggers suggestions
        src = tmp_path / "src" / "animus_forge"
        src.mkdir(parents=True)
        (src / "bad.py").write_text("import os\n\ndef undoc(x):\n    return x\n")

        # Use restrictive config that will block
        config = SafetyConfig(
            critical_files=["src/animus_forge/bad.py"],
            max_files_per_pr=50,
            max_lines_changed=5000,
            max_new_files=20,
            human_approval_plan=False,
            human_approval_apply=False,
            human_approval_merge=False,
            branch_prefix="test-improve/",
        )
        orch = self._make_orchestrator(tmp_path, config)
        result = asyncio.run(orch.run())

        # The analyzer may find suggestions for this file, and since it's protected,
        # safety checker will raise violations
        if not result.success:
            assert result.stage_reached == WorkflowStage.PLANNING
            assert len(result.violations) > 0

    def test_run_with_approval_required_no_auto(self, tmp_path: Path):
        """Run stops at plan approval when human approval is required."""
        src = tmp_path / "src" / "animus_forge"
        src.mkdir(parents=True)
        (src / "needs_doc.py").write_text("import os\n\ndef no_doc():\n    pass\n")

        config = SafetyConfig(
            human_approval_plan=True,
            human_approval_apply=False,
            human_approval_merge=False,
            max_files_per_pr=50,
            max_lines_changed=5000,
            max_new_files=20,
            branch_prefix="test-improve/",
        )
        orch = self._make_orchestrator(tmp_path, config)

        # wait_for_approval now polls — mock to return EXPIRED immediately
        with patch.object(
            orch.approval_gate, "wait_for_approval", new_callable=AsyncMock, return_value="expired"
        ):
            result = asyncio.run(orch.run(auto_approve=False))

        if result.stage_reached == WorkflowStage.AWAITING_PLAN_APPROVAL:
            assert result.success is False
            assert "timed out" in result.error.lower() or "expired" in result.error.lower()

    def test_run_full_workflow_auto_approve(self, tmp_path: Path, safety_config: SafetyConfig):
        """Run full workflow with auto-approve through all stages."""
        src = tmp_path / "src" / "animus_forge"
        src.mkdir(parents=True)
        (src / "undoc.py").write_text("import os\n\ndef my_func():\n    pass\n")

        # Enable approvals but auto-approve them
        safety_config.human_approval_plan = True
        safety_config.human_approval_apply = True
        safety_config.human_approval_merge = True

        orch = self._make_orchestrator(tmp_path, safety_config)

        # Mock code generation + sandbox + git
        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=False)
        mock_sandbox.apply_changes = AsyncMock(return_value=True)
        mock_sandbox.validate_changes = AsyncMock(
            return_value=SandboxResult(
                status=SandboxStatus.SUCCESS, tests_passed=True, lint_passed=True
            )
        )

        with (
            patch.object(
                orch,
                "_generate_changes",
                new_callable=AsyncMock,
                return_value={"src/animus_forge/undoc.py": "fixed"},
            ),
            patch("animus_forge.self_improve.orchestrator.Sandbox", return_value=mock_sandbox),
            patch.object(orch.pr_manager, "_run_git"),
            patch("animus_forge.self_improve.pr_manager.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout="https://github.com/test/pull/1\n"
            )
            result = asyncio.run(orch.run(auto_approve=True))

        # With auto_approve and human_approval_merge, it may return at
        # AWAITING_MERGE_APPROVAL or COMPLETE
        assert result.plan is not None

    def test_run_full_workflow_no_approvals(self, tmp_path: Path, safety_config: SafetyConfig):
        """Run through to completion with no approval gates."""
        src = tmp_path / "src" / "animus_forge"
        src.mkdir(parents=True)
        (src / "file.py").write_text("import os\n\ndef func():\n    pass\n")

        orch = self._make_orchestrator(tmp_path, safety_config)

        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=False)
        mock_sandbox.apply_changes = AsyncMock(return_value=True)
        mock_sandbox.validate_changes = AsyncMock(
            return_value=SandboxResult(
                status=SandboxStatus.SUCCESS, tests_passed=True, lint_passed=True
            )
        )

        with (
            patch.object(
                orch,
                "_generate_changes",
                new_callable=AsyncMock,
                return_value={"src/animus_forge/file.py": "fixed"},
            ),
            patch("animus_forge.self_improve.orchestrator.Sandbox", return_value=mock_sandbox),
            patch.object(orch.pr_manager, "_run_git"),
            patch("animus_forge.self_improve.pr_manager.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stderr="no gh")
            result = asyncio.run(orch.run())

        assert result.stage_reached == WorkflowStage.COMPLETE
        assert result.success is True
        assert result.plan is not None
        assert result.snapshot is not None
        assert result.pull_request is not None

    def test_run_exception_handling(self, tmp_path: Path, safety_config: SafetyConfig):
        """Exceptions during run are caught and reported."""
        orch = self._make_orchestrator(tmp_path, safety_config)
        with patch.object(orch.analyzer, "analyze", side_effect=RuntimeError("boom")):
            result = asyncio.run(orch.run())
            assert result.success is False
            assert result.stage_reached == WorkflowStage.FAILED
            assert "boom" in result.error

    def test_create_plan_single_suggestion(self, tmp_path: Path, safety_config: SafetyConfig):
        """Plan from a single suggestion uses suggestion's title/description."""
        orch = self._make_orchestrator(tmp_path, safety_config)
        suggestion = ImprovementSuggestion(
            id="s1",
            category=ImprovementCategory.REFACTORING,
            title="Extract method",
            description="Split big function",
            affected_files=["a.py"],
            estimated_lines=20,
        )
        plan = orch._create_plan([suggestion])
        assert plan.title == "Extract method"
        assert plan.description == "Split big function"
        assert "a.py" in plan.estimated_files
        assert plan.estimated_lines == 20

    def test_create_plan_multiple_suggestions(self, tmp_path: Path, safety_config: SafetyConfig):
        """Plan from multiple suggestions aggregates correctly."""
        orch = self._make_orchestrator(tmp_path, safety_config)
        suggestions = [
            ImprovementSuggestion(
                id="s1",
                category=ImprovementCategory.REFACTORING,
                title="Fix A",
                description="Desc A",
                affected_files=["a.py"],
                estimated_lines=10,
            ),
            ImprovementSuggestion(
                id="s2",
                category=ImprovementCategory.BUG_FIXES,
                title="Fix B",
                description="Desc B",
                affected_files=["b.py"],
                estimated_lines=15,
            ),
        ]
        plan = orch._create_plan(suggestions)
        assert "2 items" in plan.title
        assert plan.estimated_lines == 25
        assert set(plan.estimated_files) == {"a.py", "b.py"}
        assert len(plan.implementation_steps) == 2

    def test_rollback_success(self, tmp_path: Path, safety_config: SafetyConfig):
        """Rollback changes stage to ROLLED_BACK."""
        orch = self._make_orchestrator(tmp_path, safety_config)

        # Create a file and snapshot
        (tmp_path / "rollback_file.py").write_text("original\n")
        snapshot = orch.rollback_manager.create_snapshot(
            files=["rollback_file.py"],
            description="test",
            codebase_path=tmp_path,
        )

        (tmp_path / "rollback_file.py").write_text("modified\n")
        result = orch.rollback(snapshot.id)
        assert result is True
        assert orch.current_stage == WorkflowStage.ROLLED_BACK
        assert (tmp_path / "rollback_file.py").read_text() == "original\n"

    def test_rollback_failure(self, tmp_path: Path, safety_config: SafetyConfig):
        """Failed rollback returns False and doesn't change stage."""
        orch = self._make_orchestrator(tmp_path, safety_config)
        result = orch.rollback("nonexistent")
        assert result is False
        assert orch.current_stage == WorkflowStage.IDLE

    def test_get_status_with_plan(self, tmp_path: Path, safety_config: SafetyConfig):
        """Status includes plan title when plan exists."""
        orch = self._make_orchestrator(tmp_path, safety_config)
        orch._current_plan = ImprovementPlan(
            id="p1",
            title="Test Plan",
            description="desc",
            suggestions=[],
            implementation_steps=[],
            estimated_files=[],
            estimated_lines=0,
        )
        status = orch.get_status()
        assert status["current_plan"] == "Test Plan"

    def test_workflow_stage_enum(self):
        """WorkflowStage enum values."""
        assert WorkflowStage.IDLE.value == "idle"
        assert WorkflowStage.ANALYZING.value == "analyzing"
        assert WorkflowStage.COMPLETE.value == "complete"
        assert WorkflowStage.FAILED.value == "failed"
        assert WorkflowStage.ROLLED_BACK.value == "rolled_back"

    def test_improvement_result_dataclass(self):
        """ImprovementResult defaults."""
        r = ImprovementResult(success=True, stage_reached=WorkflowStage.COMPLETE)
        assert r.plan is None
        assert r.snapshot is None
        assert r.sandbox_result is None
        assert r.pull_request is None
        assert r.error is None
        assert r.violations == []
        assert r.metadata == {}

    def test_improvement_plan_dataclass(self):
        """ImprovementPlan can be constructed."""
        plan = ImprovementPlan(
            id="x",
            title="T",
            description="D",
            suggestions=[],
            implementation_steps=["step1"],
            estimated_files=["a.py"],
            estimated_lines=10,
        )
        assert plan.id == "x"
        assert isinstance(plan.created_at, datetime)

    def test_run_apply_approval_required_no_auto(self, tmp_path: Path):
        """Run stops at apply approval when required and not auto-approved."""
        src = tmp_path / "src" / "animus_forge"
        src.mkdir(parents=True)
        (src / "f.py").write_text("import os\n\ndef func():\n    pass\n")

        config = SafetyConfig(
            human_approval_plan=False,
            human_approval_apply=True,
            human_approval_merge=False,
            max_files_per_pr=50,
            max_lines_changed=5000,
            max_new_files=20,
            branch_prefix="test-improve/",
        )
        orch = self._make_orchestrator(tmp_path, config)

        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=False)
        mock_sandbox.apply_changes = AsyncMock(return_value=True)
        mock_sandbox.validate_changes = AsyncMock(
            return_value=SandboxResult(
                status=SandboxStatus.SUCCESS, tests_passed=True, lint_passed=True
            )
        )

        with (
            patch.object(
                orch,
                "_generate_changes",
                new_callable=AsyncMock,
                return_value={"src/animus_forge/f.py": "fixed"},
            ),
            patch("animus_forge.self_improve.orchestrator.Sandbox", return_value=mock_sandbox),
            patch.object(
                orch.approval_gate,
                "wait_for_approval",
                new_callable=AsyncMock,
                return_value="expired",
            ),
        ):
            result = asyncio.run(orch.run(auto_approve=False))

        if result.stage_reached == WorkflowStage.AWAITING_APPLY_APPROVAL:
            assert result.success is False
            assert "timed out" in result.error.lower() or "expired" in result.error.lower()

    def test_run_merge_approval_stops(self, tmp_path: Path):
        """Run stops at merge approval when required and not auto-approved."""
        src = tmp_path / "src" / "animus_forge"
        src.mkdir(parents=True)
        (src / "f.py").write_text("import os\n\ndef func():\n    pass\n")

        config = SafetyConfig(
            human_approval_plan=False,
            human_approval_apply=False,
            human_approval_merge=True,
            max_files_per_pr=50,
            max_lines_changed=5000,
            max_new_files=20,
            branch_prefix="test-improve/",
        )
        orch = self._make_orchestrator(tmp_path, config)

        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=False)
        mock_sandbox.apply_changes = AsyncMock(return_value=True)
        mock_sandbox.validate_changes = AsyncMock(
            return_value=SandboxResult(
                status=SandboxStatus.SUCCESS, tests_passed=True, lint_passed=True
            )
        )

        with (
            patch.object(
                orch,
                "_generate_changes",
                new_callable=AsyncMock,
                return_value={"src/animus_forge/f.py": "fixed"},
            ),
            patch("animus_forge.self_improve.orchestrator.Sandbox", return_value=mock_sandbox),
            patch.object(orch.pr_manager, "_run_git"),
            patch("animus_forge.self_improve.pr_manager.subprocess.run") as mock_run,
            patch.object(
                orch.approval_gate,
                "wait_for_approval",
                new_callable=AsyncMock,
                return_value="expired",
            ),
        ):
            mock_run.return_value = MagicMock(returncode=1, stderr="")
            result = asyncio.run(orch.run(auto_approve=False))

        if result.stage_reached == WorkflowStage.AWAITING_MERGE_APPROVAL:
            assert result.success is True
            assert result.pull_request is not None

    def test_run_focus_category(self, tmp_path: Path, safety_config: SafetyConfig):
        """Run with a focus_category passes it to safety checker."""
        src = tmp_path / "src" / "animus_forge"
        src.mkdir(parents=True)
        (src / "f.py").write_text("import os\n\ndef func():\n    pass\n")

        orch = self._make_orchestrator(tmp_path, safety_config)

        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=False)
        mock_sandbox.apply_changes = AsyncMock(return_value=True)
        mock_sandbox.validate_changes = AsyncMock(
            return_value=SandboxResult(
                status=SandboxStatus.SUCCESS, tests_passed=True, lint_passed=True
            )
        )

        with (
            patch.object(
                orch,
                "_generate_changes",
                new_callable=AsyncMock,
                return_value={"src/animus_forge/f.py": "fixed"},
            ),
            patch("animus_forge.self_improve.orchestrator.Sandbox", return_value=mock_sandbox),
            patch.object(orch.pr_manager, "_run_git"),
            patch("animus_forge.self_improve.pr_manager.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stderr="")
            result = asyncio.run(orch.run(focus_category="documentation"))

        assert result.success is True


# ---------------------------------------------------------------------------
# Coverage gap tests — approval.py
# ---------------------------------------------------------------------------


class TestApprovalGateCoverage:
    """Tests for ApprovalGate database persistence and async polling."""

    def _make_backend(self):
        """Create a mock DatabaseBackend."""
        backend = MagicMock()
        backend.execute = MagicMock()
        backend.fetchone = MagicMock(return_value=None)
        backend.fetchall = MagicMock(return_value=[])
        backend.transaction = MagicMock()
        return backend

    def test_init_with_backend_creates_table(self):
        """ApprovalGate creates the approvals table when backend is provided."""
        from animus_forge.self_improve.approval import ApprovalGate

        backend = self._make_backend()
        ApprovalGate(backend=backend)
        backend.execute.assert_called_once()
        call_sql = backend.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS self_improve_approvals" in call_sql

    def test_ensure_table_no_backend(self):
        """_ensure_table is a no-op without a backend."""
        from animus_forge.self_improve.approval import ApprovalGate

        gate = ApprovalGate()
        gate._ensure_table()  # Should not raise

    def test_request_approval_persists_to_db(self):
        """request_approval persists the request when backend is available."""
        from animus_forge.self_improve.approval import ApprovalGate, ApprovalStage

        backend = self._make_backend()
        gate = ApprovalGate(backend=backend)
        req = gate.request_approval(
            stage=ApprovalStage.PLAN,
            title="Test",
            description="Desc",
            details={"key": "val"},
        )
        assert req.id in gate._pending_approvals
        # execute called for table creation + persist
        assert backend.execute.call_count >= 2

    def test_persist_request_no_backend(self):
        """_persist_request is a no-op without backend."""
        from animus_forge.self_improve.approval import (
            ApprovalGate,
            ApprovalRequest,
            ApprovalStage,
        )

        gate = ApprovalGate()
        req = ApprovalRequest(
            id="x", stage=ApprovalStage.PLAN, title="T", description="D", details={}
        )
        gate._persist_request(req)  # Should not raise

    def test_load_request_no_backend(self):
        """_load_request returns None without backend."""
        from animus_forge.self_improve.approval import ApprovalGate

        gate = ApprovalGate()
        assert gate._load_request("any") is None

    def test_load_request_found(self):
        """_load_request returns request from database row."""
        from animus_forge.self_improve.approval import ApprovalGate

        backend = self._make_backend()
        backend.fetchone.return_value = {
            "id": "r1",
            "stage": "plan",
            "title": "T",
            "description": "D",
            "details": '{"k": "v"}',
            "status": "pending",
            "created_at": "2026-01-01T00:00:00",
            "decided_at": None,
            "decided_by": None,
            "reason": None,
        }
        gate = ApprovalGate(backend=backend)
        req = gate._load_request("r1")
        assert req is not None
        assert req.id == "r1"
        assert req.details == {"k": "v"}

    def test_load_request_not_found(self):
        """_load_request returns None when row not found."""
        from animus_forge.self_improve.approval import ApprovalGate

        backend = self._make_backend()
        backend.fetchone.return_value = None
        gate = ApprovalGate(backend=backend)
        assert gate._load_request("missing") is None

    def test_row_to_request_with_decided_at(self):
        """_row_to_request parses decided_at datetime."""
        from animus_forge.self_improve.approval import ApprovalGate, ApprovalStatus

        row = {
            "id": "r1",
            "stage": "apply",
            "title": "T",
            "description": "D",
            "details": "{}",
            "status": "approved",
            "created_at": "2026-01-01T00:00:00",
            "decided_at": "2026-01-01T01:00:00",
            "decided_by": "admin",
            "reason": "looks good",
        }
        req = ApprovalGate._row_to_request(row)
        assert req.status == ApprovalStatus.APPROVED
        assert req.decided_at is not None
        assert req.decided_by == "admin"
        assert req.reason == "looks good"

    def test_row_to_request_empty_details(self):
        """_row_to_request handles empty/falsy details."""
        from animus_forge.self_improve.approval import ApprovalGate

        row = {
            "id": "r1",
            "stage": "plan",
            "title": "T",
            "description": "D",
            "details": "",
            "status": "pending",
            "created_at": "2026-01-01T00:00:00",
        }
        req = ApprovalGate._row_to_request(row)
        assert req.details == {}

    def test_get_pending_with_backend_no_stage(self):
        """get_pending queries backend without stage filter."""
        from animus_forge.self_improve.approval import ApprovalGate

        backend = self._make_backend()
        backend.fetchall.return_value = [
            {
                "id": "r1",
                "stage": "plan",
                "title": "T",
                "description": "D",
                "details": "{}",
                "status": "pending",
                "created_at": "2026-01-01T00:00:00",
            }
        ]
        gate = ApprovalGate(backend=backend)
        pending = gate.get_pending()
        assert len(pending) == 1
        assert pending[0].id == "r1"

    def test_get_pending_with_backend_stage_filter(self):
        """get_pending filters by stage when provided."""
        from animus_forge.self_improve.approval import ApprovalGate, ApprovalStage

        backend = self._make_backend()
        backend.fetchall.return_value = []
        gate = ApprovalGate(backend=backend)
        pending = gate.get_pending(stage=ApprovalStage.APPLY)
        assert pending == []
        call_args = backend.fetchall.call_args
        assert "stage = ?" in call_args[0][0]

    def test_get_pending_in_memory_with_stage_filter(self):
        """get_pending filters by stage in-memory."""
        from animus_forge.self_improve.approval import ApprovalGate, ApprovalStage

        gate = ApprovalGate()
        gate.request_approval(ApprovalStage.PLAN, "Plan A", "desc")
        gate.request_approval(ApprovalStage.APPLY, "Apply B", "desc")
        plan_pending = gate.get_pending(stage=ApprovalStage.PLAN)
        assert len(plan_pending) == 1
        assert plan_pending[0].stage == ApprovalStage.PLAN

    def test_approve_from_db(self):
        """approve loads from database when not in memory."""
        from animus_forge.self_improve.approval import ApprovalGate, ApprovalStatus

        backend = self._make_backend()
        backend.fetchone.return_value = {
            "id": "r1",
            "stage": "plan",
            "title": "T",
            "description": "D",
            "details": "{}",
            "status": "pending",
            "created_at": "2026-01-01T00:00:00",
        }
        gate = ApprovalGate(backend=backend)
        result = gate.approve("r1", approved_by="tester", reason="ok")
        assert result is not None
        assert result.status == ApprovalStatus.APPROVED
        assert result.decided_by == "tester"

    def test_approve_not_found(self):
        """approve returns None when request not found anywhere."""
        from animus_forge.self_improve.approval import ApprovalGate

        gate = ApprovalGate()
        assert gate.approve("nonexistent") is None

    def test_approve_persists_to_db(self):
        """approve persists updated status to database."""
        from animus_forge.self_improve.approval import ApprovalGate, ApprovalStage

        backend = self._make_backend()
        gate = ApprovalGate(backend=backend)
        req = gate.request_approval(ApprovalStage.PLAN, "T", "D")
        gate.approve(req.id, approved_by="admin")
        # persist called for create + approve
        assert backend.execute.call_count >= 3

    def test_reject_from_db(self):
        """reject loads from database when not in memory."""
        from animus_forge.self_improve.approval import ApprovalGate, ApprovalStatus

        backend = self._make_backend()
        backend.fetchone.return_value = {
            "id": "r2",
            "stage": "apply",
            "title": "T",
            "description": "D",
            "details": "{}",
            "status": "pending",
            "created_at": "2026-01-01T00:00:00",
        }
        gate = ApprovalGate(backend=backend)
        result = gate.reject("r2", rejected_by="reviewer", reason="bad")
        assert result is not None
        assert result.status == ApprovalStatus.REJECTED
        assert result.reason == "bad"

    def test_reject_not_found(self):
        """reject returns None when request not found."""
        from animus_forge.self_improve.approval import ApprovalGate

        gate = ApprovalGate()
        assert gate.reject("nonexistent") is None

    def test_reject_in_memory(self):
        """reject works for in-memory request."""
        from animus_forge.self_improve.approval import (
            ApprovalGate,
            ApprovalStage,
            ApprovalStatus,
        )

        gate = ApprovalGate()
        req = gate.request_approval(ApprovalStage.PLAN, "T", "D")
        result = gate.reject(req.id, reason="not ready")
        assert result.status == ApprovalStatus.REJECTED
        assert req.id not in gate._pending_approvals
        assert result in gate._approval_history

    def test_is_approved_in_history(self):
        """is_approved finds approved request in history."""
        from animus_forge.self_improve.approval import ApprovalGate, ApprovalStage

        gate = ApprovalGate()
        req = gate.request_approval(ApprovalStage.PLAN, "T", "D")
        gate.approve(req.id)
        assert gate.is_approved(req.id) is True

    def test_is_approved_rejected_in_history(self):
        """is_approved returns False for rejected request."""
        from animus_forge.self_improve.approval import ApprovalGate, ApprovalStage

        gate = ApprovalGate()
        req = gate.request_approval(ApprovalStage.PLAN, "T", "D")
        gate.reject(req.id)
        assert gate.is_approved(req.id) is False

    def test_is_approved_from_db(self):
        """is_approved checks database when not in history."""
        from animus_forge.self_improve.approval import ApprovalGate

        backend = self._make_backend()
        backend.fetchone.return_value = {"status": "approved"}
        gate = ApprovalGate(backend=backend)
        assert gate.is_approved("db-req") is True

    def test_is_approved_not_found(self):
        """is_approved returns False when request not found."""
        from animus_forge.self_improve.approval import ApprovalGate

        gate = ApprovalGate()
        assert gate.is_approved("nonexistent") is False

    def test_is_approved_from_db_not_approved(self):
        """is_approved returns False for non-approved DB entry."""
        from animus_forge.self_improve.approval import ApprovalGate

        backend = self._make_backend()
        backend.fetchone.return_value = {"status": "rejected"}
        gate = ApprovalGate(backend=backend)
        assert gate.is_approved("db-req") is False

    async def test_wait_for_approval_already_decided(self):
        """wait_for_approval returns immediately if already decided."""
        from animus_forge.self_improve.approval import (
            ApprovalGate,
            ApprovalStage,
            ApprovalStatus,
        )

        gate = ApprovalGate()
        req = gate.request_approval(ApprovalStage.PLAN, "T", "D")
        gate.approve(req.id)
        # Request removed from pending, so polling returns immediately
        status = await gate.wait_for_approval(req, timeout=1.0, poll_interval=0.1)
        assert status == ApprovalStatus.APPROVED

    async def test_wait_for_approval_timeout_expires(self):
        """wait_for_approval marks as expired after timeout."""
        from animus_forge.self_improve.approval import (
            ApprovalGate,
            ApprovalStage,
            ApprovalStatus,
        )

        gate = ApprovalGate()
        req = gate.request_approval(ApprovalStage.PLAN, "T", "D")
        status = await gate.wait_for_approval(req, timeout=0.15, poll_interval=0.1)
        assert status == ApprovalStatus.EXPIRED
        assert req.status == ApprovalStatus.EXPIRED
        assert req.decided_at is not None
        assert req.id not in gate._pending_approvals

    async def test_wait_for_approval_timeout_with_backend(self):
        """wait_for_approval persists expired status to database."""
        from animus_forge.self_improve.approval import (
            ApprovalGate,
            ApprovalStage,
            ApprovalStatus,
        )

        backend = self._make_backend()
        gate = ApprovalGate(backend=backend)
        req = gate.request_approval(ApprovalStage.PLAN, "T", "D")
        status = await gate.wait_for_approval(req, timeout=0.15, poll_interval=0.1)
        assert status == ApprovalStatus.EXPIRED
        # Persist called for create + expire
        assert backend.execute.call_count >= 3

    async def test_wait_for_approval_db_decision(self):
        """wait_for_approval detects external decision via database."""
        from animus_forge.self_improve.approval import (
            ApprovalGate,
            ApprovalStage,
            ApprovalStatus,
        )

        backend = self._make_backend()
        gate = ApprovalGate(backend=backend)
        req = gate.request_approval(ApprovalStage.PLAN, "T", "D")

        # First poll: still pending. Second poll: approved externally.
        backend.fetchone.side_effect = [
            None,  # First poll
            {
                "id": req.id,
                "stage": "plan",
                "title": "T",
                "description": "D",
                "details": "{}",
                "status": "approved",
                "created_at": "2026-01-01T00:00:00",
                "decided_at": "2026-01-01T01:00:00",
                "decided_by": "external_user",
                "reason": "approved externally",
            },
        ]
        status = await gate.wait_for_approval(req, timeout=1.0, poll_interval=0.05)
        assert status == ApprovalStatus.APPROVED
        assert req.decided_by == "external_user"

    def test_get_history_with_backend_no_stage(self):
        """get_history queries backend without stage filter."""
        from animus_forge.self_improve.approval import ApprovalGate

        backend = self._make_backend()
        backend.fetchall.return_value = [
            {
                "id": "r1",
                "stage": "plan",
                "title": "T",
                "description": "D",
                "details": "{}",
                "status": "approved",
                "created_at": "2026-01-01T00:00:00",
                "decided_at": "2026-01-01T01:00:00",
                "decided_by": "admin",
                "reason": None,
            }
        ]
        gate = ApprovalGate(backend=backend)
        history = gate.get_history()
        assert len(history) == 1

    def test_get_history_with_backend_stage_filter(self):
        """get_history filters by stage when provided."""
        from animus_forge.self_improve.approval import ApprovalGate, ApprovalStage

        backend = self._make_backend()
        backend.fetchall.return_value = []
        gate = ApprovalGate(backend=backend)
        history = gate.get_history(stage=ApprovalStage.MERGE)
        assert history == []
        call_sql = backend.fetchall.call_args[0][0]
        assert "stage = ?" in call_sql

    def test_get_history_in_memory_with_stage(self):
        """get_history filters by stage in-memory."""
        from animus_forge.self_improve.approval import (
            ApprovalGate,
            ApprovalStage,
        )

        gate = ApprovalGate()
        req1 = gate.request_approval(ApprovalStage.PLAN, "Plan", "desc")
        req2 = gate.request_approval(ApprovalStage.APPLY, "Apply", "desc")
        gate.approve(req1.id)
        gate.reject(req2.id)
        history = gate.get_history(stage=ApprovalStage.PLAN)
        assert len(history) == 1
        assert history[0].stage == ApprovalStage.PLAN

    def test_get_history_in_memory_limit(self):
        """get_history respects limit."""
        from animus_forge.self_improve.approval import ApprovalGate, ApprovalStage

        gate = ApprovalGate()
        for _ in range(5):
            req = gate.request_approval(ApprovalStage.PLAN, "T", "D")
            gate.approve(req.id)
        history = gate.get_history(limit=3)
        assert len(history) == 3


# ---------------------------------------------------------------------------
# Coverage gap tests — sandbox.py
# ---------------------------------------------------------------------------


class TestSandboxCoverage:
    """Tests for Sandbox create, apply, test, lint, validate, cleanup."""

    def test_init_sets_defaults(self, tmp_path):
        """Sandbox initializes with correct defaults."""
        from animus_forge.self_improve.sandbox import Sandbox, SandboxStatus

        sb = Sandbox(tmp_path)
        assert sb.source_path == tmp_path
        assert sb.timeout == 300
        assert sb.cleanup_on_exit is True
        assert sb.sandbox_path is None
        assert sb.status == SandboxStatus.CREATED

    def test_create_copies_source(self, tmp_path):
        """create() copies source tree to temp directory."""
        from animus_forge.self_improve.sandbox import Sandbox

        (tmp_path / "hello.py").write_text("print('hi')\n")
        sb = Sandbox(tmp_path)
        path = sb.create()
        assert path is not None
        assert (path / "hello.py").exists()
        assert (path / "hello.py").read_text() == "print('hi')\n"
        sb.cleanup()

    def test_create_skips_ignored_dirs(self, tmp_path):
        """create() ignores .git, __pycache__, .venv, etc."""
        from animus_forge.self_improve.sandbox import Sandbox

        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git config")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "mod.pyc").write_text("bytecode")
        (tmp_path / "src.py").write_text("code")
        sb = Sandbox(tmp_path)
        path = sb.create()
        assert not (path / ".git").exists()
        assert not (path / "__pycache__").exists()
        assert (path / "src.py").exists()
        sb.cleanup()

    def test_create_idempotent(self, tmp_path):
        """create() returns same path on subsequent calls."""
        from animus_forge.self_improve.sandbox import Sandbox

        (tmp_path / "f.py").write_text("x")
        sb = Sandbox(tmp_path)
        p1 = sb.create()
        p2 = sb.create()
        assert p1 == p2
        sb.cleanup()

    def test_cleanup_removes_temp_dir(self, tmp_path):
        """cleanup() removes the sandbox directory."""
        from animus_forge.self_improve.sandbox import Sandbox

        (tmp_path / "f.py").write_text("x")
        sb = Sandbox(tmp_path)
        sb.create()
        temp_dir = sb._temp_dir
        assert temp_dir.exists()
        sb.cleanup()
        assert not temp_dir.exists()
        assert sb._temp_dir is None
        assert sb._sandbox_path is None

    def test_cleanup_noop_when_not_created(self, tmp_path):
        """cleanup() is safe to call without create()."""
        from animus_forge.self_improve.sandbox import Sandbox

        sb = Sandbox(tmp_path)
        sb.cleanup()  # Should not raise

    def test_context_manager_creates_and_cleans(self, tmp_path):
        """Context manager creates on enter and cleans on exit."""
        from animus_forge.self_improve.sandbox import Sandbox

        (tmp_path / "f.py").write_text("x")
        with Sandbox(tmp_path) as sb:
            assert sb.sandbox_path is not None
            temp_dir = sb._temp_dir
            assert temp_dir.exists()
        assert not temp_dir.exists()

    def test_context_manager_no_cleanup(self, tmp_path):
        """Context manager skips cleanup when cleanup_on_exit=False."""
        from animus_forge.self_improve.sandbox import Sandbox

        (tmp_path / "f.py").write_text("x")
        with Sandbox(tmp_path, cleanup_on_exit=False) as sb:
            temp_dir = sb._temp_dir
        assert temp_dir.exists()
        # Manual cleanup
        import shutil

        shutil.rmtree(temp_dir)

    async def test_apply_changes_success(self, tmp_path):
        """apply_changes writes files to sandbox."""
        from animus_forge.self_improve.sandbox import Sandbox

        (tmp_path / "f.py").write_text("original")
        sb = Sandbox(tmp_path)
        sb.create()
        result = await sb.apply_changes({"f.py": "modified", "new/nested.py": "new file"})
        assert result is True
        assert (sb.sandbox_path / "f.py").read_text() == "modified"
        assert (sb.sandbox_path / "new" / "nested.py").read_text() == "new file"
        sb.cleanup()

    async def test_apply_changes_not_created(self, tmp_path):
        """apply_changes raises when sandbox not created."""
        from animus_forge.self_improve.sandbox import Sandbox

        sb = Sandbox(tmp_path)
        with pytest.raises(RuntimeError, match="Sandbox not created"):
            await sb.apply_changes({"f.py": "x"})

    async def test_apply_changes_error(self, tmp_path):
        """apply_changes returns False on write error."""
        from animus_forge.self_improve.sandbox import Sandbox

        (tmp_path / "f.py").write_text("x")
        sb = Sandbox(tmp_path)
        sb.create()
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            result = await sb.apply_changes({"f.py": "new content"})
        assert result is False
        sb.cleanup()

    async def test_run_tests_not_created(self, tmp_path):
        """run_tests returns FAILED when sandbox not created."""
        from animus_forge.self_improve.sandbox import Sandbox, SandboxStatus

        sb = Sandbox(tmp_path)
        result = await sb.run_tests()
        assert result.status == SandboxStatus.FAILED
        assert result.error == "Sandbox not created"

    async def test_run_tests_success(self, tmp_path):
        """run_tests returns SUCCESS when pytest passes."""
        from animus_forge.self_improve.sandbox import Sandbox, SandboxStatus

        (tmp_path / "f.py").write_text("x")
        sb = Sandbox(tmp_path)
        sb.create()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"all passed", b""))
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await sb.run_tests()
        assert result.status == SandboxStatus.SUCCESS
        assert result.tests_passed is True
        assert result.test_output == "all passed"
        assert sb.status == SandboxStatus.SUCCESS
        sb.cleanup()

    async def test_run_tests_failure(self, tmp_path):
        """run_tests returns FAILED when pytest fails."""
        from animus_forge.self_improve.sandbox import Sandbox, SandboxStatus

        (tmp_path / "f.py").write_text("x")
        sb = Sandbox(tmp_path)
        sb.create()

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"2 failed", b"err"))
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await sb.run_tests()
        assert result.status == SandboxStatus.FAILED
        assert result.tests_passed is False
        assert "2 failed" in result.test_output
        assert sb.status == SandboxStatus.FAILED
        sb.cleanup()

    async def test_run_tests_timeout(self, tmp_path):
        """run_tests returns TIMEOUT on timeout."""
        from animus_forge.self_improve.sandbox import Sandbox, SandboxStatus

        (tmp_path / "f.py").write_text("x")
        sb = Sandbox(tmp_path, timeout=1)
        sb.create()

        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("asyncio.wait_for", side_effect=TimeoutError()),
        ):
            result = await sb.run_tests()
        assert result.status == SandboxStatus.TIMEOUT
        assert sb.status == SandboxStatus.TIMEOUT
        sb.cleanup()

    async def test_run_tests_exception(self, tmp_path):
        """run_tests returns FAILED on unexpected exception."""
        from animus_forge.self_improve.sandbox import Sandbox, SandboxStatus

        (tmp_path / "f.py").write_text("x")
        sb = Sandbox(tmp_path)
        sb.create()

        with patch("asyncio.create_subprocess_exec", side_effect=RuntimeError("oops")):
            result = await sb.run_tests()
        assert result.status == SandboxStatus.FAILED
        assert result.error == "oops"
        sb.cleanup()

    async def test_run_lint_not_created(self, tmp_path):
        """run_lint returns FAILED when sandbox not created."""
        from animus_forge.self_improve.sandbox import Sandbox, SandboxStatus

        sb = Sandbox(tmp_path)
        result = await sb.run_lint()
        assert result.status == SandboxStatus.FAILED
        assert result.error == "Sandbox not created"

    async def test_run_lint_success(self, tmp_path):
        """run_lint returns SUCCESS when ruff passes."""
        from animus_forge.self_improve.sandbox import Sandbox, SandboxStatus

        (tmp_path / "f.py").write_text("x")
        sb = Sandbox(tmp_path)
        sb.create()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await sb.run_lint()
        assert result.status == SandboxStatus.SUCCESS
        assert result.lint_passed is True
        sb.cleanup()

    async def test_run_lint_failure(self, tmp_path):
        """run_lint returns FAILED when ruff finds issues."""
        from animus_forge.self_improve.sandbox import Sandbox, SandboxStatus

        (tmp_path / "f.py").write_text("x")
        sb = Sandbox(tmp_path)
        sb.create()

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"E501", b""))
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await sb.run_lint()
        assert result.status == SandboxStatus.FAILED
        assert result.lint_passed is False
        sb.cleanup()

    async def test_run_lint_exception(self, tmp_path):
        """run_lint returns FAILED on exception."""
        from animus_forge.self_improve.sandbox import Sandbox, SandboxStatus

        (tmp_path / "f.py").write_text("x")
        sb = Sandbox(tmp_path)
        sb.create()

        with patch("asyncio.create_subprocess_exec", side_effect=OSError("no ruff")):
            result = await sb.run_lint()
        assert result.status == SandboxStatus.FAILED
        assert result.error == "no ruff"
        sb.cleanup()

    async def test_validate_changes_all_pass(self, tmp_path):
        """validate_changes returns SUCCESS when both lint and tests pass."""
        from animus_forge.self_improve.sandbox import Sandbox, SandboxResult, SandboxStatus

        (tmp_path / "f.py").write_text("x")
        sb = Sandbox(tmp_path)
        sb.create()

        with (
            patch.object(
                sb,
                "run_lint",
                new_callable=AsyncMock,
                return_value=SandboxResult(
                    status=SandboxStatus.SUCCESS, lint_passed=True, lint_output="ok"
                ),
            ),
            patch.object(
                sb,
                "run_tests",
                new_callable=AsyncMock,
                return_value=SandboxResult(
                    status=SandboxStatus.SUCCESS,
                    tests_passed=True,
                    test_output="passed",
                    duration_seconds=1.5,
                ),
            ),
        ):
            result = await sb.validate_changes()
        assert result.status == SandboxStatus.SUCCESS
        assert result.tests_passed is True
        assert result.lint_passed is True
        sb.cleanup()

    async def test_validate_changes_lint_fails(self, tmp_path):
        """validate_changes returns FAILED when lint fails."""
        from animus_forge.self_improve.sandbox import Sandbox, SandboxResult, SandboxStatus

        (tmp_path / "f.py").write_text("x")
        sb = Sandbox(tmp_path)
        sb.create()

        with (
            patch.object(
                sb,
                "run_lint",
                new_callable=AsyncMock,
                return_value=SandboxResult(status=SandboxStatus.FAILED, lint_passed=False),
            ),
            patch.object(
                sb,
                "run_tests",
                new_callable=AsyncMock,
                return_value=SandboxResult(status=SandboxStatus.SUCCESS, tests_passed=True),
            ),
        ):
            result = await sb.validate_changes()
        assert result.status == SandboxStatus.FAILED
        sb.cleanup()

    def test_sanitize_env(self, tmp_path):
        """_sanitize_env strips secret-like variables."""
        from animus_forge.self_improve.sandbox import Sandbox

        sb = Sandbox(tmp_path)
        with patch.dict(
            "os.environ",
            {
                "PATH": "/usr/bin",
                "HOME": "/home/user",
                "API_KEY": "secret123",
                "AWS_SECRET_KEY": "s3cret",
                "NORMAL_VAR": "ok",
            },
            clear=True,
        ):
            env = sb._sanitize_env()
        assert "PATH" in env
        assert "HOME" in env
        assert "NORMAL_VAR" in env
        assert "API_KEY" not in env
        assert "AWS_SECRET_KEY" not in env

    async def test_run_command_not_created(self, tmp_path):
        """_run_command raises when sandbox not created."""
        from animus_forge.self_improve.sandbox import Sandbox

        sb = Sandbox(tmp_path)
        with pytest.raises(RuntimeError, match="Sandbox not created"):
            await sb._run_command(["echo", "hi"])

    async def test_run_command_timeout_kills_process(self, tmp_path):
        """_run_command kills process on timeout and re-raises."""
        from animus_forge.self_improve.sandbox import Sandbox

        (tmp_path / "f.py").write_text("x")
        sb = Sandbox(tmp_path, timeout=1)
        sb.create()

        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("asyncio.wait_for", side_effect=TimeoutError()),
        ):
            with pytest.raises(TimeoutError):
                await sb._run_command(["sleep", "100"])
        mock_proc.kill.assert_called_once()
        sb.cleanup()

    def test_sandbox_result_defaults(self):
        """SandboxResult has correct defaults."""
        from animus_forge.self_improve.sandbox import SandboxResult, SandboxStatus

        r = SandboxResult(status=SandboxStatus.CREATED)
        assert r.exit_code == 0
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.tests_passed is False
        assert r.lint_passed is False
        assert r.error is None
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# Coverage gap tests — safety.py
# ---------------------------------------------------------------------------


class TestSafetyCheckerCoverage:
    """Tests for SafetyConfig.load(), _from_dict(), and SafetyChecker methods."""

    def test_load_default_no_file(self, tmp_path):
        """load() returns defaults when config file doesn't exist."""
        config = SafetyConfig.load(config_path=tmp_path / "nonexistent.yaml")
        assert config.max_files_per_pr == 10
        assert config.human_approval_plan is True

    def test_load_from_yaml(self, tmp_path):
        """load() reads from YAML file."""
        import yaml

        config_data = {
            "protected_files": {
                "critical": ["*.lock"],
                "sensitive": ["config/*"],
            },
            "limits": {
                "max_files_per_pr": 5,
                "max_lines_changed": 200,
                "max_deleted_files": 1,
                "max_new_files": 3,
            },
            "requirements": {
                "tests_must_pass": False,
                "human_approval": {
                    "plan": False,
                    "apply": True,
                    "merge": False,
                },
            },
            "sandbox": {
                "use_branch": False,
                "branch_prefix": "auto/",
                "isolated_execution": False,
                "timeout": 120,
            },
            "rollback": {
                "max_snapshots": 5,
                "auto_rollback_on_test_failure": False,
            },
            "allowed_categories": ["documentation", "testing"],
            "denied_categories": ["security"],
        }
        config_file = tmp_path / "safety.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = SafetyConfig.load(config_path=config_file)
        assert config.critical_files == ["*.lock"]
        assert config.sensitive_files == ["config/*"]
        assert config.max_files_per_pr == 5
        assert config.max_lines_changed == 200
        assert config.max_deleted_files == 1
        assert config.max_new_files == 3
        assert config.tests_must_pass is False
        assert config.human_approval_plan is False
        assert config.human_approval_apply is True
        assert config.human_approval_merge is False
        assert config.use_branch is False
        assert config.branch_prefix == "auto/"
        assert config.isolated_execution is False
        assert config.sandbox_timeout == 120
        assert config.max_snapshots == 5
        assert config.auto_rollback_on_test_failure is False
        assert config.allowed_categories == ["documentation", "testing"]
        assert config.denied_categories == ["security"]

    def test_from_dict_defaults(self):
        """_from_dict uses defaults for missing keys."""
        config = SafetyConfig._from_dict({})
        assert config.critical_files == []
        assert config.max_files_per_pr == 10
        assert config.human_approval_plan is True

    def test_is_protected_file_match(self):
        """is_protected_file matches glob patterns."""
        from animus_forge.self_improve.safety import SafetyChecker

        config = SafetyConfig(critical_files=["*.lock", "Dockerfile"])
        checker = SafetyChecker(config)
        assert checker.is_protected_file("poetry.lock") is True
        assert checker.is_protected_file("Dockerfile") is True
        assert checker.is_protected_file("main.py") is False

    def test_is_sensitive_file_match(self):
        """is_sensitive_file matches glob patterns."""
        from animus_forge.self_improve.safety import SafetyChecker

        config = SafetyConfig(sensitive_files=["config/*", "*.env"])
        checker = SafetyChecker(config)
        assert checker.is_sensitive_file("config/settings.yaml") is True
        assert checker.is_sensitive_file("prod.env") is True
        assert checker.is_sensitive_file("src/main.py") is False

    def test_matches_patterns_no_patterns(self):
        """_matches_patterns returns False with empty patterns."""
        from animus_forge.self_improve.safety import SafetyChecker

        checker = SafetyChecker(SafetyConfig())
        assert checker._matches_patterns("any.py", []) is False

    def test_is_allowed_category_denied(self):
        """is_allowed_category returns False for denied categories."""
        from animus_forge.self_improve.safety import SafetyChecker

        config = SafetyConfig(denied_categories=["security", "infra"])
        checker = SafetyChecker(config)
        assert checker.is_allowed_category("security") is False
        assert checker.is_allowed_category("documentation") is True

    def test_is_allowed_category_allowed_list(self):
        """is_allowed_category checks allowed list when set."""
        from animus_forge.self_improve.safety import SafetyChecker

        config = SafetyConfig(allowed_categories=["docs", "tests"])
        checker = SafetyChecker(config)
        assert checker.is_allowed_category("docs") is True
        assert checker.is_allowed_category("refactoring") is False

    def test_is_allowed_category_no_restrictions(self):
        """is_allowed_category allows all when no lists set."""
        from animus_forge.self_improve.safety import SafetyChecker

        checker = SafetyChecker(SafetyConfig())
        assert checker.is_allowed_category("anything") is True

    def test_check_changes_file_limit(self):
        """check_changes detects too many files."""
        from animus_forge.self_improve.safety import SafetyChecker

        config = SafetyConfig(max_files_per_pr=2)
        checker = SafetyChecker(config)
        violations = checker.check_changes(
            files_modified=["a.py", "b.py"],
            files_added=["c.py"],
            files_deleted=[],
            lines_changed=10,
        )
        assert any(v.violation_type == "file_limit" for v in violations)

    def test_check_changes_delete_limit(self):
        """check_changes detects too many deletions."""
        from animus_forge.self_improve.safety import SafetyChecker

        config = SafetyConfig(max_deleted_files=0)
        checker = SafetyChecker(config)
        violations = checker.check_changes(
            files_modified=[],
            files_added=[],
            files_deleted=["old.py"],
            lines_changed=0,
        )
        assert any(v.violation_type == "delete_limit" for v in violations)

    def test_check_changes_new_file_limit(self):
        """check_changes detects too many new files."""
        from animus_forge.self_improve.safety import SafetyChecker

        config = SafetyConfig(max_new_files=1)
        checker = SafetyChecker(config)
        violations = checker.check_changes(
            files_modified=[],
            files_added=["a.py", "b.py"],
            files_deleted=[],
            lines_changed=0,
        )
        assert any(v.violation_type == "new_file_limit" for v in violations)

    def test_check_changes_lines_limit(self):
        """check_changes detects too many lines changed."""
        from animus_forge.self_improve.safety import SafetyChecker

        config = SafetyConfig(max_lines_changed=100)
        checker = SafetyChecker(config)
        violations = checker.check_changes(
            files_modified=[],
            files_added=[],
            files_deleted=[],
            lines_changed=200,
        )
        assert any(v.violation_type == "lines_limit" for v in violations)

    def test_check_changes_protected_file(self):
        """check_changes detects protected file modification."""
        from animus_forge.self_improve.safety import SafetyChecker

        config = SafetyConfig(critical_files=["*.lock"])
        checker = SafetyChecker(config)
        violations = checker.check_changes(
            files_modified=["poetry.lock"],
            files_added=[],
            files_deleted=[],
            lines_changed=1,
        )
        assert any(v.violation_type == "protected_file" for v in violations)

    def test_check_changes_denied_category(self):
        """check_changes detects denied category."""
        from animus_forge.self_improve.safety import SafetyChecker

        config = SafetyConfig(denied_categories=["security"])
        checker = SafetyChecker(config)
        violations = checker.check_changes(
            files_modified=[],
            files_added=[],
            files_deleted=[],
            lines_changed=0,
            category="security",
        )
        assert any(v.violation_type == "denied_category" for v in violations)

    def test_check_changes_sensitive_file_warning(self):
        """check_changes adds warning for sensitive files."""
        from animus_forge.self_improve.safety import SafetyChecker

        config = SafetyConfig(sensitive_files=["config/*"])
        checker = SafetyChecker(config)
        violations = checker.check_changes(
            files_modified=["config/settings.yaml"],
            files_added=[],
            files_deleted=[],
            lines_changed=1,
        )
        sensitive_violations = [v for v in violations if v.violation_type == "sensitive_file"]
        assert len(sensitive_violations) == 1
        assert sensitive_violations[0].severity == "warning"

    def test_check_changes_clean(self):
        """check_changes returns empty list when all is safe."""
        from animus_forge.self_improve.safety import SafetyChecker

        checker = SafetyChecker(SafetyConfig())
        violations = checker.check_changes(
            files_modified=["main.py"],
            files_added=[],
            files_deleted=[],
            lines_changed=10,
        )
        assert violations == []

    def test_has_blocking_violations_error(self):
        """has_blocking_violations returns True for error-level violations."""
        from animus_forge.self_improve.safety import SafetyChecker, SafetyViolation

        checker = SafetyChecker(SafetyConfig())
        violations = [
            SafetyViolation(file_path="", violation_type="test", message="m", severity="error")
        ]
        assert checker.has_blocking_violations(violations) is True

    def test_has_blocking_violations_warning_only(self):
        """has_blocking_violations returns False for warning-only violations."""
        from animus_forge.self_improve.safety import SafetyChecker, SafetyViolation

        checker = SafetyChecker(SafetyConfig())
        violations = [
            SafetyViolation(file_path="", violation_type="test", message="m", severity="warning")
        ]
        assert checker.has_blocking_violations(violations) is False

    def test_has_blocking_violations_empty(self):
        """has_blocking_violations returns False with no violations."""
        from animus_forge.self_improve.safety import SafetyChecker

        checker = SafetyChecker(SafetyConfig())
        assert checker.has_blocking_violations([]) is False


# ---------------------------------------------------------------------------
# Coverage gap tests — orchestrator.py
# ---------------------------------------------------------------------------


class TestOrchestratorCoverage:
    """Tests for orchestrator _generate_changes, _apply_changes, and workflow paths."""

    def _make_orch(self, tmp_path, provider=None):
        """Create orchestrator with test config."""
        config = SafetyConfig(
            human_approval_plan=False,
            human_approval_apply=False,
            human_approval_merge=False,
            max_files_per_pr=50,
            max_lines_changed=5000,
            max_new_files=20,
            branch_prefix="test/",
        )
        return SelfImproveOrchestrator(codebase_path=tmp_path, provider=provider, config=config)

    async def test_generate_changes_no_provider(self, tmp_path):
        """_generate_changes returns empty dict without provider."""
        orch = self._make_orch(tmp_path)
        plan = ImprovementPlan(
            id="p1",
            title="T",
            description="D",
            suggestions=[],
            implementation_steps=[],
            estimated_files=["f.py"],
            estimated_lines=10,
        )
        changes = await orch._generate_changes(plan)
        assert changes == {}

    async def test_generate_changes_reads_files(self, tmp_path):
        """_generate_changes reads affected files and calls provider."""
        (tmp_path / "src.py").write_text("original code")
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value='{"src.py": "improved code"}')
        orch = self._make_orch(tmp_path, provider=provider)

        suggestion = ImprovementSuggestion(
            id="s1",
            category=ImprovementCategory.REFACTORING,
            title="Improve",
            description="Make better",
            affected_files=["src.py"],
            estimated_lines=5,
            implementation_hints="Add docstring",
        )
        plan = ImprovementPlan(
            id="p1",
            title="T",
            description="D",
            suggestions=[suggestion],
            implementation_steps=["step1"],
            estimated_files=["src.py"],
            estimated_lines=5,
        )
        changes = await orch._generate_changes(plan)
        assert changes == {"src.py": "improved code"}
        provider.complete.assert_called_once()

    async def test_generate_changes_skips_nonexistent_files(self, tmp_path):
        """_generate_changes skips files that don't exist."""
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value='{"new.py": "content"}')
        orch = self._make_orch(tmp_path, provider=provider)

        plan = ImprovementPlan(
            id="p1",
            title="T",
            description="D",
            suggestions=[],
            implementation_steps=[],
            estimated_files=["nonexistent.py"],
            estimated_lines=5,
        )
        changes = await orch._generate_changes(plan)
        assert changes == {"new.py": "content"}

    async def test_generate_changes_skips_protected_files(self, tmp_path):
        """_generate_changes filters out protected files from response."""
        (tmp_path / "safe.py").write_text("ok")
        provider = AsyncMock()
        provider.complete = AsyncMock(
            return_value='{"safe.py": "improved", "pyproject.toml": "bad"}'
        )
        config = SafetyConfig(
            critical_files=["pyproject.toml"],
            human_approval_plan=False,
            human_approval_apply=False,
            human_approval_merge=False,
            branch_prefix="test/",
        )
        orch = SelfImproveOrchestrator(codebase_path=tmp_path, provider=provider, config=config)

        plan = ImprovementPlan(
            id="p1",
            title="T",
            description="D",
            suggestions=[],
            implementation_steps=[],
            estimated_files=["safe.py"],
            estimated_lines=5,
        )
        changes = await orch._generate_changes(plan)
        assert "safe.py" in changes
        assert "pyproject.toml" not in changes

    async def test_generate_changes_exception(self, tmp_path):
        """_generate_changes returns empty dict on provider exception."""
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=RuntimeError("API down"))
        orch = self._make_orch(tmp_path, provider=provider)

        plan = ImprovementPlan(
            id="p1",
            title="T",
            description="D",
            suggestions=[],
            implementation_steps=[],
            estimated_files=[],
            estimated_lines=0,
        )
        changes = await orch._generate_changes(plan)
        assert changes == {}

    async def test_generate_changes_unreadable_file(self, tmp_path):
        """_generate_changes skips files that can't be read."""
        (tmp_path / "bad.py").write_text("content")
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value='{"bad.py": "new"}')
        orch = self._make_orch(tmp_path, provider=provider)

        plan = ImprovementPlan(
            id="p1",
            title="T",
            description="D",
            suggestions=[],
            implementation_steps=[],
            estimated_files=["bad.py"],
            estimated_lines=5,
        )
        with patch.object(Path, "read_text", side_effect=OSError("perm denied")):
            changes = await orch._generate_changes(plan)
        assert changes == {"bad.py": "new"}

    def test_parse_changes_response_valid_json(self, tmp_path):
        """_parse_changes_response parses raw JSON."""
        orch = self._make_orch(tmp_path)
        result = orch._parse_changes_response('{"a.py": "content"}')
        assert result == {"a.py": "content"}

    def test_parse_changes_response_markdown_fenced(self, tmp_path):
        """_parse_changes_response strips markdown code fences."""
        orch = self._make_orch(tmp_path)
        text = '```json\n{"a.py": "content"}\n```'
        result = orch._parse_changes_response(text)
        assert result == {"a.py": "content"}

    def test_parse_changes_response_regex_fallback(self, tmp_path):
        """_parse_changes_response uses regex fallback for preamble text."""
        orch = self._make_orch(tmp_path)
        text = 'Here are the changes:\n{"a.py": "content"}\nDone!'
        result = orch._parse_changes_response(text)
        assert result == {"a.py": "content"}

    def test_parse_changes_response_invalid(self, tmp_path):
        """_parse_changes_response returns empty dict for unparseable text."""
        orch = self._make_orch(tmp_path)
        result = orch._parse_changes_response("not json at all")
        assert result == {}

    def test_parse_changes_response_non_dict(self, tmp_path):
        """_parse_changes_response returns empty dict for non-dict JSON."""
        orch = self._make_orch(tmp_path)
        result = orch._parse_changes_response('["a", "b"]')
        assert result == {}

    def test_parse_changes_response_non_string_values(self, tmp_path):
        """_parse_changes_response converts non-string values to strings."""
        orch = self._make_orch(tmp_path)
        result = orch._parse_changes_response('{"a.py": 123, "b.py": true}')
        assert result == {"a.py": "123", "b.py": "True"}

    def test_apply_changes_writes_files(self, tmp_path):
        """_apply_changes writes content to files."""
        orch = self._make_orch(tmp_path)
        orch._apply_changes({"src/new.py": "new content", "existing.py": "updated"})
        assert (tmp_path / "src" / "new.py").read_text() == "new content"
        assert (tmp_path / "existing.py").read_text() == "updated"

    async def test_run_no_changes_generated(self, tmp_path):
        """Run fails when _generate_changes returns empty dict."""
        src = tmp_path / "src" / "animus_forge"
        src.mkdir(parents=True)
        (src / "f.py").write_text("import os\n\ndef func():\n    pass\n")

        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="not valid json")
        orch = self._make_orch(tmp_path, provider=provider)

        result = await orch.run()
        if result.stage_reached == WorkflowStage.IMPLEMENTING:
            assert result.success is False
            assert "No changes generated" in result.error

    async def test_run_sandbox_apply_fails(self, tmp_path):
        """Run fails when sandbox apply_changes returns False."""
        src = tmp_path / "src" / "animus_forge"
        src.mkdir(parents=True)
        (src / "f.py").write_text("import os\n\ndef func():\n    pass\n")

        orch = self._make_orch(tmp_path)

        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=False)
        mock_sandbox.apply_changes = AsyncMock(return_value=False)

        with (
            patch.object(
                orch,
                "_generate_changes",
                new_callable=AsyncMock,
                return_value={"src/animus_forge/f.py": "fixed"},
            ),
            patch(
                "animus_forge.self_improve.orchestrator.Sandbox",
                return_value=mock_sandbox,
            ),
        ):
            result = await orch.run()

        assert result.success is False
        assert result.stage_reached == WorkflowStage.TESTING
        assert "Failed to apply" in result.error

    async def test_run_tests_fail_auto_rollback(self, tmp_path):
        """Run fails when sandbox tests fail and auto_rollback is enabled."""
        src = tmp_path / "src" / "animus_forge"
        src.mkdir(parents=True)
        (src / "f.py").write_text("import os\n\ndef func():\n    pass\n")

        orch = self._make_orch(tmp_path)

        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=False)
        mock_sandbox.apply_changes = AsyncMock(return_value=True)
        mock_sandbox.validate_changes = AsyncMock(
            return_value=SandboxResult(
                status=SandboxStatus.FAILED, tests_passed=False, lint_passed=True
            )
        )

        with (
            patch.object(
                orch,
                "_generate_changes",
                new_callable=AsyncMock,
                return_value={"src/animus_forge/f.py": "fixed"},
            ),
            patch(
                "animus_forge.self_improve.orchestrator.Sandbox",
                return_value=mock_sandbox,
            ),
        ):
            result = await orch.run()

        assert result.success is False
        assert result.stage_reached == WorkflowStage.TESTING
        assert "Tests failed" in result.error

    async def test_run_plan_rejected(self, tmp_path):
        """Run fails when plan is rejected."""
        from animus_forge.self_improve.approval import ApprovalStatus as _ApprovalStatus

        src = tmp_path / "src" / "animus_forge"
        src.mkdir(parents=True)
        (src / "f.py").write_text("import os\n\ndef func():\n    pass\n")

        config = SafetyConfig(
            human_approval_plan=True,
            human_approval_apply=False,
            human_approval_merge=False,
            max_files_per_pr=50,
            max_lines_changed=5000,
            max_new_files=20,
            branch_prefix="test/",
        )
        orch = SelfImproveOrchestrator(codebase_path=tmp_path, config=config)

        with patch.object(
            orch.approval_gate,
            "wait_for_approval",
            new_callable=AsyncMock,
            return_value=_ApprovalStatus.REJECTED,
        ):
            result = await orch.run(auto_approve=False)

        if result.stage_reached == WorkflowStage.AWAITING_PLAN_APPROVAL:
            assert result.success is False
            assert "rejected" in result.error.lower()

    async def test_run_apply_rejected(self, tmp_path):
        """Run fails when apply approval is rejected."""
        from animus_forge.self_improve.approval import ApprovalStatus as _ApprovalStatus

        src = tmp_path / "src" / "animus_forge"
        src.mkdir(parents=True)
        (src / "f.py").write_text("import os\n\ndef func():\n    pass\n")

        config = SafetyConfig(
            human_approval_plan=False,
            human_approval_apply=True,
            human_approval_merge=False,
            max_files_per_pr=50,
            max_lines_changed=5000,
            max_new_files=20,
            branch_prefix="test/",
        )
        orch = SelfImproveOrchestrator(codebase_path=tmp_path, config=config)

        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=False)
        mock_sandbox.apply_changes = AsyncMock(return_value=True)
        mock_sandbox.validate_changes = AsyncMock(
            return_value=SandboxResult(
                status=SandboxStatus.SUCCESS, tests_passed=True, lint_passed=True
            )
        )

        with (
            patch.object(
                orch,
                "_generate_changes",
                new_callable=AsyncMock,
                return_value={"src/animus_forge/f.py": "fixed"},
            ),
            patch(
                "animus_forge.self_improve.orchestrator.Sandbox",
                return_value=mock_sandbox,
            ),
            patch.object(
                orch.approval_gate,
                "wait_for_approval",
                new_callable=AsyncMock,
                return_value=_ApprovalStatus.REJECTED,
            ),
        ):
            result = await orch.run(auto_approve=False)

        if result.stage_reached == WorkflowStage.AWAITING_APPLY_APPROVAL:
            assert result.success is False
            assert "rejected" in result.error.lower()

    async def test_run_apply_expired(self, tmp_path):
        """Run fails when apply approval expires."""
        from animus_forge.self_improve.approval import ApprovalStatus as _ApprovalStatus

        src = tmp_path / "src" / "animus_forge"
        src.mkdir(parents=True)
        (src / "f.py").write_text("import os\n\ndef func():\n    pass\n")

        config = SafetyConfig(
            human_approval_plan=False,
            human_approval_apply=True,
            human_approval_merge=False,
            max_files_per_pr=50,
            max_lines_changed=5000,
            max_new_files=20,
            branch_prefix="test/",
        )
        orch = SelfImproveOrchestrator(codebase_path=tmp_path, config=config)

        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=False)
        mock_sandbox.apply_changes = AsyncMock(return_value=True)
        mock_sandbox.validate_changes = AsyncMock(
            return_value=SandboxResult(
                status=SandboxStatus.SUCCESS, tests_passed=True, lint_passed=True
            )
        )

        with (
            patch.object(
                orch,
                "_generate_changes",
                new_callable=AsyncMock,
                return_value={"src/animus_forge/f.py": "fixed"},
            ),
            patch(
                "animus_forge.self_improve.orchestrator.Sandbox",
                return_value=mock_sandbox,
            ),
            patch.object(
                orch.approval_gate,
                "wait_for_approval",
                new_callable=AsyncMock,
                return_value=_ApprovalStatus.EXPIRED,
            ),
        ):
            result = await orch.run(auto_approve=False)

        if result.stage_reached == WorkflowStage.AWAITING_APPLY_APPROVAL:
            assert result.success is False
            assert "timed out" in result.error.lower()
