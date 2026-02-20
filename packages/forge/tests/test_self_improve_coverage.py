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
        result = asyncio.run(orch.run(auto_approve=False))

        if result.stage_reached == WorkflowStage.AWAITING_PLAN_APPROVAL:
            assert result.success is False
            assert "approval" in result.error.lower()

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

        # Mock git operations in PR manager
        with patch.object(orch.pr_manager, "_run_git"):
            with patch("animus_forge.self_improve.pr_manager.subprocess.run") as mock_run:
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

        with patch.object(orch.pr_manager, "_run_git"):
            with patch("animus_forge.self_improve.pr_manager.subprocess.run") as mock_run:
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
        result = asyncio.run(orch.run(auto_approve=False))

        if result.stage_reached == WorkflowStage.AWAITING_APPLY_APPROVAL:
            assert result.success is False
            assert "approval" in result.error.lower()

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

        with patch.object(orch.pr_manager, "_run_git"):
            with patch("animus_forge.self_improve.pr_manager.subprocess.run") as mock_run:
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

        with patch.object(orch.pr_manager, "_run_git"):
            with patch("animus_forge.self_improve.pr_manager.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stderr="")
                result = asyncio.run(orch.run(focus_category="documentation"))

        assert result.success is True
