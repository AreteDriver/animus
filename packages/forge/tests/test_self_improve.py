"""Tests for self-improvement module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from animus_forge.self_improve import (
    SafetyConfig,
)
from animus_forge.self_improve.orchestrator import WorkflowStage


class TestSafetyConfig:
    """Tests for SafetyConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SafetyConfig()
        assert config.max_files_per_pr == 10
        assert config.max_lines_changed == 500
        assert config.max_deleted_files == 0
        assert config.max_new_files == 5
        assert config.tests_must_pass is True
        assert config.human_approval_plan is True
        assert config.human_approval_apply is True
        assert config.human_approval_merge is True
        assert config.use_branch is True
        assert config.branch_prefix == "gorgon-self-improve/"
        assert config.max_snapshots == 10
        assert config.auto_rollback_on_test_failure is True

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "protected_files": {
                "critical": ["src/auth/**", "*.key"],
                "sensitive": ["config.py"],
            },
            "limits": {
                "max_files_per_pr": 5,
                "max_lines_changed": 200,
            },
            "requirements": {
                "tests_must_pass": True,
                "human_approval": {
                    "plan": True,
                    "apply": False,
                    "merge": True,
                },
            },
        }
        config = SafetyConfig._from_dict(data)
        assert config.critical_files == ["src/auth/**", "*.key"]
        assert config.sensitive_files == ["config.py"]
        assert config.max_files_per_pr == 5
        assert config.max_lines_changed == 200
        assert config.human_approval_apply is False

    def test_config_load_missing_file(self):
        """Test loading config from missing file returns defaults."""
        config = SafetyConfig.load("/nonexistent/path.yaml")
        assert config.max_files_per_pr == 10  # Default value

    def test_config_load_from_yaml(self):
        """Test loading config from YAML file."""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "safety.yaml"
            config_path.write_text("""
protected_files:
  critical:
    - "src/security/**"
limits:
  max_files_per_pr: 3
""")
            config = SafetyConfig.load(config_path)
            assert config.critical_files == ["src/security/**"]
            assert config.max_files_per_pr == 3


# Note: SafetyChecker, ApprovalGate, RollbackManager, Sandbox, and PRManager
# tests require matching the actual implementation's API.
# The config and workflow stage tests above provide coverage of core data structures.


class TestWorkflowStage:
    """Tests for WorkflowStage enum."""

    def test_all_stages_defined(self):
        """Test all workflow stages are defined."""
        expected = [
            "idle",
            "analyzing",
            "planning",
            "awaiting_plan_approval",
            "implementing",
            "testing",
            "awaiting_apply_approval",
            "applying",
            "creating_pr",
            "awaiting_merge_approval",
            "complete",
            "failed",
            "rolled_back",
        ]
        actual = [s.value for s in WorkflowStage]
        for stage in expected:
            assert stage in actual


# Note: SelfImproveOrchestrator integration tests require
# more complex mocking. The config and workflow stage tests
# above provide coverage of core data structures.


class TestCodebaseAnalyzer:
    """Tests for CodebaseAnalyzer."""

    def test_analyzer_creation(self):
        """Test creating an analyzer."""
        from animus_forge.self_improve.analyzer import CodebaseAnalyzer

        analyzer = CodebaseAnalyzer(codebase_path=".")
        assert analyzer.codebase_path == Path(".")
        assert analyzer.provider is None

    def test_parse_ai_suggestions_valid_json(self):
        """Test parsing valid AI suggestions."""
        from animus_forge.self_improve.analyzer import (
            CodebaseAnalyzer,
            ImprovementCategory,
        )

        analyzer = CodebaseAnalyzer()
        response = """[
            {
                "category": "refactoring",
                "title": "Extract method",
                "description": "Extract common logic",
                "affected_files": ["src/main.py"],
                "priority": 2,
                "reasoning": "Reduces duplication",
                "implementation_hints": "Use helper function"
            }
        ]"""

        suggestions = analyzer._parse_ai_suggestions(response)
        assert len(suggestions) == 1
        assert suggestions[0].category == ImprovementCategory.REFACTORING
        assert suggestions[0].title == "Extract method"
        assert suggestions[0].priority == 2

    def test_parse_ai_suggestions_with_code_block(self):
        """Test parsing AI suggestions wrapped in markdown code block."""
        from animus_forge.self_improve.analyzer import CodebaseAnalyzer

        analyzer = CodebaseAnalyzer()
        response = """```json
[{"category": "bug_fixes", "title": "Fix bug", "description": "Desc", "affected_files": []}]
```"""

        suggestions = analyzer._parse_ai_suggestions(response)
        assert len(suggestions) == 1
        assert suggestions[0].title == "Fix bug"

    def test_parse_ai_suggestions_invalid_json(self):
        """Test parsing invalid JSON returns empty list."""
        from animus_forge.self_improve.analyzer import CodebaseAnalyzer

        analyzer = CodebaseAnalyzer()
        response = "This is not valid JSON"

        suggestions = analyzer._parse_ai_suggestions(response)
        assert len(suggestions) == 0

    def test_parse_ai_suggestions_invalid_category(self):
        """Test parsing with invalid category falls back to code_quality."""
        from animus_forge.self_improve.analyzer import (
            CodebaseAnalyzer,
            ImprovementCategory,
        )

        analyzer = CodebaseAnalyzer()
        response = '[{"category": "invalid_category", "title": "Test", "description": "Desc", "affected_files": []}]'

        suggestions = analyzer._parse_ai_suggestions(response)
        assert len(suggestions) == 1
        assert suggestions[0].category == ImprovementCategory.CODE_QUALITY


# ---------------------------------------------------------------------------
# Git Conflict Detection (TODO 5)
# ---------------------------------------------------------------------------


class TestConflictResult:
    """Test the ConflictResult dataclass."""

    def test_default_no_conflicts(self):
        from animus_forge.self_improve.pr_manager import ConflictResult

        result = ConflictResult()
        assert not result.has_conflicts
        assert result.conflicting_files == []
        assert result.error is None

    def test_with_conflicts(self):
        from animus_forge.self_improve.pr_manager import ConflictResult

        result = ConflictResult(
            has_conflicts=True,
            conflicting_files=["src/main.py", "README.md"],
        )
        assert result.has_conflicts
        assert len(result.conflicting_files) == 2

    def test_with_error(self):
        from animus_forge.self_improve.pr_manager import ConflictResult

        result = ConflictResult(error="git not found")
        assert not result.has_conflicts
        assert result.error == "git not found"


class TestCheckConflicts:
    """Test PRManager.check_conflicts() method."""

    def test_clean_merge_detected(self, tmp_path):
        from animus_forge.self_improve.pr_manager import PRManager

        pr = PRManager(repo_path=tmp_path)

        with patch("subprocess.run") as mock_run:
            # First call: merge succeeds (returncode=0)
            # Second call: merge --abort succeeds
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="", stderr=""),
                MagicMock(returncode=0, stdout="", stderr=""),
            ]
            result = pr.check_conflicts("feature-branch")

        assert not result.has_conflicts
        assert result.conflicting_files == []
        assert result.error is None

    def test_conflicting_files_listed(self, tmp_path):
        from animus_forge.self_improve.pr_manager import PRManager

        pr = PRManager(repo_path=tmp_path)

        with patch("subprocess.run") as mock_run:
            # Merge fails (conflict), diff lists files, abort cleans up
            mock_run.side_effect = [
                MagicMock(returncode=1, stdout="CONFLICT", stderr=""),
                MagicMock(
                    returncode=0,
                    stdout="src/main.py\nREADME.md\n",
                    stderr="",
                ),
                MagicMock(returncode=0, stdout="", stderr=""),
            ]
            result = pr.check_conflicts("feature-branch")

        assert result.has_conflicts
        assert "src/main.py" in result.conflicting_files
        assert "README.md" in result.conflicting_files

    def test_merge_aborted_cleanly(self, tmp_path):
        from animus_forge.self_improve.pr_manager import PRManager

        pr = PRManager(repo_path=tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=1, stdout="CONFLICT", stderr=""),
                MagicMock(returncode=0, stdout="file.py\n", stderr=""),
                MagicMock(returncode=0, stdout="", stderr=""),
            ]
            pr.check_conflicts("feature-branch")

        # Verify merge --abort was called
        abort_calls = [c for c in mock_run.call_args_list if "--abort" in c[0][0]]
        assert len(abort_calls) == 1

    def test_git_not_available(self, tmp_path):
        from animus_forge.self_improve.pr_manager import PRManager

        pr = PRManager(repo_path=tmp_path)

        with patch("subprocess.run", side_effect=FileNotFoundError("git")):
            result = pr.check_conflicts("feature-branch")

        assert not result.has_conflicts
        assert result.error == "git not found"

    def test_timeout_handled(self, tmp_path):
        from animus_forge.self_improve.pr_manager import PRManager

        pr = PRManager(repo_path=tmp_path)

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="git", timeout=60),
        ):
            result = pr.check_conflicts("feature-branch")

        assert not result.has_conflicts
        assert "timed out" in result.error
