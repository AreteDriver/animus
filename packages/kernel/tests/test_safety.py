"""Tests for sandbox safety configuration and checking."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from animus_kernel.sandbox.safety import SafetyChecker, SafetyConfig, SafetyViolation


# ═══════════════════════════════════════════════════════════════════
# SafetyConfig tests
# ═══════════════════════════════════════════════════════════════════


class TestSafetyConfig:
    def test_default_values(self):
        config = SafetyConfig()
        assert config.max_files_per_pr == 10
        assert config.max_lines_changed == 500
        assert config.max_deleted_files == 0
        assert config.max_new_files == 5
        assert config.tests_must_pass is True
        assert config.human_approval_plan is True
        assert config.human_approval_apply is True
        assert config.human_approval_merge is True
        assert config.allow_self_targeting is False
        assert config.max_recursive_depth == 3
        assert config.use_branch is True
        assert config.branch_prefix == "animus-kernel-self-improve/"
        assert config.isolated_execution is True
        assert config.sandbox_timeout == 300
        assert config.max_snapshots == 10
        assert config.auto_rollback_on_test_failure is True

    def test_load_missing_file_returns_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SafetyConfig.load(Path(tmpdir) / "nonexistent.yaml")
        assert isinstance(config, SafetyConfig)
        assert config.max_files_per_pr == 10

    def test_load_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "safety.yaml"
            yaml_path.write_text("""
protected_files:
  critical:
    - "*.key"
    - "secrets.yaml"
  sensitive:
    - "config/production.yaml"
limits:
  max_files_per_pr: 3
  max_lines_changed: 200
  max_deleted_files: 1
  max_new_files: 2
requirements:
  tests_must_pass: false
  human_approval:
    plan: false
    apply: false
    merge: false
allowed_categories:
  - refactoring
  - documentation
denied_categories:
  - bug_fixes
self_targeting:
  enabled: true
  max_recursive_depth: 5
sandbox:
  use_branch: false
  branch_prefix: "test/"
  isolated_execution: false
  timeout: 60
rollback:
  max_snapshots: 5
  auto_rollback_on_test_failure: false
""")
            config = SafetyConfig.load(yaml_path)

        assert config.max_files_per_pr == 3
        assert config.max_lines_changed == 200
        assert config.max_deleted_files == 1
        assert config.max_new_files == 2
        assert config.tests_must_pass is False
        assert config.human_approval_plan is False
        assert config.critical_files == ["*.key", "secrets.yaml"]
        assert config.sensitive_files == ["config/production.yaml"]
        assert config.allowed_categories == ["refactoring", "documentation"]
        assert config.denied_categories == ["bug_fixes"]
        assert config.use_branch is False
        assert config.branch_prefix == "test/"
        assert config.isolated_execution is False
        assert config.sandbox_timeout == 60
        assert config.max_snapshots == 5
        assert config.auto_rollback_on_test_failure is False
        assert config.allow_self_targeting is True
        assert config.max_recursive_depth == 5


# ═══════════════════════════════════════════════════════════════════
# SafetyChecker tests
# ═══════════════════════════════════════════════════════════════════


class TestSafetyChecker:
    def test_protected_file_matching(self):
        config = SafetyConfig(critical_files=["*.key", "secrets.*", "config/production.yaml"])
        checker = SafetyChecker(config)

        assert checker.is_protected_file("api.key") is True
        assert checker.is_protected_file("secrets.yaml") is True
        assert checker.is_protected_file("config/production.yaml") is True
        assert checker.is_protected_file("src/main.py") is False
        assert checker.is_protected_file("README.md") is False

    def test_sensitive_file_matching(self):
        config = SafetyConfig(sensitive_files=["config/*.yaml"])
        checker = SafetyChecker(config)

        assert checker.is_sensitive_file("config/production.yaml") is True
        assert checker.is_sensitive_file("config/dev.yaml") is True
        assert checker.is_sensitive_file("src/main.py") is False

    def test_allowed_file_with_empty_allow_list(self):
        config = SafetyConfig(allowed_files=[])
        checker = SafetyChecker(config)

        # Empty allow-list = legacy mode, all files allowed
        assert checker.is_allowed_file("src/main.py") is True
        assert checker.is_allowed_file("anything.py") is True

    def test_allowed_file_with_non_empty_allow_list(self):
        config = SafetyConfig(allowed_files=["src/*.py", "tests/*.py"])
        checker = SafetyChecker(config)

        assert checker.is_allowed_file("src/main.py") is True
        assert checker.is_allowed_file("tests/test_main.py") is True
        assert checker.is_allowed_file("docs/readme.md") is False
        assert checker.is_allowed_file("lib/helper.py") is False

    def test_allowed_category_default(self):
        config = SafetyConfig()
        checker = SafetyChecker(config)

        # Default: all categories allowed
        assert checker.is_allowed_category("performance") is True
        assert checker.is_allowed_category("refactoring") is True

    def test_allowed_category_with_deny_list(self):
        config = SafetyConfig(denied_categories=["bug_fixes", "security"])
        checker = SafetyChecker(config)

        assert checker.is_allowed_category("performance") is True
        assert checker.is_allowed_category("bug_fixes") is False
        assert checker.is_allowed_category("security") is False

    def test_allowed_category_with_allow_list(self):
        config = SafetyConfig(allowed_categories=["refactoring", "documentation"])
        checker = SafetyChecker(config)

        assert checker.is_allowed_category("refactoring") is True
        assert checker.is_allowed_category("documentation") is True
        assert checker.is_allowed_category("performance") is False

    def test_check_changes_no_violations(self):
        config = SafetyConfig()
        checker = SafetyChecker(config)

        violations = checker.check_changes(
            files_modified=["src/main.py"],
            files_added=["src/helper.py"],
            files_deleted=[],
            lines_changed=50,
        )
        assert violations == []

    def test_check_changes_file_limit_exceeded(self):
        config = SafetyConfig(max_files_per_pr=2)
        checker = SafetyChecker(config)

        violations = checker.check_changes(
            files_modified=["a.py", "b.py", "c.py"],
            files_added=[],
            files_deleted=[],
            lines_changed=10,
        )
        assert len(violations) == 1
        assert violations[0].violation_type == "file_limit"
        assert "Too many files" in violations[0].message

    def test_check_changes_lines_limit_exceeded(self):
        config = SafetyConfig(max_lines_changed=100)
        checker = SafetyChecker(config)

        violations = checker.check_changes(
            files_modified=["src/main.py"],
            files_added=[],
            files_deleted=[],
            lines_changed=200,
        )
        assert len(violations) == 1
        assert violations[0].violation_type == "lines_limit"

    def test_check_changes_protected_file(self):
        config = SafetyConfig(critical_files=["secrets.yaml"])
        checker = SafetyChecker(config)

        violations = checker.check_changes(
            files_modified=["secrets.yaml"],
            files_added=[],
            files_deleted=[],
            lines_changed=10,
        )
        assert any(v.violation_type == "protected_file" for v in violations)

    def test_check_changes_not_allow_listed(self):
        config = SafetyConfig(allowed_files=["src/*.py"])
        checker = SafetyChecker(config)

        violations = checker.check_changes(
            files_modified=["docs/readme.md"],
            files_added=["lib/helper.py"],
            files_deleted=[],
            lines_changed=10,
        )
        assert any(v.violation_type == "not_allow_listed" for v in violations)
        # Two violations: one for docs/readme.md, one for lib/helper.py
        assert len([v for v in violations if v.violation_type == "not_allow_listed"]) == 2

    def test_check_changes_sensitive_file_warning(self):
        config = SafetyConfig(sensitive_files=["config/*.yaml"])
        checker = SafetyChecker(config)

        violations = checker.check_changes(
            files_modified=["config/production.yaml"],
            files_added=[],
            files_deleted=[],
            lines_changed=10,
        )
        sensitive = [v for v in violations if v.violation_type == "sensitive_file"]
        assert len(sensitive) == 1
        assert sensitive[0].severity == "warning"

    def test_check_changes_delete_limit(self):
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
        config = SafetyConfig(max_new_files=1)
        checker = SafetyChecker(config)

        violations = checker.check_changes(
            files_modified=[],
            files_added=["a.py", "b.py"],
            files_deleted=[],
            lines_changed=0,
        )
        assert any(v.violation_type == "new_file_limit" for v in violations)

    def test_check_changes_denied_category(self):
        config = SafetyConfig(denied_categories=["performance"])
        checker = SafetyChecker(config)

        violations = checker.check_changes(
            files_modified=["src/main.py"],
            files_added=[],
            files_deleted=[],
            lines_changed=10,
            category="performance",
        )
        assert any(v.violation_type == "denied_category" for v in violations)

    def test_has_blocking_violations_with_errors(self):
        config = SafetyConfig()
        checker = SafetyChecker(config)

        violations = [
            SafetyViolation("a.py", "file_limit", "too many files"),
        ]
        assert checker.has_blocking_violations(violations) is True

    def test_has_blocking_violations_only_warnings(self):
        config = SafetyConfig()
        checker = SafetyChecker(config)

        violations = [
            SafetyViolation("a.py", "sensitive_file", "review needed", severity="warning"),
        ]
        assert checker.has_blocking_violations(violations) is False
