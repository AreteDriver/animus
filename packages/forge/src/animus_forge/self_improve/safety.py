"""Safety configuration and checking for self-improvement.

This module enforces what files can be modified and what changes
are allowed during self-improvement operations.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Configuration for self-improvement safety.

    Loaded from config/self_improve_safety.yaml.
    """

    # Protected files
    critical_files: list[str] = field(default_factory=list)
    sensitive_files: list[str] = field(default_factory=list)

    # Limits
    max_files_per_pr: int = 10
    max_lines_changed: int = 500
    max_deleted_files: int = 0
    max_new_files: int = 5

    # Requirements
    tests_must_pass: bool = True
    human_approval_plan: bool = True
    human_approval_apply: bool = True
    human_approval_merge: bool = True

    # Categories
    allowed_categories: list[str] = field(default_factory=list)
    denied_categories: list[str] = field(default_factory=list)

    # Sandbox
    use_branch: bool = True
    branch_prefix: str = "gorgon-self-improve/"
    isolated_execution: bool = True
    sandbox_timeout: int = 300

    # Rollback
    max_snapshots: int = 10
    auto_rollback_on_test_failure: bool = True

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> SafetyConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file. Defaults to config/self_improve_safety.yaml

        Returns:
            Loaded SafetyConfig instance.
        """
        if config_path is None:
            config_path = Path("config/self_improve_safety.yaml")

        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Safety config not found at {config_path}, using defaults")
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> SafetyConfig:
        """Create config from dictionary."""
        protected = data.get("protected_files", {})
        limits = data.get("limits", {})
        requirements = data.get("requirements", {})
        human_approval = requirements.get("human_approval", {})
        sandbox = data.get("sandbox", {})
        rollback = data.get("rollback", {})

        return cls(
            critical_files=protected.get("critical", []),
            sensitive_files=protected.get("sensitive", []),
            max_files_per_pr=limits.get("max_files_per_pr", 10),
            max_lines_changed=limits.get("max_lines_changed", 500),
            max_deleted_files=limits.get("max_deleted_files", 0),
            max_new_files=limits.get("max_new_files", 5),
            tests_must_pass=requirements.get("tests_must_pass", True),
            human_approval_plan=human_approval.get("plan", True),
            human_approval_apply=human_approval.get("apply", True),
            human_approval_merge=human_approval.get("merge", True),
            allowed_categories=data.get("allowed_categories", []),
            denied_categories=data.get("denied_categories", []),
            use_branch=sandbox.get("use_branch", True),
            branch_prefix=sandbox.get("branch_prefix", "gorgon-self-improve/"),
            isolated_execution=sandbox.get("isolated_execution", True),
            sandbox_timeout=sandbox.get("timeout", 300),
            max_snapshots=rollback.get("max_snapshots", 10),
            auto_rollback_on_test_failure=rollback.get("auto_rollback_on_test_failure", True),
        )


@dataclass
class SafetyViolation:
    """Represents a safety violation."""

    file_path: str
    violation_type: str
    message: str
    severity: str = "error"  # error, warning


class SafetyChecker:
    """Checks proposed changes against safety configuration."""

    def __init__(self, config: SafetyConfig | None = None):
        """Initialize with configuration.

        Args:
            config: Safety configuration. Loads default if not provided.
        """
        self.config = config or SafetyConfig.load()

    def is_protected_file(self, file_path: str) -> bool:
        """Check if a file is protected (cannot be modified).

        Args:
            file_path: Path to check.

        Returns:
            True if file is protected.
        """
        return self._matches_patterns(file_path, self.config.critical_files)

    def is_sensitive_file(self, file_path: str) -> bool:
        """Check if a file is sensitive (requires extra review).

        Args:
            file_path: Path to check.

        Returns:
            True if file is sensitive.
        """
        return self._matches_patterns(file_path, self.config.sensitive_files)

    def _matches_patterns(self, file_path: str, patterns: list[str]) -> bool:
        """Check if file matches any pattern.

        Args:
            file_path: File path to check.
            patterns: List of glob patterns.

        Returns:
            True if matches any pattern.
        """
        for pattern in patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        return False

    def is_allowed_category(self, category: str) -> bool:
        """Check if an improvement category is allowed.

        Args:
            category: Category name.

        Returns:
            True if category is allowed.
        """
        if category in self.config.denied_categories:
            return False
        if self.config.allowed_categories:
            return category in self.config.allowed_categories
        return True

    def check_changes(
        self,
        files_modified: list[str],
        files_added: list[str],
        files_deleted: list[str],
        lines_changed: int,
        category: str | None = None,
    ) -> list[SafetyViolation]:
        """Check proposed changes for safety violations.

        Args:
            files_modified: List of files being modified.
            files_added: List of new files being added.
            files_deleted: List of files being deleted.
            lines_changed: Total lines changed.
            category: Optional improvement category.

        Returns:
            List of safety violations (empty if safe).
        """
        violations = []

        # Check file limits
        total_files = len(files_modified) + len(files_added)
        if total_files > self.config.max_files_per_pr:
            violations.append(
                SafetyViolation(
                    file_path="",
                    violation_type="file_limit",
                    message=f"Too many files ({total_files} > {self.config.max_files_per_pr})",
                )
            )

        if len(files_deleted) > self.config.max_deleted_files:
            violations.append(
                SafetyViolation(
                    file_path="",
                    violation_type="delete_limit",
                    message=f"Too many deletions ({len(files_deleted)} > {self.config.max_deleted_files})",
                )
            )

        if len(files_added) > self.config.max_new_files:
            violations.append(
                SafetyViolation(
                    file_path="",
                    violation_type="new_file_limit",
                    message=f"Too many new files ({len(files_added)} > {self.config.max_new_files})",
                )
            )

        # Check lines changed
        if lines_changed > self.config.max_lines_changed:
            violations.append(
                SafetyViolation(
                    file_path="",
                    violation_type="lines_limit",
                    message=f"Too many lines changed ({lines_changed} > {self.config.max_lines_changed})",
                )
            )

        # Check protected files
        all_files = files_modified + files_added + files_deleted
        for file_path in all_files:
            if self.is_protected_file(file_path):
                violations.append(
                    SafetyViolation(
                        file_path=file_path,
                        violation_type="protected_file",
                        message=f"File is protected and cannot be modified: {file_path}",
                    )
                )

        # Check category
        if category and not self.is_allowed_category(category):
            violations.append(
                SafetyViolation(
                    file_path="",
                    violation_type="denied_category",
                    message=f"Improvement category '{category}' is not allowed",
                )
            )

        # Log sensitive file warnings
        for file_path in files_modified:
            if self.is_sensitive_file(file_path):
                violations.append(
                    SafetyViolation(
                        file_path=file_path,
                        violation_type="sensitive_file",
                        message=f"File is sensitive and requires extra review: {file_path}",
                        severity="warning",
                    )
                )

        return violations

    def has_blocking_violations(self, violations: list[SafetyViolation]) -> bool:
        """Check if any violations are blocking (errors).

        Args:
            violations: List of violations.

        Returns:
            True if there are error-level violations.
        """
        return any(v.severity == "error" for v in violations)
