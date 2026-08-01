"""Safety configuration and checking for self-improvement.

This module enforces what files can be modified and what changes
are allowed during self-improvement operations.
"""

from __future__ import annotations

import fnmatch
import logging
import re
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
    # Stage 4.B — explicit allow-list. When non-empty, *only* files
    # matching these patterns can be modified. Empty = no allow-list
    # enforcement (legacy deny-list-only behavior).
    allowed_files: list[str] = field(default_factory=list)

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
            allowed_files=data.get("allowed_files", []),
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

    def is_allowed_file(self, file_path: str) -> bool:
        """Check if a file is permitted under the Stage 4.B allow-list.

        Returns True if no allow-list is configured (legacy mode — fall
        back to deny-list-only). Otherwise returns True iff the file
        matches at least one allow pattern.

        Args:
            file_path: Path to check.
        """
        if not self.config.allowed_files:
            return True  # no allow-list enforcement
        return self._matches_patterns(file_path, self.config.allowed_files)

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

        # Stage 4.B — allow-list enforcement. If config.allowed_files is
        # non-empty, every modified/added file must match at least one
        # allow pattern. Deletions are still governed by max_deleted_files
        # above; an allow-list violation on a deleted file is not raised
        # separately to avoid double-counting against protected_file.
        for file_path in files_modified + files_added:
            if not self.is_allowed_file(file_path):
                violations.append(
                    SafetyViolation(
                        file_path=file_path,
                        violation_type="not_allow_listed",
                        message=(
                            f"File is outside the allow-list and cannot be modified: {file_path}"
                        ),
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

    # ------------------------------------------------------------------
    # E11 — Red-team loop-abuse hardening
    # ------------------------------------------------------------------

    _PROBE_PATTERNS: tuple[str, ...] = (
        # Jailbreak / instruction-override probes
        r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|directives)",
        r"ignore\s+(?:the\s+)?system\s+(?:prompt|message|instruction)",
        r"you\s+(?:are\s+now|have\s+been)\s+(?:in|switched\s+to)\s+DAN\s+mode",
        r"DAN\s*[:\-]\s*do\s+anything\s+now",
        r"(?:developer|debug|admin)\s+mode\s*(?:activated|enabled|on)",
        r"simulate\s+(?:being|acting\s+as)\s+(?:an?\s+)?(?:unrestricted|unfiltered|uncensored)",
        r"pretend\s+(?:you\s+are|to\s+be)\s+(?:an?\s+)?(?:evil|malicious|hacker|attacker)",
        r"(?:bypass|disable|turn\s+off)\s+(?:safety|guardrail|protection|restriction|filter)",
        # Self-improve loop manipulation
        r"(?:approve|auto-approve|skip)\s+(?:all\s+)?(?:safety|approval|human\s+review|check)",
        r"set\s+(?:ANIMUS_FORGE_ALLOW_AUTO_APPROVE|auto_approve)\s*=\s*(?:1|true|yes)",
        r"inject\s+(?:this\s+)?(?:code|payload|exploit)\s+(?:into|inside)\s+(?:the\s+)?(?:system|orchestrator|pipeline)",
        # Encoding evasion (common red-team obfuscation)
        r"base64\s*[:\-]\s*[A-Za-z0-9+/=]{40,}",
        r"rot13\s*[:\-]\s*[A-Za-z ]{20,}",
        # Repetition / flooding probes
        r"(.)\1{80,}",
    )

    def is_probe_shaped(self, text: str) -> bool:
        """Check if text matches known red-team probe signatures.

        Probes may arrive disguised as improvement suggestions, attempting
        to manipulate the self-improve loop into bypassing safety gates or
        injecting malicious code. This is a coarse-grained filter; false
        positives are acceptable (human review is the backstop).

        Args:
            text: Suggestion title or description to check.

        Returns:
            True if the text looks like an adversarial probe.
        """
        lowered = text.lower()
        for pattern in self._PROBE_PATTERNS:
            if re.search(pattern, lowered, re.IGNORECASE):
                return True
        return False

    def check_suggestion_content(self, suggestions: list[Any]) -> list[SafetyViolation]:
        """Check improvement suggestions for probe-shaped content.

        E11 — before an analyzer suggestion becomes a plan, each title and
        description is screened for adversarial patterns. A single probe
        triggers a blocking violation that halts the pipeline before any
        code is generated or applied.

        Args:
            suggestions: Raw suggestions from the analyzer.

        Returns:
            List of violations (empty if all suggestions look legitimate).
        """
        violations: list[SafetyViolation] = []
        for suggestion in suggestions:
            title = getattr(suggestion, "title", "")
            description = getattr(suggestion, "description", "")
            for field_name, field_text in (("title", title), ("description", description)):
                if self.is_probe_shaped(field_text):
                    violations.append(
                        SafetyViolation(
                            file_path="",
                            violation_type="probe_detected",
                            message=(
                                f"Red-team probe detected in suggestion "
                                f"'{title}' ({field_name}): content matches known "
                                f"adversarial patterns — aborting self-improve loop."
                            ),
                        )
                    )
                    # Don't duplicate-report the same suggestion
                    break
        return violations

    def has_blocking_violations(self, violations: list[SafetyViolation]) -> bool:
        """Check if any violations are blocking (errors).

        Args:
            violations: List of violations.

        Returns:
            True if there are error-level violations.
        """
        return any(v.severity == "error" for v in violations)
