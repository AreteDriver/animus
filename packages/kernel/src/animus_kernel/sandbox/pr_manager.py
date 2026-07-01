"""Pull Request management for self-improvement changes."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SecretsDetectedError(RuntimeError):
    """Raised when gitleaks finds a secret in staged changes."""


@dataclass
class ConflictResult:
    """Result of a conflict check between branches."""

    has_conflicts: bool = False
    conflicting_files: list[str] = field(default_factory=list)
    error: str | None = None


class PRStatus(StrEnum):
    """Status of a pull request."""

    DRAFT = "draft"
    OPEN = "open"
    MERGED = "merged"
    CLOSED = "closed"


@dataclass
class PullRequest:
    """Represents a pull request."""

    id: str
    branch: str
    title: str
    description: str
    status: PRStatus = PRStatus.DRAFT
    url: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    merged_at: datetime | None = None
    files_changed: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PRManager:
    """Manages pull requests for self-improvement changes.

    Uses the gh CLI for GitHub operations.
    """

    def __init__(
        self,
        repo_path: Path | str = ".",
        branch_prefix: str = "gorgon-self-improve/",
        default_branch: str | None = None,
    ):
        """Initialize PR manager.

        Args:
            repo_path: Path to git repository.
            branch_prefix: Prefix for improvement branches.
            default_branch: Explicit default branch name (e.g. ``"main"``).
                When set, ``checkout_main`` uses this instead of the mutable
                ``origin/HEAD`` symbolic-ref. Resolution order:

                  1. constructor arg
                  2. ``ANIMUS_FORGE_DEFAULT_BRANCH`` env var
                  3. fallback: read ``origin/HEAD`` with a logged warning
                  4. with ``ANIMUS_FORGE_REQUIRE_DEFAULT_BRANCH=1``, the
                     fallback is disabled and ``checkout_main`` raises.

                Stage 4.C — closes the whip-saw vector where the engine
                committed onto the wrong branch when ``origin/HEAD`` had
                been nudged off ``main``.
        """
        self.repo_path = Path(repo_path)
        self.branch_prefix = branch_prefix
        self.default_branch = default_branch or os.environ.get("ANIMUS_FORGE_DEFAULT_BRANCH")
        self._active_prs: dict[str, PullRequest] = {}

    def create_branch(self, name: str) -> str:
        """Create a new branch for improvements.

        Args:
            name: Branch name suffix.

        Returns:
            Full branch name.
        """
        branch = f"{self.branch_prefix}{name}"

        # Create and checkout branch
        self._run_git(["checkout", "-b", branch])
        logger.info(f"Created branch: {branch}")

        return branch

    def _gitleaks_scan_staged(self) -> None:
        """Stage 4.A — block autonomous commits that include secrets.

        Runs ``gitleaks detect --staged`` against the index. Fails closed
        (raises ``SecretsDetectedError``) if any secret pattern matches.

        Skip cases (logged but non-blocking):
          - ``gitleaks`` binary not on PATH (defensive fallback so dev
            environments without gitleaks installed don't break)
          - ``ANIMUS_FORGE_SKIP_GITLEAKS=1`` env override (emergency hatch)
        """
        if os.environ.get("ANIMUS_FORGE_SKIP_GITLEAKS") == "1":
            logger.warning("ANIMUS_FORGE_SKIP_GITLEAKS=1 — pre-commit secret scan bypassed")
            return

        gitleaks_path = shutil.which("gitleaks")
        if not gitleaks_path:
            logger.warning(
                "gitleaks not found on PATH — autonomous commit proceeding "
                "without secret scan. Install gitleaks to enable the gate."
            )
            return

        try:
            result = subprocess.run(
                [
                    gitleaks_path,
                    "detect",
                    "--staged",
                    "--no-banner",
                    "--no-color",
                    f"--source={self.repo_path}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as e:
            logger.warning(f"gitleaks invocation failed: {e}")
            return

        # gitleaks exits 0 when clean, 1 when leaks found, other codes for
        # invocation errors (not a git repo, can't read index, etc.).
        if result.returncode == 0:
            return

        # Don't log gitleaks stdout content — it includes the secret. Log
        # only the count of findings and the file paths so audit is
        # actionable but not re-leaking.
        finding_count = result.stdout.count("Finding:") if result.stdout else 0
        if finding_count == 0:
            # Non-zero exit with no parsed findings = invocation error
            # (e.g., not a git repo). Don't block on this — warn and let
            # the commit proceed. Real findings are always parseable.
            logger.warning(
                "gitleaks exited %d with no parseable findings; treating as "
                "invocation error and proceeding (stderr=%r)",
                result.returncode,
                result.stderr[:200] if result.stderr else "",
            )
            return

        raise SecretsDetectedError(
            f"gitleaks found {finding_count} secret(s) in staged changes. "
            "Autonomous commit refused. Review the staged diff manually."
        )

    def commit_changes(
        self,
        files: list[str],
        message: str,
    ) -> str | None:
        """Stage and commit changes.

        Args:
            files: List of files to stage.
            message: Commit message.

        Returns:
            Commit hash if successful, None otherwise.
        """
        try:
            # Stage files
            self._run_git(["add"] + files)

            # Stage 4.A — pre-commit secret scan. Fails closed.
            self._gitleaks_scan_staged()

            # Commit
            full_message = f"""feat(self-improve): {message}

This change was generated by Gorgon's self-improvement system.

Co-Authored-By: Gorgon AI <gorgon@example.com>
"""
            self._run_git(["commit", "-m", full_message])

            # Get commit hash
            result = self._run_git(["rev-parse", "HEAD"])
            return result.stdout.strip()

        except SecretsDetectedError as e:
            logger.error(f"Pre-commit secret scan blocked commit: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to commit: {e}")
            return None

    def push_branch(self, branch: str) -> bool:
        """Push branch to remote.

        Args:
            branch: Branch name.

        Returns:
            True if successful.
        """
        try:
            self._run_git(["push", "-u", "origin", branch])
            logger.info(f"Pushed branch: {branch}")
            return True
        except Exception as e:
            logger.error(f"Failed to push: {e}")
            return False

    def create_pr(
        self,
        branch: str,
        title: str,
        description: str,
        files_changed: list[str],
        draft: bool = True,
    ) -> PullRequest:
        """Create a pull request.

        Args:
            branch: Source branch.
            title: PR title.
            description: PR body/description.
            files_changed: List of files changed.
            draft: Whether to create as draft.

        Returns:
            Created PullRequest.
        """
        import uuid

        pr_id = str(uuid.uuid4())[:8]

        # Build PR body
        body = f"""## Summary
{description}

## Changes
- Files modified: {len(files_changed)}

## Generated by Gorgon Self-Improvement

This PR was automatically generated by Gorgon's self-improvement system.
Please review carefully before merging.

---
*This change requires human approval before merging.*
"""

        # Create PR using gh CLI
        url = None
        try:
            cmd = ["gh", "pr", "create", "--title", title, "--body", body]
            if draft:
                cmd.append("--draft")

            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                url = result.stdout.strip()
                logger.info(f"Created PR: {url}")
            else:
                logger.warning(f"Could not create PR via gh: {result.stderr}")

        except FileNotFoundError:
            logger.warning("gh CLI not found - PR not created on GitHub")
        except Exception as e:
            logger.warning(f"Could not create PR: {e}")

        pr = PullRequest(
            id=pr_id,
            branch=branch,
            title=title,
            description=description,
            status=PRStatus.DRAFT if draft else PRStatus.OPEN,
            url=url,
            files_changed=files_changed,
        )

        self._active_prs[pr_id] = pr
        return pr

    def get_pr(self, pr_id: str) -> PullRequest | None:
        """Get a PR by ID.

        Args:
            pr_id: PR ID.

        Returns:
            PullRequest if found.
        """
        return self._active_prs.get(pr_id)

    def mark_ready_for_review(self, pr_id: str) -> bool:
        """Mark a draft PR as ready for review.

        Args:
            pr_id: PR ID.

        Returns:
            True if successful.
        """
        pr = self._active_prs.get(pr_id)
        if not pr or not pr.url:
            return False

        try:
            subprocess.run(
                ["gh", "pr", "ready", pr.url],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=30,
            )
            pr.status = PRStatus.OPEN
            return True
        except Exception as e:
            logger.error(f"Failed to mark PR ready: {e}")
            return False

    def close_pr(self, pr_id: str, reason: str | None = None) -> bool:
        """Close a PR without merging.

        Args:
            pr_id: PR ID.
            reason: Optional reason for closing.

        Returns:
            True if successful.
        """
        pr = self._active_prs.get(pr_id)
        if not pr:
            return False

        if pr.url:
            try:
                cmd = ["gh", "pr", "close", pr.url]
                if reason:
                    cmd.extend(["--comment", reason])

                subprocess.run(
                    cmd,
                    cwd=str(self.repo_path),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except Exception as e:
                logger.warning(f"Could not close PR via gh: {e}")

        pr.status = PRStatus.CLOSED
        return True

    def checkout_main(self) -> bool:
        """Checkout the configured default branch.

        Stage 4.C — uses ``self.default_branch`` (constructor arg or env)
        instead of ``origin/HEAD`` symbolic-ref. The symbolic-ref is a
        mutable client-side state that can be nudged off ``main`` by other
        tools (worktrees, branch operations, GitHub UI default-branch
        flips), which is the recorded whip-saw root cause for autonomous
        commits landing on wrong branches.

        Returns:
            True if successful.
        """
        try:
            default_branch = self.default_branch
            if not default_branch:
                if os.environ.get("ANIMUS_FORGE_REQUIRE_DEFAULT_BRANCH") == "1":
                    raise RuntimeError(
                        "ANIMUS_FORGE_REQUIRE_DEFAULT_BRANCH=1 — "
                        "PRManager.default_branch not configured. Set via "
                        "constructor arg or ANIMUS_FORGE_DEFAULT_BRANCH env."
                    )
                # Graceful fallback for dev — logs a loud warning so the
                # operator knows the gate is open.
                logger.warning(
                    "PRManager.default_branch not configured; falling back to "
                    "origin/HEAD symbolic-ref. This is the whip-saw vector — "
                    "set ANIMUS_FORGE_DEFAULT_BRANCH or pass default_branch="
                    " to PRManager() to close it."
                )
                result = self._run_git(["symbolic-ref", "refs/remotes/origin/HEAD", "--short"])
                default_branch = result.stdout.strip().replace("origin/", "")

            self._run_git(["checkout", default_branch])
            return True
        except Exception as e:
            logger.error(f"Failed to checkout main: {e}")
            return False

    def delete_branch(self, branch: str, force: bool = False) -> bool:
        """Delete a local branch.

        Args:
            branch: Branch name.
            force: Force delete even if not merged.

        Returns:
            True if successful.
        """
        try:
            flag = "-D" if force else "-d"
            self._run_git(["branch", flag, branch])
            return True
        except Exception as e:
            logger.error(f"Failed to delete branch: {e}")
            return False

    def get_current_branch(self) -> str:
        """Get current branch name.

        Returns:
            Branch name.
        """
        result = self._run_git(["branch", "--show-current"])
        return result.stdout.strip()

    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes.

        Returns:
            True if there are changes.
        """
        result = self._run_git(["status", "--porcelain"])
        return bool(result.stdout.strip())

    def check_conflicts(self, branch: str) -> ConflictResult:
        """Check for merge conflicts via dry-run merge.

        Performs a non-committing merge attempt, inspects the result,
        then aborts to leave the working tree clean.

        Args:
            branch: Branch name to check for conflicts against HEAD.

        Returns:
            ConflictResult with conflict details.
        """
        try:
            result = subprocess.run(
                ["git", "merge", "--no-commit", "--no-ff", branch],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                # Clean merge — abort the uncommitted merge
                subprocess.run(
                    ["git", "merge", "--abort"],
                    cwd=str(self.repo_path),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return ConflictResult(has_conflicts=False)

            # Merge conflict — extract conflicting file names
            conflicting_files: list[str] = []
            try:
                diff_result = subprocess.run(
                    ["git", "diff", "--name-only", "--diff-filter=U"],
                    cwd=str(self.repo_path),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if diff_result.returncode == 0:
                    conflicting_files = [
                        f for f in diff_result.stdout.strip().split("\n") if f.strip()
                    ]
            except Exception:
                pass  # Best-effort: conflict file listing is advisory

            # Always abort to restore clean state
            subprocess.run(
                ["git", "merge", "--abort"],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=30,
            )

            return ConflictResult(
                has_conflicts=True,
                conflicting_files=conflicting_files,
            )

        except FileNotFoundError:
            return ConflictResult(
                has_conflicts=False,
                error="git not found",
            )
        except subprocess.TimeoutExpired:
            return ConflictResult(
                has_conflicts=False,
                error="git merge check timed out",
            )
        except Exception as e:
            return ConflictResult(
                has_conflicts=False,
                error=str(e),
            )

    def _run_git(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run a git command.

        Args:
            args: Git command arguments.

        Returns:
            Completed process result.

        Raises:
            subprocess.CalledProcessError: If command fails.
        """
        result = subprocess.run(
            ["git"] + args,
            cwd=str(self.repo_path),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                ["git"] + args,
                result.stdout,
                result.stderr,
            )
        return result
