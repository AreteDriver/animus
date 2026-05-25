"""Tests for the Stage 4 Forge sandbox hardening.

Covers three substages:

- **4.A** — gitleaks pre-commit on autonomous commits (pr_manager.py).
- **4.B** — allow-list flip in safety config (safety.py + safety.yaml).
- **4.C** — origin/HEAD whip-saw fix (pr_manager.py checkout_main).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus_forge.self_improve.pr_manager import PRManager, SecretsDetectedError
from animus_forge.self_improve.safety import SafetyChecker, SafetyConfig

# ---------------------------------------------------------------------------
# 4.A — gitleaks pre-commit
# ---------------------------------------------------------------------------


class TestGitleaksPreCommit:
    """Pre-commit secret scan blocks autonomous commits."""

    def test_skip_env_var_bypasses_scan(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANIMUS_FORGE_SKIP_GITLEAKS", "1")
        mgr = PRManager(repo_path=tmp_path)
        # Should not raise even without gitleaks running.
        mgr._gitleaks_scan_staged()

    def test_no_gitleaks_binary_logs_warning_and_proceeds(self, tmp_path, monkeypatch, caplog):
        monkeypatch.delenv("ANIMUS_FORGE_SKIP_GITLEAKS", raising=False)
        monkeypatch.setattr("animus_forge.self_improve.pr_manager.shutil.which", lambda _: None)
        mgr = PRManager(repo_path=tmp_path)
        with caplog.at_level("WARNING"):
            mgr._gitleaks_scan_staged()
        assert any("gitleaks not found" in r.message for r in caplog.records)

    def test_clean_scan_returns_silently(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANIMUS_FORGE_SKIP_GITLEAKS", raising=False)
        monkeypatch.setattr(
            "animus_forge.self_improve.pr_manager.shutil.which",
            lambda _: "/usr/bin/gitleaks",
        )
        clean_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch(
            "animus_forge.self_improve.pr_manager.subprocess.run", return_value=clean_result
        ):
            mgr = PRManager(repo_path=tmp_path)
            mgr._gitleaks_scan_staged()  # no raise

    def test_dirty_scan_raises_secrets_detected_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANIMUS_FORGE_SKIP_GITLEAKS", raising=False)
        monkeypatch.setattr(
            "animus_forge.self_improve.pr_manager.shutil.which",
            lambda _: "/usr/bin/gitleaks",
        )
        dirty_result = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=(
                "Finding:     Anthropic key\n"
                "Secret:      sk-ant-redacted\n"
                "Finding:     OpenAI key\n"
                "Secret:      sk-redacted\n"
            ),
            stderr="",
        )
        with patch(
            "animus_forge.self_improve.pr_manager.subprocess.run", return_value=dirty_result
        ):
            mgr = PRManager(repo_path=tmp_path)
            with pytest.raises(SecretsDetectedError, match="found 2 secret"):
                mgr._gitleaks_scan_staged()

    def test_commit_changes_returns_none_when_scan_blocks(self, tmp_path, monkeypatch):
        """The headline adversarial test — commit_changes is the autonomous
        commit path. With staged secrets it must refuse rather than push."""
        monkeypatch.delenv("ANIMUS_FORGE_SKIP_GITLEAKS", raising=False)
        monkeypatch.setattr(
            "animus_forge.self_improve.pr_manager.shutil.which",
            lambda _: "/usr/bin/gitleaks",
        )
        mgr = PRManager(repo_path=tmp_path)

        with patch.object(mgr, "_run_git") as mock_run_git:
            mock_run_git.return_value = MagicMock(returncode=0, stdout="")
            dirty = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="Finding: bad\n", stderr=""
            )
            with patch(
                "animus_forge.self_improve.pr_manager.subprocess.run",
                return_value=dirty,
            ):
                commit_hash = mgr.commit_changes(["some_file.py"], "fix something")
        assert commit_hash is None
        # `git add` was called, `git commit` was NOT
        added = any(call.args[0][0] == "add" for call in mock_run_git.call_args_list)
        committed = any(call.args[0][0] == "commit" for call in mock_run_git.call_args_list)
        assert added is True
        assert committed is False


# ---------------------------------------------------------------------------
# 4.B — Allow-list enforcement
# ---------------------------------------------------------------------------


class TestSafetyAllowList:
    """allowed_files flips deny-list semantics to allow-list when non-empty."""

    def test_empty_allow_list_no_enforcement(self):
        """Backward compat — empty allow-list means no new gate."""
        cfg = SafetyConfig()  # allowed_files default = []
        checker = SafetyChecker(cfg)
        assert checker.is_allowed_file("anything.py") is True
        assert checker.is_allowed_file("random/path.txt") is True

    def test_allow_list_matches_pattern(self):
        cfg = SafetyConfig(allowed_files=["packages/*/src/**/*.py"])
        checker = SafetyChecker(cfg)
        assert checker.is_allowed_file("packages/core/src/animus/foo.py") is True

    def test_allow_list_blocks_non_matching(self):
        cfg = SafetyConfig(allowed_files=["packages/*/src/**/*.py"])
        checker = SafetyChecker(cfg)
        assert checker.is_allowed_file("README.md") is False
        assert checker.is_allowed_file("scripts/deploy.sh") is False

    def test_check_changes_flags_non_allow_listed_modify(self):
        cfg = SafetyConfig(allowed_files=["packages/*/src/**/*.py"])
        checker = SafetyChecker(cfg)
        violations = checker.check_changes(
            files_modified=["scripts/deploy.sh"],
            files_added=[],
            files_deleted=[],
            lines_changed=10,
        )
        types = {v.violation_type for v in violations}
        assert "not_allow_listed" in types

    def test_check_changes_flags_non_allow_listed_add(self):
        cfg = SafetyConfig(allowed_files=["packages/*/src/**/*.py"])
        checker = SafetyChecker(cfg)
        violations = checker.check_changes(
            files_modified=[],
            files_added=["scripts/new_thing.sh"],
            files_deleted=[],
            lines_changed=5,
        )
        types = {v.violation_type for v in violations}
        assert "not_allow_listed" in types

    def test_check_changes_allows_matching_file(self):
        cfg = SafetyConfig(allowed_files=["packages/*/src/**/*.py"])
        checker = SafetyChecker(cfg)
        violations = checker.check_changes(
            files_modified=["packages/forge/src/animus_forge/workflow.py"],
            files_added=[],
            files_deleted=[],
            lines_changed=5,
        )
        types = {v.violation_type for v in violations}
        assert "not_allow_listed" not in types

    def test_protected_file_still_blocks_under_allow_list(self):
        """A file matching the allow-list AND the deny-list is still blocked."""
        cfg = SafetyConfig(
            allowed_files=["packages/*/src/**/*.py"],
            critical_files=["packages/*/src/**/self_improve/**"],
        )
        checker = SafetyChecker(cfg)
        violations = checker.check_changes(
            files_modified=["packages/forge/src/animus_forge/self_improve/safety.py"],
            files_added=[],
            files_deleted=[],
            lines_changed=5,
        )
        types = {v.violation_type for v in violations}
        # Both violations should fire — protected_file is the load-bearing
        # one; not_allow_listed doesn't apply here because the file DOES
        # match the allow pattern.
        assert "protected_file" in types
        assert "not_allow_listed" not in types

    def test_real_yaml_has_allow_list_configured(self):
        """Sanity check the shipped YAML actually defines the allow-list."""
        yaml_path = Path(__file__).resolve().parent.parent / "config" / "self_improve_safety.yaml"
        config = SafetyConfig.load(yaml_path)
        assert config.allowed_files, "self_improve_safety.yaml must ship an allow-list"
        # Every entry should look like a glob (contains '*' or '/')
        for pattern in config.allowed_files:
            assert "*" in pattern or "/" in pattern


# ---------------------------------------------------------------------------
# 4.C — origin/HEAD whip-saw fix
# ---------------------------------------------------------------------------


class TestDefaultBranchResolution:
    """checkout_main uses the configured default_branch, not origin/HEAD."""

    def test_constructor_arg_takes_precedence(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANIMUS_FORGE_DEFAULT_BRANCH", "from-env")
        mgr = PRManager(repo_path=tmp_path, default_branch="from-arg")
        assert mgr.default_branch == "from-arg"

    def test_env_var_used_when_no_arg(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANIMUS_FORGE_DEFAULT_BRANCH", "main-from-env")
        mgr = PRManager(repo_path=tmp_path)
        assert mgr.default_branch == "main-from-env"

    def test_default_branch_unset_when_neither_provided(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANIMUS_FORGE_DEFAULT_BRANCH", raising=False)
        mgr = PRManager(repo_path=tmp_path)
        assert mgr.default_branch is None

    def test_checkout_main_uses_configured_branch(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANIMUS_FORGE_DEFAULT_BRANCH", raising=False)
        mgr = PRManager(repo_path=tmp_path, default_branch="main")
        with patch.object(mgr, "_run_git") as mock_run_git:
            mgr.checkout_main()
        # Only one call — direct checkout, no symbolic-ref lookup
        calls = [c.args[0] for c in mock_run_git.call_args_list]
        assert ["checkout", "main"] in calls
        assert all("symbolic-ref" not in c for c in calls)

    def test_checkout_main_uses_env_branch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANIMUS_FORGE_DEFAULT_BRANCH", "trunk")
        mgr = PRManager(repo_path=tmp_path)
        with patch.object(mgr, "_run_git") as mock_run_git:
            mgr.checkout_main()
        calls = [c.args[0] for c in mock_run_git.call_args_list]
        assert ["checkout", "trunk"] in calls

    def test_strict_mode_raises_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANIMUS_FORGE_DEFAULT_BRANCH", raising=False)
        monkeypatch.setenv("ANIMUS_FORGE_REQUIRE_DEFAULT_BRANCH", "1")
        mgr = PRManager(repo_path=tmp_path)
        with patch.object(mgr, "_run_git") as mock_run_git:
            result = mgr.checkout_main()
        # checkout_main catches and logs, returning False
        assert result is False
        # No git calls were made — guard fired before any subprocess
        assert mock_run_git.call_count == 0

    def test_fallback_to_symbolic_ref_logs_warning(self, tmp_path, monkeypatch, caplog):
        monkeypatch.delenv("ANIMUS_FORGE_DEFAULT_BRANCH", raising=False)
        monkeypatch.delenv("ANIMUS_FORGE_REQUIRE_DEFAULT_BRANCH", raising=False)
        mgr = PRManager(repo_path=tmp_path)
        with patch.object(mgr, "_run_git") as mock_run_git:
            mock_run_git.return_value = MagicMock(stdout="origin/legacy-main\n")
            with caplog.at_level("WARNING"):
                mgr.checkout_main()
        assert any("whip-saw" in r.message for r in caplog.records)
        # Symbolic-ref lookup happened in fallback
        calls = [c.args[0] for c in mock_run_git.call_args_list]
        assert any("symbolic-ref" in c for c in calls)
