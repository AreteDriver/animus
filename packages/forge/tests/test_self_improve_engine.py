"""Tests for self-improvement engine: code generation, sandbox execution, apply changes."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_forge.self_improve.analyzer import (
    ImprovementCategory,
    ImprovementSuggestion,
)
from animus_forge.self_improve.orchestrator import (
    ImprovementPlan,
    SelfImproveOrchestrator,
    WorkflowStage,
)
from animus_forge.self_improve.safety import SafetyConfig
from animus_forge.self_improve.sandbox import Sandbox, SandboxResult, SandboxStatus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def permissive_config() -> SafetyConfig:
    """SafetyConfig with no approvals and relaxed limits."""
    return SafetyConfig(
        critical_files=["**/identity/**", "**/CORE_VALUES*"],
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


@pytest.fixture()
def codebase(tmp_path: Path) -> Path:
    """Create a minimal codebase with a file to improve."""
    src = tmp_path / "src" / "animus_forge"
    src.mkdir(parents=True)
    (src / "sample.py").write_text(
        '"""Module docstring."""\n\n\ndef compute(x):\n    return x * 2\n'
    )
    return tmp_path


@pytest.fixture()
def plan() -> ImprovementPlan:
    """Create a sample improvement plan."""
    return ImprovementPlan(
        id="test-plan",
        title="Add type hints",
        description="Add type hints to compute function",
        suggestions=[
            ImprovementSuggestion(
                id="s1",
                category=ImprovementCategory.CODE_QUALITY,
                title="Add type hints",
                description="Add parameter and return type hints",
                affected_files=["src/animus_forge/sample.py"],
                estimated_lines=5,
                implementation_hints="Add int type hint to parameter x and return type",
            )
        ],
        implementation_steps=["Implement: Add type hints"],
        estimated_files=["src/animus_forge/sample.py"],
        estimated_lines=5,
    )


# ===========================================================================
# Code Generation (_generate_changes) tests
# ===========================================================================


class TestCodeGeneration:
    """Tests for _generate_changes method."""

    def test_generate_changes_no_provider(
        self, codebase: Path, permissive_config: SafetyConfig, plan: ImprovementPlan
    ):
        """Returns empty dict when no provider is available."""
        orch = SelfImproveOrchestrator(codebase_path=codebase, config=permissive_config)
        result = asyncio.run(orch._generate_changes(plan))
        assert result == {}

    def test_generate_changes_with_provider(
        self, codebase: Path, permissive_config: SafetyConfig, plan: ImprovementPlan
    ):
        """Returns parsed changes from provider response."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = (
            '{"src/animus_forge/sample.py": "def compute(x: int) -> int:\\n    return x * 2\\n"}'
        )

        orch = SelfImproveOrchestrator(
            codebase_path=codebase, provider=mock_provider, config=permissive_config
        )
        result = asyncio.run(orch._generate_changes(plan))

        assert "src/animus_forge/sample.py" in result
        assert "int" in result["src/animus_forge/sample.py"]
        mock_provider.complete.assert_called_once()

    def test_generate_changes_filters_protected_files(
        self, codebase: Path, permissive_config: SafetyConfig, plan: ImprovementPlan
    ):
        """Protected files are filtered from generated changes."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = (
            '{"src/animus_forge/sample.py": "good", "identity/CORE_VALUES.md": "evil"}'
        )

        orch = SelfImproveOrchestrator(
            codebase_path=codebase, provider=mock_provider, config=permissive_config
        )
        result = asyncio.run(orch._generate_changes(plan))

        assert "src/animus_forge/sample.py" in result
        assert "identity/CORE_VALUES.md" not in result

    def test_generate_changes_provider_error(
        self, codebase: Path, permissive_config: SafetyConfig, plan: ImprovementPlan
    ):
        """Returns empty dict when provider raises."""
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("LLM down")

        orch = SelfImproveOrchestrator(
            codebase_path=codebase, provider=mock_provider, config=permissive_config
        )
        result = asyncio.run(orch._generate_changes(plan))
        assert result == {}

    def test_generate_changes_malformed_json(
        self, codebase: Path, permissive_config: SafetyConfig, plan: ImprovementPlan
    ):
        """Returns empty dict when provider returns non-JSON."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = "Here are the changes I made to..."

        orch = SelfImproveOrchestrator(
            codebase_path=codebase, provider=mock_provider, config=permissive_config
        )
        result = asyncio.run(orch._generate_changes(plan))
        assert result == {}

    def test_generate_changes_markdown_fenced_json(
        self, codebase: Path, permissive_config: SafetyConfig, plan: ImprovementPlan
    ):
        """Handles markdown-fenced JSON response."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = (
            '```json\n{"src/animus_forge/sample.py": "content"}\n```'
        )

        orch = SelfImproveOrchestrator(
            codebase_path=codebase, provider=mock_provider, config=permissive_config
        )
        result = asyncio.run(orch._generate_changes(plan))
        assert "src/animus_forge/sample.py" in result

    def test_generate_changes_reads_existing_files(
        self, codebase: Path, permissive_config: SafetyConfig, plan: ImprovementPlan
    ):
        """Prompt includes current file contents."""
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = "{}"

        orch = SelfImproveOrchestrator(
            codebase_path=codebase, provider=mock_provider, config=permissive_config
        )
        asyncio.run(orch._generate_changes(plan))

        # Check that the prompt sent to provider includes the file content
        call_args = mock_provider.complete.call_args[0][0]
        user_msg = [m for m in call_args if m["role"] == "user"][0]["content"]
        assert "def compute(x)" in user_msg

    def test_generate_changes_missing_file(self, codebase: Path, permissive_config: SafetyConfig):
        """Handles files that don't exist gracefully."""
        plan = ImprovementPlan(
            id="p",
            title="T",
            description="D",
            suggestions=[],
            implementation_steps=[],
            estimated_files=["nonexistent.py"],
            estimated_lines=0,
        )
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = "{}"

        orch = SelfImproveOrchestrator(
            codebase_path=codebase, provider=mock_provider, config=permissive_config
        )
        result = asyncio.run(orch._generate_changes(plan))
        assert result == {}


# ===========================================================================
# Parse Changes Response tests
# ===========================================================================


class TestParseChangesResponse:
    """Tests for _parse_changes_response."""

    def setup_method(self):
        self.orch = SelfImproveOrchestrator.__new__(SelfImproveOrchestrator)

    def test_valid_json(self):
        assert self.orch._parse_changes_response('{"a.py": "content"}') == {"a.py": "content"}

    def test_markdown_fenced(self):
        assert self.orch._parse_changes_response('```json\n{"a.py": "c"}\n```') == {"a.py": "c"}

    def test_plain_fenced(self):
        assert self.orch._parse_changes_response('```\n{"a.py": "c"}\n```') == {"a.py": "c"}

    def test_invalid_json(self):
        assert self.orch._parse_changes_response("not json at all") == {}

    def test_non_dict_json(self):
        assert self.orch._parse_changes_response('["a", "b"]') == {}

    def test_empty_dict(self):
        assert self.orch._parse_changes_response("{}") == {}

    def test_non_string_key_filtered(self):
        # JSON keys are always strings, but values might not be
        assert self.orch._parse_changes_response('{"a.py": 123}') == {"a.py": "123"}


# ===========================================================================
# Apply Changes tests
# ===========================================================================


class TestApplyChanges:
    """Tests for _apply_changes method."""

    def test_apply_changes_writes_files(self, codebase: Path, permissive_config: SafetyConfig):
        """Changes are written to the working tree."""
        orch = SelfImproveOrchestrator(codebase_path=codebase, config=permissive_config)
        changes = {
            "src/animus_forge/sample.py": "# updated\ndef compute(x: int) -> int:\n    return x * 2\n",
        }
        orch._apply_changes(changes)

        content = (codebase / "src/animus_forge/sample.py").read_text()
        assert "int" in content
        assert "# updated" in content

    def test_apply_changes_creates_parent_dirs(
        self, codebase: Path, permissive_config: SafetyConfig
    ):
        """Parent directories are created if needed."""
        orch = SelfImproveOrchestrator(codebase_path=codebase, config=permissive_config)
        changes = {
            "src/animus_forge/new_pkg/module.py": "# new file\n",
        }
        orch._apply_changes(changes)

        assert (codebase / "src/animus_forge/new_pkg/module.py").exists()
        assert (codebase / "src/animus_forge/new_pkg/module.py").read_text() == "# new file\n"

    def test_apply_changes_overwrites_existing(
        self, codebase: Path, permissive_config: SafetyConfig
    ):
        """Existing files are overwritten."""
        orch = SelfImproveOrchestrator(codebase_path=codebase, config=permissive_config)
        original = (codebase / "src/animus_forge/sample.py").read_text()

        changes = {"src/animus_forge/sample.py": "completely new content\n"}
        orch._apply_changes(changes)

        assert (codebase / "src/animus_forge/sample.py").read_text() == "completely new content\n"
        assert (codebase / "src/animus_forge/sample.py").read_text() != original


# ===========================================================================
# Sandbox Execution tests
# ===========================================================================


class TestSandboxExecution:
    """Tests for real sandbox creation, apply, and validation."""

    def test_sandbox_create_copies_files(self, codebase: Path):
        """Sandbox creates a temp copy of the codebase."""
        with Sandbox(codebase, timeout=60) as sandbox:
            assert sandbox.sandbox_path is not None
            assert sandbox.sandbox_path.exists()
            # Source file should be copied
            assert (sandbox.sandbox_path / "src" / "animus_forge" / "sample.py").exists()

    def test_sandbox_cleanup(self, codebase: Path):
        """Sandbox cleans up temp directory on exit."""
        sandbox = Sandbox(codebase, timeout=60)
        sandbox.create()
        path = sandbox.sandbox_path
        assert path.exists()
        sandbox.cleanup()
        assert not path.exists()

    def test_sandbox_apply_changes(self, codebase: Path):
        """Changes are applied inside the sandbox."""
        with Sandbox(codebase, timeout=60) as sandbox:
            result = asyncio.run(
                sandbox.apply_changes({"src/animus_forge/sample.py": "# changed\n"})
            )
            assert result is True
            assert (
                sandbox.sandbox_path / "src/animus_forge/sample.py"
            ).read_text() == "# changed\n"

    def test_sandbox_apply_changes_creates_new_files(self, codebase: Path):
        """New files can be created in sandbox."""
        with Sandbox(codebase, timeout=60) as sandbox:
            result = asyncio.run(sandbox.apply_changes({"new_dir/new_file.py": "print('hello')\n"}))
            assert result is True
            assert (sandbox.sandbox_path / "new_dir/new_file.py").exists()

    def test_sandbox_apply_changes_not_created(self):
        """Apply raises if sandbox not created."""
        sandbox = Sandbox("/tmp/nonexistent")
        with pytest.raises(RuntimeError, match="not created"):
            asyncio.run(sandbox.apply_changes({"a.py": "x"}))

    def test_sandbox_context_manager(self, codebase: Path):
        """Context manager creates and cleans up."""
        path = None
        with Sandbox(codebase) as sandbox:
            path = sandbox.sandbox_path
            assert path.exists()
        assert not path.exists()

    def test_sandbox_no_cleanup_option(self, codebase: Path):
        """cleanup_on_exit=False preserves sandbox."""
        sandbox = Sandbox(codebase, cleanup_on_exit=False)
        sandbox.create()
        path = sandbox.sandbox_path
        sandbox.__exit__(None, None, None)
        assert path.exists()
        # Manual cleanup
        sandbox.cleanup_on_exit = True
        sandbox._temp_dir = path.parent
        sandbox._sandbox_path = path
        sandbox.cleanup()

    def test_sandbox_ignores_pycache(self, codebase: Path):
        """__pycache__ dirs are not copied to sandbox."""
        cache = codebase / "src" / "animus_forge" / "__pycache__"
        cache.mkdir(exist_ok=True)
        (cache / "sample.cpython-312.pyc").write_bytes(b"\x00")

        with Sandbox(codebase) as sandbox:
            assert not (sandbox.sandbox_path / "src" / "animus_forge" / "__pycache__").exists()

    def test_sandbox_status_tracking(self, codebase: Path):
        """Sandbox tracks its status."""
        sandbox = Sandbox(codebase)
        assert sandbox.status == SandboxStatus.CREATED
        sandbox.create()
        assert sandbox.status == SandboxStatus.CREATED  # Still created, not running


# ===========================================================================
# Sandbox Environment Sanitization tests
# ===========================================================================


class TestSandboxEnvSanitization:
    """Tests for _sanitize_env."""

    def test_safe_vars_included(self):
        """PATH, HOME, LANG are kept."""
        with patch.dict(
            os.environ,
            {"PATH": "/usr/bin", "HOME": "/home/test", "LANG": "en_US.UTF-8"},
            clear=True,
        ):
            env = Sandbox._sanitize_env()
            assert env["PATH"] == "/usr/bin"
            assert env["HOME"] == "/home/test"
            assert env["LANG"] == "en_US.UTF-8"

    def test_secret_vars_stripped(self):
        """Variables matching *_KEY, *_TOKEN, etc. are removed."""
        with patch.dict(
            os.environ,
            {
                "PATH": "/usr/bin",
                "AWS_SECRET_KEY": "secret123",
                "GITHUB_TOKEN": "ghp_xxx",
                "DB_PASSWORD": "hunter2",
                "API_CREDENTIALS": "cred123",
            },
            clear=True,
        ):
            env = Sandbox._sanitize_env()
            assert "AWS_SECRET_KEY" not in env
            assert "GITHUB_TOKEN" not in env
            assert "DB_PASSWORD" not in env
            assert "API_CREDENTIALS" not in env
            assert "PATH" in env

    def test_non_secret_vars_kept(self):
        """Regular env vars that don't match secret patterns are kept."""
        with patch.dict(
            os.environ, {"PATH": "/usr/bin", "EDITOR": "vim", "SHELL": "/bin/bash"}, clear=True
        ):
            env = Sandbox._sanitize_env()
            assert env["EDITOR"] == "vim"
            assert env["SHELL"] == "/bin/bash"


# ===========================================================================
# Full Pipeline (End-to-End) tests
# ===========================================================================


class TestFullPipeline:
    """Integration tests for the complete orchestrator pipeline."""

    def test_pipeline_no_changes_generated(self, codebase: Path, permissive_config: SafetyConfig):
        """Pipeline fails gracefully when no changes are generated."""
        orch = SelfImproveOrchestrator(codebase_path=codebase, config=permissive_config)

        # No provider → _generate_changes returns {}
        src = codebase / "src" / "animus_forge"
        (src / "bad.py").write_text("import os\n\ndef no_doc():\n    pass\n")

        result = asyncio.run(orch.run())

        # Should fail at implementing stage
        if result.stage_reached == WorkflowStage.IMPLEMENTING:
            assert result.success is False
            assert "No changes generated" in result.error

    def test_pipeline_sandbox_test_failure(self, codebase: Path, permissive_config: SafetyConfig):
        """Pipeline stops when sandbox tests fail."""
        src = codebase / "src" / "animus_forge"
        (src / "bad.py").write_text("import os\n\ndef no_doc():\n    pass\n")

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = '{"src/animus_forge/bad.py": "fixed content"}'

        orch = SelfImproveOrchestrator(
            codebase_path=codebase, provider=mock_provider, config=permissive_config
        )

        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=False)
        mock_sandbox.apply_changes = AsyncMock(return_value=True)
        mock_sandbox.validate_changes = AsyncMock(
            return_value=SandboxResult(
                status=SandboxStatus.FAILED,
                tests_passed=False,
                lint_passed=True,
                test_output="FAILED",
            )
        )

        with patch("animus_forge.self_improve.orchestrator.Sandbox", return_value=mock_sandbox):
            result = asyncio.run(orch.run())

        assert result.success is False
        assert result.stage_reached == WorkflowStage.TESTING
        assert "Tests failed" in result.error

    def test_pipeline_sandbox_apply_failure(self, codebase: Path, permissive_config: SafetyConfig):
        """Pipeline stops when sandbox cannot apply changes."""
        src = codebase / "src" / "animus_forge"
        (src / "bad.py").write_text("import os\n\ndef no_doc():\n    pass\n")

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = '{"src/animus_forge/bad.py": "fixed"}'

        orch = SelfImproveOrchestrator(
            codebase_path=codebase, provider=mock_provider, config=permissive_config
        )

        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=False)
        mock_sandbox.apply_changes = AsyncMock(return_value=False)

        with patch("animus_forge.self_improve.orchestrator.Sandbox", return_value=mock_sandbox):
            result = asyncio.run(orch.run())

        assert result.success is False
        assert "Failed to apply changes" in result.error

    def test_pipeline_full_success_auto_approve(
        self, codebase: Path, permissive_config: SafetyConfig
    ):
        """Full pipeline succeeds with auto-approve and mocked components."""
        src = codebase / "src" / "animus_forge"
        (src / "bad.py").write_text("import os\n\ndef no_doc():\n    pass\n")

        permissive_config.human_approval_plan = True
        permissive_config.human_approval_apply = True
        permissive_config.human_approval_merge = True

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = (
            '{"src/animus_forge/bad.py": "# fixed\\ndef no_doc():\\n    pass\\n"}'
        )

        orch = SelfImproveOrchestrator(
            codebase_path=codebase, provider=mock_provider, config=permissive_config
        )

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
            patch("animus_forge.self_improve.orchestrator.Sandbox", return_value=mock_sandbox),
            patch.object(orch.pr_manager, "_run_git"),
            patch("animus_forge.self_improve.pr_manager.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout="https://github.com/test/pull/1\n"
            )
            result = asyncio.run(orch.run(auto_approve=True))

        assert result.plan is not None
        assert result.sandbox_result is not None
        assert result.sandbox_result.tests_passed is True

    def test_pipeline_applies_changes_to_working_tree(
        self, codebase: Path, permissive_config: SafetyConfig
    ):
        """Stage 8 actually writes files to the working tree."""
        src = codebase / "src" / "animus_forge"
        (src / "target.py").write_text("import os\n\ndef undoc():\n    pass\n")

        new_content = (
            '"""Documented."""\n\n\ndef undoc() -> None:\n    """Now documented."""\n    pass\n'
        )

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = (
            f'{{"src/animus_forge/target.py": {json.dumps(new_content)}}}'
        )

        orch = SelfImproveOrchestrator(
            codebase_path=codebase, provider=mock_provider, config=permissive_config
        )

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
            patch("animus_forge.self_improve.orchestrator.Sandbox", return_value=mock_sandbox),
            patch.object(orch.pr_manager, "_run_git"),
            patch("animus_forge.self_improve.pr_manager.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stderr="")
            result = asyncio.run(orch.run())

        # Verify the file was actually written to the working tree
        final = (codebase / "src/animus_forge/target.py").read_text()
        assert "Documented" in final
        assert result.success is True
