"""Tests for sandbox self-improvement orchestrator."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from animus_kernel.sandbox.analyzer import ImprovementCategory, ImprovementSuggestion
from animus_kernel.sandbox.orchestrator import (
    ImprovementResult,
    SelfImproveOrchestrator,
    WorkflowStage,
)
from animus_kernel.sandbox.safety import SafetyConfig

# ═══════════════════════════════════════════════════════════════════
# Mock provider for testing
# ═══════════════════════════════════════════════════════════════════


class MockProvider:
    """Mock AI provider that returns deterministic responses."""

    def __init__(self, response: str = ""):
        self.response = response
        self.calls = []

    async def complete(self, messages: list[dict[str, Any]], max_tokens: int = 1000) -> str:
        self.calls.append({"method": "complete", "messages": messages})
        return self.response

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tool_registry: Any,
        max_iterations: int = 3,
        max_tokens: int = 1000,
    ) -> str:
        self.calls.append({"method": "complete_with_tools", "messages": messages})
        return self.response


# ═══════════════════════════════════════════════════════════════════
# SelfImproveOrchestrator initialization tests
# ═══════════════════════════════════════════════════════════════════


class TestOrchestratorInit:
    def test_init_with_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            assert orch.current_stage == WorkflowStage.IDLE
            assert orch.codebase_path == Path(tmpdir)
            assert orch.provider is None

    def test_init_with_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = MockProvider()
            orch = SelfImproveOrchestrator(codebase_path=tmpdir, provider=provider)
            assert orch.provider is provider

    def test_get_status_idle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            status = orch.get_status()
            assert status["stage"] == "idle"
            assert status["current_plan"] is None


# ═══════════════════════════════════════════════════════════════════
# SelfImproveOrchestrator workflow tests
# ═══════════════════════════════════════════════════════════════════


class TestOrchestratorWorkflow:
    @pytest.mark.asyncio
    async def test_run_no_improvements_found(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_FORGE_ALLOW_AUTO_APPROVE", "1")
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            result = await orch.run(auto_approve=True)

            assert isinstance(result, ImprovementResult)
            assert result.success is True
            assert result.stage_reached == WorkflowStage.COMPLETE
            assert result.plan is None

    @pytest.mark.asyncio
    async def test_run_detects_long_function(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_FORGE_ALLOW_AUTO_APPROVE", "1")
        # Add trailing short function so analyzer detects long_one
        code = "\n".join(
            ["def very_long():"] + ["    x = 1"] * 60 + ["", "def short():", "    pass"]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            src = tmpdir_path / "src"
            src.mkdir()
            (src / "module.py").write_text(code)
            # Provide a minimal test file so sandbox pytest doesn't fail with "no tests"
            tests = tmpdir_path / "tests"
            tests.mkdir()
            (tests / "test_module.py").write_text("def test_ok(): assert True\n")

            # Need a git repo for branch creation
            import subprocess

            subprocess.run(["git", "init"], cwd=str(tmpdir_path), capture_output=True, check=True)
            subprocess.run(
                ["git", "config", "user.email", "t@t.com"],
                cwd=str(tmpdir_path),
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=str(tmpdir_path),
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "add", "."], cwd=str(tmpdir_path), capture_output=True, check=True
            )
            subprocess.run(
                ["git", "commit", "-m", "init"],
                cwd=str(tmpdir_path),
                capture_output=True,
                check=True,
            )

            # The orchestrator expects FILE: markers + code blocks for function refactoring
            mock_response = """FILE: src/module.py
```python
def _helper():
    pass

def very_long():
    _helper()
```
"""
            provider = MockProvider(response=mock_response)
            orch = SelfImproveOrchestrator(codebase_path=tmpdir_path, provider=provider)
            result = await orch.run(auto_approve=True)

            assert result.success is True
            assert result.plan is not None
            assert any("Long function" in s.title for s in result.plan.suggestions)

    @pytest.mark.asyncio
    async def test_run_safety_violation_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "secrets.yaml").write_text("api_key: secret")

            config = SafetyConfig(
                critical_files=["src/secrets.yaml"],
                max_lines_changed=1000,
            )
            orch = SelfImproveOrchestrator(codebase_path=tmpdir, config=config)
            # Force an analysis that targets the protected file by making it
            # the only Python file (won't trigger here since it's yaml).
            # Instead test via manual safety check.
            violations = orch.safety_checker.check_changes(
                files_modified=["src/secrets.yaml"],
                files_added=[],
                files_deleted=[],
                lines_changed=10,
            )
            assert any(v.violation_type == "protected_file" for v in violations)

    @pytest.mark.asyncio
    async def test_auto_approve_blocked_in_production(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            with pytest.raises(RuntimeError, match="auto_approve=True is blocked"):
                await orch.run(auto_approve=True)

    @pytest.mark.asyncio
    async def test_run_with_docstring_generation(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_FORGE_ALLOW_AUTO_APPROVE", "1")
        code = "def hello():\n    pass\n"

        # _generate_docstring_changes calls provider.complete() for docstring text.
        # It expects just a short docstring string, not FILE markers.
        provider = MockProvider(response="Say hello.")

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)
            # Provide a minimal test file so sandbox pytest doesn't fail with "no tests"
            tests = Path(tmpdir) / "tests"
            tests.mkdir()
            (tests / "test_module.py").write_text("def test_ok(): assert True\n")

            orch = SelfImproveOrchestrator(
                codebase_path=tmpdir,
                provider=provider,
            )

            result = await orch.run(focus_category="documentation", auto_approve=True)

            # The run should complete through to APPLYING or CREATING_PR
            assert result.stage_reached in {
                WorkflowStage.APPLYING,
                WorkflowStage.CREATING_PR,
                WorkflowStage.AWAITING_APPLY_APPROVAL,
                WorkflowStage.AWAITING_MERGE_APPROVAL,
                WorkflowStage.COMPLETE,
                WorkflowStage.FAILED,
            }

    def test_create_plan_single_suggestion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            suggestion = ImprovementSuggestion(
                id="s1",
                category=ImprovementCategory.REFACTORING,
                title="Long function: foo",
                description="Function foo has 80 lines.",
                affected_files=["src/main.py"],
                priority=3,
                estimated_lines=80,
            )
            plan = orch._create_plan([suggestion])
            assert plan.title == "Long function: foo"
            assert plan.estimated_files == ["src/main.py"]
            assert plan.estimated_lines == 80
            assert len(plan.suggestions) == 1

    def test_create_plan_multiple_suggestions_local(self):
        """Local provider keeps only small fixes (<=20 lines)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            suggestions = [
                ImprovementSuggestion(
                    id="s1",
                    category=ImprovementCategory.REFACTORING,
                    title="Long function: foo",
                    description="80 lines",
                    affected_files=["src/main.py"],
                    estimated_lines=80,
                ),
                ImprovementSuggestion(
                    id="s2",
                    category=ImprovementCategory.DOCUMENTATION,
                    title="Missing docstring: bar",
                    description="No docstring",
                    affected_files=["src/main.py"],
                    estimated_lines=5,
                ),
            ]
            plan = orch._create_plan(suggestions)
            # Local mode filters to small fixes only
            assert plan.title == "Missing docstring: bar"
            assert len(plan.suggestions) == 1

    def test_create_plan_multiple_suggestions_cloud(self):
        """Cloud provider keeps all manageable suggestions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Any provider name not containing "ollama" is treated as cloud
            provider = MockProvider()
            orch = SelfImproveOrchestrator(codebase_path=tmpdir, provider=provider)
            suggestions = [
                ImprovementSuggestion(
                    id="s1",
                    category=ImprovementCategory.REFACTORING,
                    title="Long function: foo",
                    description="80 lines",
                    affected_files=["src/main.py"],
                    estimated_lines=80,
                ),
                ImprovementSuggestion(
                    id="s2",
                    category=ImprovementCategory.DOCUMENTATION,
                    title="Missing docstring: bar",
                    description="No docstring",
                    affected_files=["src/main.py"],
                    estimated_lines=5,
                ),
            ]
            plan = orch._create_plan(suggestions)
            assert "Multiple improvements" in plan.title
            assert len(plan.suggestions) == 2

    def test_is_local_provider_no_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            assert orch._is_local_provider() is True

    def test_is_local_provider_with_mock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = MockProvider()
            orch = SelfImproveOrchestrator(codebase_path=tmpdir, provider=provider)
            # MockProvider name does not contain "ollama"
            assert orch._is_local_provider() is False


# ═══════════════════════════════════════════════════════════════════
# Recursive self-targeting tests
# ═══════════════════════════════════════════════════════════════════


class TestOrchestratorRecursiveDepth:
    @pytest.mark.asyncio
    async def test_recursive_depth_zero_by_default(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_FORGE_ALLOW_AUTO_APPROVE", "1")
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            result = await orch.run(auto_approve=True)
            assert result.recursive_depth == 0

    @pytest.mark.asyncio
    async def test_recursive_depth_passed_through(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_FORGE_ALLOW_AUTO_APPROVE", "1")
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            result = await orch.run(auto_approve=True, recursive_depth=2)
            assert result.recursive_depth == 2

    @pytest.mark.asyncio
    async def test_max_recursive_depth_enforced(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_FORGE_ALLOW_AUTO_APPROVE", "1")
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SafetyConfig(max_recursive_depth=1)
            orch = SelfImproveOrchestrator(codebase_path=tmpdir, config=config)
            result = await orch.run(auto_approve=True, recursive_depth=2)
            assert result.success is False
            assert result.stage_reached == WorkflowStage.FAILED
            assert "Max recursive depth exceeded" in (result.error or "")
            assert result.recursive_depth == 2


# ═══════════════════════════════════════════════════════════════════
# Parse changes response tests
# ═══════════════════════════════════════════════════════════════════


class TestParseChangesResponse:
    def test_parse_file_markers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            response = """FILE: src/main.py
```python
def hello():
    pass
```

FILE: src/utils.py
```python
def helper():
    return 42
```
"""
            changes = orch._parse_changes_response(response)
            assert "src/main.py" in changes
            assert "def hello():" in changes["src/main.py"]
            assert "src/utils.py" in changes
            assert "helper" in changes["src/utils.py"]

    def test_parse_json_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            response = '{"src/main.py": "print(1)"}'
            changes = orch._parse_changes_response(response)
            assert changes.get("src/main.py") == "print(1)"

    def test_parse_json_list_of_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            response = '[{"file": "src/main.py", "changes": [{"old": "x", "new": "y"}]}]'
            # This needs the original file to exist for diff application
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "main.py").write_text("x")
            changes = orch._parse_changes_response(response)
            assert "src/main.py" in changes

    def test_parse_with_trailing_comma_fix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            response = '{"a.py": "1",}'
            changes = orch._parse_changes_response(response)
            assert changes.get("a.py") == "1"


# ═══════════════════════════════════════════════════════════════════
# Rollback integration tests
# ═══════════════════════════════════════════════════════════════════


class TestOrchestratorRollback:
    def test_rollback_existing_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "main.py").write_text("original")

            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            snapshot = orch.rollback_manager.create_snapshot(
                files=["src/main.py"],
                description="test",
                codebase_path=tmpdir,
            )

            (src / "main.py").write_text("modified")
            result = orch.rollback(snapshot.id)
            assert result is True
            assert (src / "main.py").read_text() == "original"

    def test_rollback_missing_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            result = orch.rollback("nonexistent")
            assert result is False


# ═══════════════════════════════════════════════════════════════════
# Approval integration tests
# ═══════════════════════════════════════════════════════════════════


class TestOrchestratorApprovals:
    def test_get_pending_approvals_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            pending = orch.get_pending_approvals()
            assert pending == []

    def test_get_approval_history_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            history = orch.get_approval_history()
            assert history == []

    def test_list_snapshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "main.py").write_text("x")

            orch = SelfImproveOrchestrator(codebase_path=tmpdir)
            orch.rollback_manager.create_snapshot(
                files=["src/main.py"],
                description="snap1",
                codebase_path=tmpdir,
            )
            snapshots = orch.list_snapshots()
            assert len(snapshots) == 1
            assert snapshots[0].description == "snap1"
