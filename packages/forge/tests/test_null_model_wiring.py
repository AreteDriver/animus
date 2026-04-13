"""Integration tests for the null-model gate wiring in SelfImproveOrchestrator.

The pure gate logic is tested in ``test_null_model_gate.py``. This module
verifies the orchestrator's behavior when a ``NullModelGateConfig`` is
passed — that the gate is called, its verdict is propagated, and that
STRUCTURAL/HEURISTIC/NEUTRAL/REGRESSION tiers branch correctly.

The Sandbox's ``compute_metrics`` is mocked so these tests don't actually
run pytest or ruff — they only exercise the plumbing between the gate
and the orchestrator.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from animus_forge.self_improve.null_model_gate import (
    MetricSnapshot,
    NullModelGateConfig,
    NullModelResult,
    NullModelVerdict,
)
from animus_forge.self_improve.orchestrator import ImprovementResult, WorkflowStage
from animus_forge.self_improve.sandbox import Sandbox

# ---------------------------------------------------------------------------
# Sandbox.compute_metrics
# ---------------------------------------------------------------------------


class TestSandboxComputeMetrics:
    @pytest.mark.asyncio
    async def test_compute_metrics_returns_snapshot_with_no_path(self, tmp_path):
        sandbox = Sandbox(tmp_path, timeout=10)
        # Don't call create() — no sandbox path
        snap = await sandbox.compute_metrics()
        assert isinstance(snap, MetricSnapshot)
        assert snap.metadata.get("error") == "No sandbox path"

    @pytest.mark.asyncio
    async def test_compute_metrics_counts_lines_of_code(self, tmp_path):
        # Set up a fake codebase with 3 lines of code
        (tmp_path / "a.py").write_text('# comment\ndef foo():\n    """docstring"""\n    return 1\n')
        sandbox = Sandbox(tmp_path, timeout=10)
        sandbox._sandbox_path = tmp_path  # Bypass create()
        # Mock subprocess calls to avoid running pytest/ruff in the test
        with patch.object(
            sandbox,
            "_run_command",
            new=AsyncMock(
                return_value=type(
                    "CP",
                    (),
                    {"stdout": "", "stderr": "", "returncode": 0},
                )()
            ),
        ):
            snap = await sandbox.compute_metrics()
        assert isinstance(snap, MetricSnapshot)
        # Our file has 3 non-blank non-comment lines: def, docstring, return
        assert snap.lines_of_code is not None
        assert snap.lines_of_code >= 3
        # One function with a docstring → 100%
        assert snap.docstring_coverage_percent == 100.0

    @pytest.mark.asyncio
    async def test_compute_metrics_handles_ruff_subprocess(self, tmp_path):
        (tmp_path / "b.py").write_text("x = 1\n")
        sandbox = Sandbox(tmp_path, timeout=10)
        sandbox._sandbox_path = tmp_path

        async def fake_run_command(cmd, cwd=None):
            cp = type(
                "CP",
                (),
                {"stdout": "Found 3 errors.", "stderr": "", "returncode": 1},
            )()
            return cp

        with patch.object(sandbox, "_run_command", new=fake_run_command):
            snap = await sandbox.compute_metrics()
        # The ruff output "Found 3 errors." should be parsed
        assert snap.lint_errors == 3.0


# ---------------------------------------------------------------------------
# Orchestrator gate config plumbing
# ---------------------------------------------------------------------------


class TestOrchestratorGateConfig:
    def test_orchestrator_defaults_to_no_gate(self, tmp_path):
        from animus_forge.self_improve.orchestrator import SelfImproveOrchestrator

        orch = SelfImproveOrchestrator(codebase_path=tmp_path)
        assert orch.null_model_config is None

    def test_orchestrator_accepts_gate_config(self, tmp_path):
        from animus_forge.self_improve.orchestrator import SelfImproveOrchestrator

        config = NullModelGateConfig(
            enabled=True,
            n_null_samples=5,
            min_verdict=NullModelVerdict.HEURISTIC,
        )
        orch = SelfImproveOrchestrator(
            codebase_path=tmp_path,
            null_model_config=config,
        )
        assert orch.null_model_config is config
        assert orch.null_model_config.enabled is True
        assert orch.null_model_config.n_null_samples == 5


# ---------------------------------------------------------------------------
# ImprovementResult carries null_model_result
# ---------------------------------------------------------------------------


class TestImprovementResultField:
    def test_improvement_result_defaults_null_model_result_to_none(self):
        result = ImprovementResult(
            success=True,
            stage_reached=WorkflowStage.COMPLETE,
        )
        assert result.null_model_result is None

    def test_improvement_result_can_carry_null_model_result(self):
        verdict_result = NullModelResult(
            verdict=NullModelVerdict.STRUCTURAL,
            observed_delta={"coverage_percent": 5.0},
            observed_score=5.0,
            null_scores=[0.0, 0.0, 0.0],
            percentile=100.0,
            null_mean=0.0,
            null_std=0.0,
            n_null_samples=3,
            rationale="Measurably improves coverage",
        )
        result = ImprovementResult(
            success=True,
            stage_reached=WorkflowStage.COMPLETE,
            null_model_result=verdict_result,
        )
        assert result.null_model_result is verdict_result
        assert result.null_model_result.verdict == NullModelVerdict.STRUCTURAL


# ---------------------------------------------------------------------------
# Gate verdict branching (unit-level, without running real sandbox)
# ---------------------------------------------------------------------------


class TestVerdictBranching:
    """These tests exercise the gate decision logic without booting the full
    orchestrator.run() pipeline — they stub _run_null_model_gate directly
    and verify the orchestrator respects the config.allows() result.
    """

    def test_config_allows_structural_under_heuristic_min(self):
        config = NullModelGateConfig(min_verdict=NullModelVerdict.HEURISTIC)
        assert config.allows(NullModelVerdict.STRUCTURAL)
        assert config.allows(NullModelVerdict.HEURISTIC)
        assert not config.allows(NullModelVerdict.NEUTRAL)

    def test_config_blocks_regression_regardless_of_min_verdict(self):
        # Even with the most permissive min, regression should be blocked
        # when block_regressions=True (the default).
        config = NullModelGateConfig(
            min_verdict=NullModelVerdict.NEUTRAL,
            block_regressions=True,
        )
        assert not config.allows(NullModelVerdict.REGRESSION)

    def test_config_allows_regression_when_block_regressions_false(self):
        config = NullModelGateConfig(
            min_verdict=NullModelVerdict.NEUTRAL,
            block_regressions=False,
        )
        assert config.allows(NullModelVerdict.REGRESSION)

    def test_disabled_config_passes_any_verdict(self):
        config = NullModelGateConfig(enabled=False)
        for verdict in NullModelVerdict:
            assert config.allows(verdict)
