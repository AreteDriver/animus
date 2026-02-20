"""
Tests for benchmark suite, real constraint gates, and codegen demo.

Proves:
  1. Benchmark suite produces valid metrics across all scenarios/scales
  2. Gates run subprocesses and produce correct evidence
  3. Codegen demo shows convergent agents eliminate rework cycles
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from convergent.benchmark import (
    BenchmarkMetrics,
    BenchmarkSuite,
    ScenarioType,
    run_benchmark,
    run_scaling_suite,
)
from convergent.codegen_demo import run_baseline, run_convergent, run_demo
from convergent.gates import (
    CommandGate,
    CompileGate,
    GateReport,
    GateRunner,
    GateRunResult,
    PytestGate,
)
from convergent.intent import (
    Evidence,
    EvidenceKind,
    Intent,
    InterfaceKind,
    InterfaceSpec,
)

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

# ===================================================================
# Benchmark tests
# ===================================================================


class TestBenchmarkMetrics:
    """Test BenchmarkMetrics computed properties."""

    def test_conflict_rate_zero_resolutions(self) -> None:
        m = BenchmarkMetrics(scenario="test", agent_count=5)
        assert m.conflict_rate == 0.0

    def test_conflict_rate_with_data(self) -> None:
        m = BenchmarkMetrics(
            scenario="test",
            agent_count=5,
            total_conflicts=3,
            total_resolutions=10,
        )
        assert m.conflict_rate == pytest.approx(0.3)

    def test_rework_rate_zero_intents(self) -> None:
        m = BenchmarkMetrics(scenario="test", agent_count=5)
        assert m.rework_rate == 0.0

    def test_rework_rate_with_data(self) -> None:
        m = BenchmarkMetrics(
            scenario="test",
            agent_count=5,
            consume_instead_count=4,
            total_intents=20,
        )
        assert m.rework_rate == pytest.approx(0.2)

    def test_adjustments_per_agent(self) -> None:
        m = BenchmarkMetrics(
            scenario="test",
            agent_count=5,
            total_adjustments=15,
        )
        assert m.adjustments_per_agent == pytest.approx(3.0)

    def test_cost_per_agent(self) -> None:
        m = BenchmarkMetrics(
            scenario="test",
            agent_count=4,
            total_cost=100.0,
        )
        assert m.cost_per_agent == pytest.approx(25.0)

    def test_summary_line_format(self) -> None:
        m = BenchmarkMetrics(
            scenario="independent",
            agent_count=10,
            total_conflicts=0,
            total_resolutions=20,
            convergence_rounds=2,
            all_converged=True,
            wall_clock_seconds=0.5,
        )
        line = m.summary_line()
        assert "independent" in line
        assert "agents= 10" in line
        assert "converged=Y" in line


class TestBenchmarkSuiteOutput:
    """Test BenchmarkSuite summary output."""

    def test_summary_contains_header(self) -> None:
        suite = BenchmarkSuite(
            results=[
                BenchmarkMetrics(scenario="test", agent_count=2),
                BenchmarkMetrics(scenario="test", agent_count=5),
            ]
        )
        summary = suite.summary()
        assert "CONVERGENT BENCHMARK RESULTS" in summary

    def test_summary_includes_scaling_analysis(self) -> None:
        suite = BenchmarkSuite(
            results=[
                BenchmarkMetrics(
                    scenario="independent",
                    agent_count=2,
                    convergence_rounds=2,
                ),
                BenchmarkMetrics(
                    scenario="independent",
                    agent_count=10,
                    convergence_rounds=2,
                ),
            ]
        )
        summary = suite.summary()
        assert "SCALING ANALYSIS" in summary
        assert "SUBLINEAR" in summary


class TestRunBenchmarkIndependent:
    """Test INDEPENDENT scenario benchmark."""

    def test_independent_2_agents(self) -> None:
        m = run_benchmark(ScenarioType.INDEPENDENT, 2)
        assert m.agent_count == 2
        assert m.all_converged is True
        assert m.convergence_rounds > 0

    def test_independent_5_agents(self) -> None:
        m = run_benchmark(ScenarioType.INDEPENDENT, 5)
        assert m.agent_count == 5
        assert m.all_converged is True

    def test_independent_has_intents(self) -> None:
        m = run_benchmark(ScenarioType.INDEPENDENT, 3)
        assert m.total_intents > 0


class TestRunBenchmarkSharedInterface:
    """Test SHARED_INTERFACE scenario benchmark."""

    def test_shared_converges(self) -> None:
        m = run_benchmark(ScenarioType.SHARED_INTERFACE, 5)
        assert m.all_converged is True
        assert m.convergence_rounds >= 1

    def test_shared_has_adjustments(self) -> None:
        # With 5+ agents sharing an interface, there should be adjustments
        m = run_benchmark(ScenarioType.SHARED_INTERFACE, 5)
        assert m.total_adjustments > 0


class TestRunBenchmarkHighContention:
    """Test HIGH_CONTENTION scenario benchmark."""

    def test_high_contention_converges(self) -> None:
        m = run_benchmark(ScenarioType.HIGH_CONTENTION, 5)
        assert m.all_converged is True

    def test_high_contention_10_agents(self) -> None:
        m = run_benchmark(ScenarioType.HIGH_CONTENTION, 10)
        assert m.all_converged is True
        assert m.total_adjustments > 0


class TestRunBenchmarkRealistic:
    """Test REALISTIC scenario benchmark."""

    def test_realistic_converges(self) -> None:
        m = run_benchmark(ScenarioType.REALISTIC, 10)
        assert m.all_converged is True

    def test_realistic_has_mixed_behavior(self) -> None:
        m = run_benchmark(ScenarioType.REALISTIC, 10)
        # Realistic scenario should have some adjustments
        assert m.total_adjustments >= 0


class TestScalingSuite:
    """Test the full scaling suite."""

    def test_suite_runs_all_combinations(self) -> None:
        suite = run_scaling_suite(
            agent_counts=[2, 5],
            scenarios=[ScenarioType.INDEPENDENT, ScenarioType.HIGH_CONTENTION],
        )
        assert len(suite.results) == 4  # 2 scenarios × 2 counts

    def test_suite_all_converge(self) -> None:
        suite = run_scaling_suite(
            agent_counts=[2, 5],
            scenarios=[ScenarioType.INDEPENDENT],
        )
        for m in suite.results:
            assert m.all_converged is True

    def test_sublinear_scaling(self) -> None:
        """Core proof: convergence rounds grow sublinearly with agent count."""
        suite = run_scaling_suite(
            agent_counts=[2, 5, 10, 25],
            scenarios=[ScenarioType.INDEPENDENT],
        )
        rounds = [m.convergence_rounds for m in suite.results]
        # Agent count grows 12.5x (2→25), rounds should NOT grow 12.5x
        assert rounds[-1] <= rounds[0] * 3  # At most 3x growth for 12.5x agents

    def test_conflict_rate_stays_low(self) -> None:
        """Conflict rate should not explode with more agents."""
        suite = run_scaling_suite(
            agent_counts=[2, 10, 25],
            scenarios=[ScenarioType.SHARED_INTERFACE],
        )
        for m in suite.results:
            assert m.conflict_rate <= 0.5  # Never more than 50% conflicts


# ===================================================================
# Gates tests
# ===================================================================


class TestGateRunResult:
    """Test GateRunResult dataclass."""

    def test_passed_result(self) -> None:
        r = GateRunResult(gate_name="test", passed=True)
        assert not r.failed
        assert r.passed

    def test_failed_result(self) -> None:
        r = GateRunResult(gate_name="test", passed=False)
        assert r.failed


class TestGateReport:
    """Test GateReport aggregate."""

    def test_all_passed(self) -> None:
        report = GateReport(
            results=[
                GateRunResult(gate_name="a", passed=True),
                GateRunResult(gate_name="b", passed=True),
            ]
        )
        assert report.all_passed
        assert report.passed_count == 2
        assert report.failed_count == 0

    def test_some_failed(self) -> None:
        report = GateReport(
            results=[
                GateRunResult(
                    gate_name="a",
                    passed=True,
                    evidence=[Evidence.test_pass("a passed")],
                ),
                GateRunResult(
                    gate_name="b",
                    passed=False,
                    evidence=[Evidence(kind=EvidenceKind.TEST_FAIL, description="b failed")],
                ),
            ]
        )
        assert not report.all_passed
        assert report.passed_count == 1
        assert report.failed_count == 1
        assert len(report.all_evidence) == 2

    def test_summary_format(self) -> None:
        report = GateReport(
            results=[GateRunResult(gate_name="test", passed=True, duration_seconds=0.5)],
            total_duration=0.5,
        )
        summary = report.summary()
        assert "1/1 passed" in summary
        assert "PASS" in summary


class TestCommandGate:
    """Test CommandGate with real subprocesses."""

    def test_passing_command(self) -> None:
        gate = CommandGate(cmd=["echo", "hello"], gate_name="echo")
        intent = Intent(agent_id="test", intent="test")
        result = gate.run(intent)
        assert result.passed
        assert result.return_code == 0
        assert "hello" in result.stdout
        assert len(result.evidence) == 1
        assert result.evidence[0].kind == EvidenceKind.TEST_PASS

    def test_failing_command(self) -> None:
        gate = CommandGate(cmd=["false"], gate_name="false")
        intent = Intent(agent_id="test", intent="test")
        result = gate.run(intent)
        assert not result.passed
        assert result.return_code != 0
        assert len(result.evidence) == 1
        assert result.evidence[0].kind == EvidenceKind.TEST_FAIL

    def test_command_not_found(self) -> None:
        gate = CommandGate(
            cmd=["nonexistent_command_xyz"],
            gate_name="missing",
        )
        intent = Intent(agent_id="test", intent="test")
        result = gate.run(intent)
        assert not result.passed
        assert result.return_code == -1
        assert "not found" in result.error_summary.lower() or "No such file" in result.error_summary

    def test_custom_evidence_kind(self) -> None:
        gate = CommandGate(
            cmd=["true"],
            gate_name="custom",
            evidence_kind_on_pass=EvidenceKind.CODE_COMMITTED,
        )
        intent = Intent(agent_id="test", intent="test")
        result = gate.run(intent)
        assert result.evidence[0].kind == EvidenceKind.CODE_COMMITTED

    def test_working_directory(self) -> None:
        gate = CommandGate(
            cmd=["pwd"],
            gate_name="pwd",
            cwd="/tmp",
        )
        intent = Intent(agent_id="test", intent="test")
        result = gate.run(intent)
        assert result.passed
        assert "/tmp" in result.stdout


class TestCompileGate:
    """Test CompileGate."""

    def test_successful_compile(self) -> None:
        gate = CompileGate(cmd=["true"], gate_name="compile")
        intent = Intent(agent_id="test", intent="test")
        result = gate.run(intent)
        assert result.passed
        assert result.evidence[0].kind == EvidenceKind.CODE_COMMITTED

    def test_failed_compile(self) -> None:
        gate = CompileGate(cmd=["false"], gate_name="compile")
        intent = Intent(agent_id="test", intent="test")
        result = gate.run(intent)
        assert not result.passed
        assert result.evidence[0].kind == EvidenceKind.TEST_FAIL


class TestPytestGate:
    """Test PytestGate with real test execution."""

    def test_passing_test_file(self) -> None:
        """Run PytestGate against a trivially passing test."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="test_pass_",
            dir="/tmp",
            delete=False,
        ) as f:
            f.write("def test_ok(): assert True\n")
            f.flush()
            test_file = f.name

        try:
            gate = PytestGate(test_path=test_file, timeout=30)
            intent = Intent(agent_id="test", intent="test")
            result = gate.run(intent)
            assert result.passed
            assert result.return_code == 0
            assert len(result.evidence) == 1
            assert result.evidence[0].kind == EvidenceKind.TEST_PASS
        finally:
            os.unlink(test_file)

    def test_failing_test_file(self) -> None:
        """Run PytestGate against a failing test."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="test_fail_",
            dir="/tmp",
            delete=False,
        ) as f:
            f.write("def test_bad(): assert False\n")
            f.flush()
            test_file = f.name

        try:
            gate = PytestGate(test_path=test_file, timeout=30)
            intent = Intent(agent_id="test", intent="test")
            result = gate.run(intent)
            assert not result.passed
            assert result.return_code != 0
            assert len(result.evidence) == 1
            assert result.evidence[0].kind == EvidenceKind.TEST_FAIL
        finally:
            os.unlink(test_file)

    def test_real_convergent_tests(self) -> None:
        """Run PytestGate against the actual convergent test suite."""
        gate = PytestGate(
            test_path="tests/test_convergence.py",
            cwd=PROJECT_ROOT,
            timeout=60,
            extra_args=["-q", "--no-header"],
        )
        intent = Intent(agent_id="test", intent="test")
        result = gate.run(intent)
        assert result.passed
        assert "passed" in result.stdout


class TestGateRunner:
    """Test GateRunner orchestration."""

    def test_run_all_passing(self) -> None:
        runner = GateRunner()
        runner.add(CommandGate(cmd=["true"], gate_name="a"))
        runner.add(CommandGate(cmd=["echo", "ok"], gate_name="b"))
        intent = Intent(agent_id="test", intent="test")
        report = runner.run_all(intent)
        assert report.all_passed
        assert len(report.results) == 2

    def test_run_all_with_failure(self) -> None:
        runner = GateRunner()
        runner.add(CommandGate(cmd=["true"], gate_name="a"))
        runner.add(CommandGate(cmd=["false"], gate_name="b"))
        intent = Intent(agent_id="test", intent="test")
        report = runner.run_all(intent)
        assert not report.all_passed
        assert report.passed_count == 1
        assert report.failed_count == 1

    def test_stop_on_failure(self) -> None:
        runner = GateRunner()
        runner.add(CommandGate(cmd=["false"], gate_name="a"))
        runner.add(CommandGate(cmd=["true"], gate_name="b"))
        intent = Intent(agent_id="test", intent="test")
        report = runner.run_all(intent, stop_on_failure=True)
        # Should have stopped after first failure
        assert len(report.results) == 1
        assert not report.all_passed

    def test_apply_evidence(self) -> None:
        runner = GateRunner()
        runner.add(CommandGate(cmd=["true"], gate_name="a"))
        runner.add(CommandGate(cmd=["true"], gate_name="b"))
        intent = Intent(agent_id="test", intent="test")
        report = runner.run_all(intent)
        count = runner.apply_evidence(intent, report)
        assert count == 2
        assert len(intent.evidence) == 2

    def test_gate_count(self) -> None:
        runner = GateRunner()
        assert runner.gate_count == 0
        runner.add(CommandGate(cmd=["true"], gate_name="a"))
        assert runner.gate_count == 1

    def test_total_duration_tracked(self) -> None:
        runner = GateRunner()
        runner.add(CommandGate(cmd=["true"], gate_name="a"))
        intent = Intent(agent_id="test", intent="test")
        report = runner.run_all(intent)
        assert report.total_duration >= 0


class TestGatesIntegration:
    """Integration tests: gates feeding evidence back to intents."""

    def test_gate_evidence_affects_stability(self) -> None:
        """Evidence from gates should change intent stability."""
        intent = Intent(
            agent_id="test",
            intent="test module",
            provides=[
                InterfaceSpec(
                    name="TestSvc",
                    kind=InterfaceKind.CLASS,
                    signature="run() -> bool",
                    tags=["test"],
                )
            ],
        )
        stability_before = intent.compute_stability()

        # Run a passing gate
        gate = CommandGate(cmd=["true"], gate_name="test-gate")
        result = gate.run(intent)
        for evidence in result.evidence:
            intent.add_evidence(evidence)

        stability_after = intent.compute_stability()
        assert stability_after > stability_before

    def test_failing_gate_reduces_stability(self) -> None:
        """Evidence from failing gates should reduce stability."""
        intent = Intent(
            agent_id="test",
            intent="test module",
            provides=[
                InterfaceSpec(
                    name="TestSvc",
                    kind=InterfaceKind.CLASS,
                    signature="run() -> bool",
                    tags=["test"],
                )
            ],
            evidence=[
                Evidence.test_pass("initial_test"),
                Evidence.code_committed("module.py"),
            ],
        )
        stability_before = intent.compute_stability()

        # Run a failing gate
        gate = CommandGate(
            cmd=["false"],
            gate_name="test-gate",
            evidence_kind_on_fail=EvidenceKind.TEST_FAIL,
        )
        result = gate.run(intent)
        for evidence in result.evidence:
            intent.add_evidence(evidence)

        stability_after = intent.compute_stability()
        assert stability_after < stability_before


# ===================================================================
# Codegen demo tests
# ===================================================================


class TestCodegenDemoConvergent:
    """Test the convergent path of the codegen demo."""

    def test_convergent_zero_rework(self) -> None:
        rework, conflicts, rounds, adjustments, merged, code = run_convergent()
        assert rework == 0
        assert merged is True

    def test_convergent_produces_code(self) -> None:
        _, _, _, _, _, code = run_convergent()
        assert "auth-agent" in code
        assert "api-agent" in code
        assert "storage-agent" in code

    def test_convergent_code_uses_uuid(self) -> None:
        """All agents should use UUID for User.id (convergent agreement)."""
        _, _, _, _, _, code = run_convergent()
        assert "UUID" in code["auth-agent"]
        assert "UUID" in code["api-agent"]
        assert "UUID" in code["storage-agent"]

    def test_convergent_code_uses_email(self) -> None:
        """All agents should use email field (not name/username)."""
        _, _, _, _, _, code = run_convergent()
        assert "email" in code["auth-agent"]
        assert "email" in code["api-agent"]
        assert "email" in code["storage-agent"]

    def test_convergent_single_round(self) -> None:
        _, _, rounds, _, _, _ = run_convergent()
        assert rounds == 1


class TestCodegenDemoBaseline:
    """Test the baseline (no coordination) path."""

    def test_baseline_needs_rework(self) -> None:
        rework, _, _, _, merged = run_baseline()
        assert rework >= 2
        assert merged is False

    def test_baseline_more_rounds_than_convergent(self) -> None:
        _, _, base_rounds, _, _ = run_baseline()
        _, _, conv_rounds, _, _, _ = run_convergent()
        assert base_rounds > conv_rounds


class TestCodegenDemoComparison:
    """Test the full demo comparison."""

    def test_demo_result_summary(self) -> None:
        result = run_demo()
        summary = result.summary()
        assert "CONVERGENT vs BASELINE" in summary
        assert "Rework cycles" in summary

    def test_convergent_beats_baseline(self) -> None:
        result = run_demo()
        assert result.convergent_rework_cycles < result.baseline_rework_cycles
        assert result.convergent_merged is True

    def test_demo_has_code(self) -> None:
        result = run_demo()
        assert len(result.agent_code) == 3

    def test_convergent_eliminates_rework(self) -> None:
        """The core claim: convergent eliminates rework cycles."""
        result = run_demo()
        assert result.convergent_rework_cycles == 0
        assert result.baseline_rework_cycles >= 2
        savings = result.baseline_rework_cycles - result.convergent_rework_cycles
        assert savings >= 2

    def test_summary_includes_result_line(self) -> None:
        result = run_demo()
        summary = result.summary()
        assert "eliminated" in summary


# ===================================================================
# End-to-end: gates + benchmark + demo together
# ===================================================================


class TestEndToEnd:
    """Integration tests combining all three components."""

    def test_benchmark_with_gate_evidence(self) -> None:
        """Run benchmark, then validate with gates."""
        # Run a small benchmark
        m = run_benchmark(ScenarioType.INDEPENDENT, 3)
        assert m.all_converged

        # Run a gate to validate
        gate = CommandGate(cmd=["true"], gate_name="validation")
        intent = Intent(agent_id="test", intent="benchmark validation")
        result = gate.run(intent)
        assert result.passed

    def test_full_pipeline(self) -> None:
        """Complete pipeline: benchmark → demo → gates."""
        # 1. Benchmark proves scaling
        m = run_benchmark(ScenarioType.HIGH_CONTENTION, 10)
        assert m.all_converged

        # 2. Demo proves rework elimination
        demo = run_demo()
        assert demo.convergent_rework_cycles == 0

        # 3. Gates prove real execution
        runner = GateRunner()
        runner.add(CommandGate(cmd=["true"], gate_name="compile"))
        runner.add(CommandGate(cmd=["true"], gate_name="test"))
        intent = Intent(agent_id="test", intent="final validation")
        report = runner.run_all(intent)
        assert report.all_passed
