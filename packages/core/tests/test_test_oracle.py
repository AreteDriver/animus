"""Tests for TestOracleCitizen."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from animus.citizens import (
    ImprovementProposal,
    ProposalStatus,
    TestOracleCitizen,
)


class TestTestOracleCitizen:
    def test_initialization(self):
        oracle = TestOracleCitizen(codebase_path="/tmp/test")
        assert oracle.codebase_path == Path("/tmp/test")
        assert oracle._regressions == []

    # ------------------------------------------------------------------
    # observe_test_failures
    # ------------------------------------------------------------------

    def test_observe_test_failures_with_output(self):
        pytest_output = (
            "test_foo.py::test_bar FAILED\n"
            "test_foo.py::test_baz PASSED\n"
            "=========================== short test summary info ===========================\n"
            "FAILED test_foo.py::test_bar\n"
            "========================= 5 failed, 10 passed, 1 error in 0.5s ==========================\n"
        )

        oracle = TestOracleCitizen()
        observations = oracle.observe_test_failures(pytest_output)

        assert len(observations) >= 2
        fail_obs = [o for o in observations if o.context.get("pattern_type") == "test_failure"]
        assert len(fail_obs) >= 1
        assert fail_obs[0].context["failed"] == 5
        assert fail_obs[0].context["errors"] == 1
        assert fail_obs[0].severity == "critical"  # errors > 0

    def test_observe_test_failures_all_pass(self):
        pytest_output = "========================= 10 passed in 0.5s ==========================\n"
        oracle = TestOracleCitizen()
        observations = oracle.observe_test_failures(pytest_output)
        assert len(observations) == 0

    def test_observe_test_failures_flaky(self):
        pytest_output = (
            "test_x.py::test_y FAILED\n"
            "test_x.py::test_y PASSED\n"
            "========================= 0 failed, 1 passed in 0.5s ==========================\n"
        )
        oracle = TestOracleCitizen()
        observations = oracle.observe_test_failures(pytest_output)

        flaky = [o for o in observations if o.context.get("pattern_type") == "flaky_test"]
        assert len(flaky) == 1

    # ------------------------------------------------------------------
    # observe_coverage_gaps
    # ------------------------------------------------------------------

    def test_observe_coverage_gaps_low_total(self):
        report = (
            "Name         Stmts   Miss  Cover\n"
            "----------------------------------\n"
            "module.py       10     5    50%\n"
            "TOTAL           10     5    50%\n"
        )
        oracle = TestOracleCitizen()
        observations = oracle.observe_coverage_gaps(report)

        assert len(observations) >= 1
        assert any("50%" in o.description for o in observations)

    def test_observe_coverage_gaps_uncovered_files(self):
        # Regex expects: name + digits + 0 + 0%  (Miss=0, Cover=0%)
        report = (
            "Name         Stmts   Miss  Cover\n"
            "----------------------------------\n"
            "uncovered.py    10     0     0%\n"
            "TOTAL           10     0     0%\n"
        )
        oracle = TestOracleCitizen()
        observations = oracle.observe_coverage_gaps(report)

        uncovered = [o for o in observations if o.context.get("pattern_type") == "missing_coverage"]
        assert len(uncovered) == 1

    # ------------------------------------------------------------------
    # observe_eval_drift
    # ------------------------------------------------------------------

    def test_observe_eval_drift_detects_drop(self):
        eval_results = [
            {
                "suite": "personal-quality",
                "score": 0.85,
                "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
            },
            {
                "suite": "personal-quality",
                "score": 0.60,
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            },
        ]
        oracle = TestOracleCitizen()
        observations = oracle.observe_eval_drift(eval_results)

        assert len(observations) == 1
        assert observations[0].severity == "high"
        assert "dropped" in observations[0].description

    def test_observe_eval_drift_no_drift(self):
        eval_results = [
            {
                "suite": "personal-quality",
                "score": 0.85,
                "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
            },
            {
                "suite": "personal-quality",
                "score": 0.86,
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            },
        ]
        oracle = TestOracleCitizen()
        observations = oracle.observe_eval_drift(eval_results)
        assert len(observations) == 0

    def test_observe_eval_drift_small_decline(self):
        eval_results = [
            {
                "suite": "personal-quality",
                "score": 0.85,
                "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
            },
            {
                "suite": "personal-quality",
                "score": 0.78,
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            },
        ]
        oracle = TestOracleCitizen()
        observations = oracle.observe_eval_drift(eval_results)

        assert len(observations) == 1
        assert observations[0].severity == "medium"

    # ------------------------------------------------------------------
    # observe_uncitizened_modules
    # ------------------------------------------------------------------

    def test_observe_uncitizened_modules_finds_missing_tests(self, tmp_path):
        codebase = tmp_path / "codebase"
        codebase.mkdir()
        src = codebase / "module.py"
        src.write_text("# new code")
        # Touch to ensure recent mtime
        src.touch()

        oracle = TestOracleCitizen(codebase_path=codebase)
        observations = oracle.observe_uncitizened_modules()

        assert len(observations) >= 1
        assert observations[0].context["pattern_type"] == "missing_coverage"

    def test_observe_uncitizened_modules_skips_old_files(self, tmp_path):
        codebase = tmp_path / "codebase"
        codebase.mkdir()
        src = codebase / "old_module.py"
        src.write_text("# old code")
        # Set mtime to 30 days ago
        old_time = (datetime.now() - timedelta(days=30)).timestamp()
        import os

        os.utime(src, (old_time, old_time))

        oracle = TestOracleCitizen(codebase_path=codebase)
        observations = oracle.observe_uncitizened_modules()
        assert len(observations) == 0

    # ------------------------------------------------------------------
    # analyze
    # ------------------------------------------------------------------

    def test_analyze_no_findings(self):
        oracle = TestOracleCitizen()
        with (
            patch.object(oracle, "observe_test_failures", return_value=[]),
            patch.object(oracle, "observe_coverage_gaps", return_value=[]),
            patch.object(oracle, "observe_eval_drift", return_value=[]),
            patch.object(oracle, "observe_uncitizened_modules", return_value=[]),
        ):
            regressions = oracle.analyze()
        assert regressions == []

    def test_analyze_aggregates(self):
        from animus.citizens.architect import Observation

        oracle = TestOracleCitizen()
        with (
            patch.object(
                oracle,
                "observe_test_failures",
                return_value=[
                    Observation(
                        source="test_oracle",
                        description="5 failures",
                        severity="high",
                        context={"pattern_type": "test_failure"},
                    ),
                ],
            ),
            patch.object(oracle, "observe_coverage_gaps", return_value=[]),
            patch.object(oracle, "observe_eval_drift", return_value=[]),
            patch.object(oracle, "observe_uncitizened_modules", return_value=[]),
        ):
            regressions = oracle.analyze()

        assert len(regressions) == 1
        assert regressions[0].regression_type == "test_failure"

    # ------------------------------------------------------------------
    # generate_proposal
    # ------------------------------------------------------------------

    def test_generate_proposal_no_findings(self):
        oracle = TestOracleCitizen()
        with patch.object(oracle, "analyze", return_value=[]):
            proposal = oracle.generate_proposal()
        assert proposal is None

    def test_generate_proposal_with_findings(self):
        from animus.citizens.test_oracle import QualityRegression

        oracle = TestOracleCitizen()
        regressions = [
            QualityRegression(
                regression_type="test_failure",
                description="5 tests failed",
                severity="high",
            ),
        ]
        with patch.object(oracle, "analyze", return_value=regressions):
            proposal = oracle.generate_proposal()

        assert proposal is not None
        assert proposal.status == ProposalStatus.DRAFT
        assert proposal.id.startswith("ADL-")
        assert proposal.affected_components == ["Factory", "Kernel"]

    def test_generate_proposal_for_eval_drift(self):
        from animus.citizens.test_oracle import QualityRegression

        oracle = TestOracleCitizen()
        regressions = [
            QualityRegression(
                regression_type="eval_drift",
                description="Eval score dropped",
                severity="high",
            ),
        ]
        with patch.object(oracle, "analyze", return_value=regressions):
            proposal = oracle.generate_proposal()

        assert proposal.affected_components == ["Mind", "Factory"]

    # ------------------------------------------------------------------
    # store_proposal
    # ------------------------------------------------------------------

    def test_store_proposal_without_memory(self):
        oracle = TestOracleCitizen()
        proposal = ImprovementProposal(id="1", title="T", problem="P")
        assert oracle.store_proposal(proposal) is False

    def test_store_proposal_with_memory(self):
        mock_memory = MagicMock()
        oracle = TestOracleCitizen(memory_layer=mock_memory)
        proposal = ImprovementProposal(id="1", title="T", problem="P", recommendation="R")

        assert oracle.store_proposal(proposal) is True
        mock_memory.remember.assert_called_once()
        call_kwargs = mock_memory.remember.call_args.kwargs
        assert "test_oracle" in call_kwargs["tags"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def test_suggest_for_regression(self):
        assert "Investigate" in TestOracleCitizen._suggest_for_regression("test_failure")
        assert "mutation" in TestOracleCitizen._suggest_for_regression("coverage_drop")
        assert "calibration" in TestOracleCitizen._suggest_for_regression("eval_drift")
        assert "Isolate" in TestOracleCitizen._suggest_for_regression("flaky_test")
        assert "Create test" in TestOracleCitizen._suggest_for_regression("missing_coverage")

    def test_build_problem_recommendation(self):
        from animus.citizens.test_oracle import QualityRegression

        r = QualityRegression(
            regression_type="test_failure", description="5 failed", severity="high"
        )
        problem, recommendation = TestOracleCitizen._build_problem_recommendation(r)
        assert "failures" in problem
        assert "Fix" in recommendation

    def test_repr(self):
        oracle = TestOracleCitizen()
        assert "TestOracleCitizen" in repr(oracle)
