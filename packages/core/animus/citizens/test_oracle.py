"""Citizen 004 — The Test Oracle.

The permanent "quality sentinel" of Animus.

Responsibilities:
- Observe test suite health (counts, failure rates, flaky tests, coverage trends)
- Observe eval system results (calibration drift, suite scores, rubric health)
- Detect regressions: failing tests that were previously passing, eval scores dropping
- Identify uncovered code paths and missing test cases
- Propose test coverage improvements, eval suite expansions, and flaky-test fixes

Never:
- Modify code or tests directly
- Delete test cases autonomously
- Change eval rubrics without human approval

Instead:
    Observe → Analyze → Propose → Human Approval → Forge → Evidence → Merge
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from animus.citizens.architect import Observation
from animus.citizens.proposal import (
    EvidenceItem,
    ImprovementProposal,
    ProposalStatus,
    RiskAssessment,
)
from animus.logging import get_logger

logger = get_logger("citizens.test_oracle")


@dataclass
class QualityRegression:
    """A detected regression in test or eval quality."""

    regression_type: str  # "test_failure", "coverage_drop", "eval_drift", "flaky_test", "missing_coverage"
    description: str
    severity: str = "low"
    metric_before: float | None = None
    metric_after: float | None = None
    suggested_action: str = ""
    context: dict[str, Any] = field(default_factory=dict)


class TestOracleCitizen:
    """Citizen 004 — The Test Oracle.

    Continuously evaluates test suite and eval system health,
    proposing improvements to maintain code quality and evidence standards.

    This citizen NEVER modifies code, tests, or eval rubrics directly.
    It only observes, analyzes, and produces proposals.
    """

    def __init__(
        self,
        codebase_path: Path | str | None = None,
        memory_layer: Any = None,
        eval_results_dir: Path | str | None = None,
    ):
        self.codebase_path = Path(codebase_path).expanduser() if codebase_path else None
        self.memory = memory_layer
        self.eval_results_dir = (
            Path(eval_results_dir).expanduser() if eval_results_dir else None
        )
        self._regressions: list[QualityRegression] = []

    # ------------------------------------------------------------------
    # Observation methods (read-only)
    # ------------------------------------------------------------------

    def observe_test_failures(self, pytest_output: str = "") -> list[Observation]:
        """Observe recent test results for failures.

        Args:
            pytest_output: Raw pytest output text. If empty, attempts to read
                from a known log location.

        Returns:
            List of observations about test failures.
        """
        observations: list[Observation] = []

        if not pytest_output:
            pytest_output = self._read_pytest_output()

        if not pytest_output:
            return observations

        # Count failed tests
        failed_match = re.search(r"(\d+) failed", pytest_output)
        error_match = re.search(r"(\d+) error", pytest_output)
        passed_match = re.search(r"(\d+) passed", pytest_output)

        failed = int(failed_match.group(1)) if failed_match else 0
        errors = int(error_match.group(1)) if error_match else 0
        passed = int(passed_match.group(1)) if passed_match else 0

        if failed > 0 or errors > 0:
            severity = "critical" if errors > 0 else "high" if failed >= 5 else "medium"
            observations.append(
                Observation(
                    source="test_oracle",
                    description=f"Test suite has {failed} failures and {errors} errors ({passed} passed)",
                    severity=severity,
                    context={
                        "failed": failed,
                        "errors": errors,
                        "passed": passed,
                        "pattern_type": "test_failure",
                    },
                )
            )

        # Detect flaky patterns: FAILED in output followed by PASSED (rerun)
        flaky_runs = re.findall(r"FAILED.*?\n.*?PASSED", pytest_output, re.DOTALL)
        if flaky_runs:
            observations.append(
                Observation(
                    source="test_oracle",
                    description=f"Detected {len(flaky_runs)} potential flaky test(s) (failed then passed on rerun)",
                    severity="medium",
                    context={
                        "flaky_count": len(flaky_runs),
                        "pattern_type": "flaky_test",
                    },
                )
            )

        # Extract specific failing test names
        failure_lines = re.findall(r"FAILED\s+([\w/._:]+)", pytest_output)
        if failure_lines:
            observations.append(
                Observation(
                    source="test_oracle",
                    description=f"Failing tests: {', '.join(failure_lines[:5])}",
                    severity="high",
                    context={
                        "failing_tests": failure_lines,
                        "pattern_type": "test_failure",
                    },
                )
            )

        return observations

    def observe_coverage_gaps(self, coverage_report: str = "") -> list[Observation]:
        """Identify files with low or missing test coverage.

        Args:
            coverage_report: Coverage report text. If empty, attempts to read
                from a known log location.

        Returns:
            List of coverage gap observations.
        """
        observations: list[Observation] = []

        if not coverage_report:
            coverage_report = self._read_coverage_report()

        if not coverage_report:
            return observations

        # Parse coverage percentage
        total_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", coverage_report)
        if total_match:
            total_pct = int(total_match.group(1))
            if total_pct < 80:
                severity = "high" if total_pct < 60 else "medium"
                observations.append(
                    Observation(
                        source="test_oracle",
                        description=f"Overall test coverage is {total_pct}% — below recommended threshold",
                        severity=severity,
                        context={
                            "total_coverage": total_pct,
                            "pattern_type": "coverage_drop",
                        },
                    )
                )

        # Find uncovered files
        uncovered = re.findall(r"([\w/._]+)\s+\d+\s+0\s+0%", coverage_report)
        if uncovered:
            observations.append(
                Observation(
                    source="test_oracle",
                    description=f"{len(uncovered)} file(s) have 0% test coverage",
                    severity="medium",
                    context={
                        "uncovered_files": uncovered[:10],
                        "pattern_type": "missing_coverage",
                    },
                )
            )

        return observations

    def observe_eval_drift(self, eval_results: list[dict] | None = None) -> list[Observation]:
        """Observe eval system results for calibration drift or score regressions.

        Args:
            eval_results: List of eval result dicts. If None, attempts to read
                from eval_results_dir.

        Returns:
            List of eval drift observations.
        """
        observations: list[Observation] = []

        if eval_results is None:
            eval_results = self._read_eval_results()

        if not eval_results:
            return observations

        # Group by suite
        suite_scores: dict[str, list[tuple[datetime, float]]] = {}
        for result in eval_results:
            suite = result.get("suite", "unknown")
            score = result.get("score")
            ts_str = result.get("timestamp", "")
            if score is None:
                continue
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else datetime.now()
            except ValueError:
                ts = datetime.now()
            suite_scores.setdefault(suite, []).append((ts, float(score)))

        # Detect drift: compare latest vs previous
        for suite, scores in suite_scores.items():
            if len(scores) < 2:
                continue
            scores.sort(key=lambda x: x[0])
            prev_score = scores[-2][1]
            latest_score = scores[-1][1]
            delta = latest_score - prev_score

            if delta < -0.1:
                observations.append(
                    Observation(
                        source="test_oracle",
                        description=f"Eval suite '{suite}' score dropped by {abs(delta):.2f} ({prev_score:.2f} → {latest_score:.2f})",
                        severity="high",
                        context={
                            "suite": suite,
                            "previous_score": prev_score,
                            "latest_score": latest_score,
                            "delta": delta,
                            "pattern_type": "eval_drift",
                        },
                    )
                )
            elif delta < -0.05:
                observations.append(
                    Observation(
                        source="test_oracle",
                        description=f"Eval suite '{suite}' score declined by {abs(delta):.2f} ({prev_score:.2f} → {latest_score:.2f})",
                        severity="medium",
                        context={
                            "suite": suite,
                            "previous_score": prev_score,
                            "latest_score": latest_score,
                            "delta": delta,
                            "pattern_type": "eval_drift",
                        },
                    )
                )

        return observations

    def observe_uncitizened_modules(self) -> list[Observation]:
        """Find recently modified modules with no test coverage.

        Returns:
            List of observations about untested recent changes.
        """
        observations: list[Observation] = []

        if not self.codebase_path or not self.codebase_path.exists():
            return observations

        # Find recently modified Python files (last 14 days)
        cutoff = datetime.now() - timedelta(days=14)
        recent_files: list[Path] = []

        skip_prefixes = ("tests/", "/tests/", "test_", ".venv/", "node_modules/", ".git/", ".tox/")
        for py_file in self.codebase_path.rglob("*.py"):
            rel_path = str(py_file.relative_to(self.codebase_path))
            if any(rel_path.startswith(p) or p in rel_path for p in skip_prefixes):
                continue
            try:
                mtime = datetime.fromtimestamp(py_file.stat().st_mtime)
                if mtime > cutoff:
                    recent_files.append(py_file)
            except OSError:
                continue

        # Score by line count and surface only the top 5 most impactful
        uncovered: list[tuple[int, Path, str]] = []  # (line_count, path, rel_path)
        for src_file in recent_files:
            rel = src_file.relative_to(self.codebase_path)
            test_candidates = [
                src_file.parent / f"test_{src_file.name}",
                src_file.parent / "tests" / f"test_{src_file.name}",
                self.codebase_path / "tests" / f"test_{src_file.name}",
            ]
            has_test = any(t.exists() for t in test_candidates)
            if not has_test:
                try:
                    line_count = len(src_file.read_text().splitlines())
                except Exception:
                    line_count = 0
                uncovered.append((line_count, src_file, str(rel)))

        # Sort descending by line count and emit top 5
        uncovered.sort(key=lambda x: -x[0])
        for line_count, src_file, rel in uncovered[:5]:
            severity = "high" if line_count > 200 else "medium"
            observations.append(
                Observation(
                    source="test_oracle",
                    description=f"Recently modified module has no test file: '{rel}' ({line_count} lines)",
                    severity=severity,
                    context={
                        "file": rel,
                        "line_count": line_count,
                        "modified": src_file.stat().st_mtime,
                        "pattern_type": "missing_coverage",
                    },
                )
            )

        return observations

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def analyze(self) -> list[QualityRegression]:
        """Run all observations and produce QualityRegression records.

        Returns:
            List of detected quality regressions.
        """
        observations: list[Observation] = []
        observations.extend(self.observe_test_failures())
        observations.extend(self.observe_coverage_gaps())
        observations.extend(self.observe_eval_drift())
        observations.extend(self.observe_uncitizened_modules())

        regressions: list[QualityRegression] = []

        for obs in observations:
            rt = obs.context.get("pattern_type", "unknown") if obs.context else "unknown"
            regressions.append(
                QualityRegression(
                    regression_type=rt,
                    description=obs.description,
                    severity=obs.severity,
                    metric_before=obs.context.get("previous_score") if obs.context else None,
                    metric_after=obs.context.get("latest_score") if obs.context else None,
                    suggested_action=self._suggest_for_regression(rt),
                    context=obs.context or {},
                )
            )

        self._regressions = regressions
        return regressions

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def generate_proposal(self) -> ImprovementProposal | None:
        """Generate an improvement proposal from quality regression analysis.

        Returns:
            Improvement proposal, or None if no actionable findings.
        """
        regressions = self.analyze()

        if not regressions:
            logger.info("No quality regressions detected — no proposal generated")
            return None

        # Focus on highest-severity regression
        top = max(regressions, key=lambda r: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(r.severity, 0))

        evidence = [
            EvidenceItem(
                source="test_oracle",
                description=f"{r.regression_type}: {r.description}",
                data={"severity": r.severity, **r.context},
            )
            for r in regressions
            if r.regression_type == top.regression_type
        ]

        problem, recommendation = self._build_problem_recommendation(top)

        risks = [
            RiskAssessment(
                description="New tests may not catch the actual bug pattern",
                severity="low",
                mitigation="Use property-based testing and mutation testing",
                probability=0.3,
            ),
            RiskAssessment(
                description="Eval suite expansion may increase runtime costs",
                severity="low",
                mitigation="Run heavy evals nightly, lightweight on PR",
                probability=0.4,
            ),
        ]

        components = ["Factory", "Kernel"]
        if top.regression_type == "eval_drift":
            components = ["Mind", "Factory"]
        elif top.regression_type == "missing_coverage":
            components = ["Kernel"]

        proposal = ImprovementProposal(
            id=f"ADL-{datetime.now().strftime('%Y%m%d')}-{__import__('uuid').uuid4().hex[:6]}",
            title=f"Quality Maintenance: {problem[:50]}",
            problem=problem,
            evidence=evidence[:5],
            root_cause="Insufficient automated quality monitoring or test coverage gaps",
            recommendation=recommendation,
            alternatives_considered=["Status quo (regressions accumulate)", "Manual QA cycles"],
            expected_benefits="Faster detection of quality regressions; higher confidence in changes",
            potential_risks=risks,
            confidence_score=0.7,
            estimated_effort_hours=5.0,
            affected_components=components,
            evaluation_plan="Re-run Test Oracle scan after fixes; verify failure count and coverage improved",
            rollback_plan="Revert test changes via git; restore previous eval baseline",
            success_metrics=["Test failure count reduced to zero", "Coverage maintained or improved", "Eval drift eliminated"],
            status=ProposalStatus.DRAFT,
        )

        logger.info(f"Test Oracle generated proposal {proposal.id}")
        return proposal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def store_proposal(self, proposal: ImprovementProposal) -> bool:
        """Store a proposal in Animus memory.

        Args:
            proposal: Proposal to store.

        Returns:
            True if stored successfully.
        """
        if self.memory is None:
            logger.warning("Memory layer not available — proposal not persisted")
            return False

        try:
            from animus.memory import MemoryType

            self.memory.remember(
                content=f"{proposal.title}\n\n{proposal.problem}\n\nRecommendation: {proposal.recommendation}",
                memory_type=MemoryType.PROCEDURAL,
                tags=["test_oracle", "proposal", proposal.status.value],
                metadata=proposal.to_dict(),
            )
            logger.info(f"Proposal {proposal.id} stored in memory")
            return True
        except Exception as e:
            logger.error(f"Failed to store proposal: {e}")
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_pytest_output(self) -> str:
        """Attempt to read pytest output from known locations."""
        if not self.codebase_path:
            return ""
        for path in [
            self.codebase_path / "pytest-output.txt",
            self.codebase_path / ".animus" / "pytest-output.txt",
            self.codebase_path / "test-results" / "latest.txt",
        ]:
            if path.exists():
                return path.read_text()
        return ""

    def _read_coverage_report(self) -> str:
        """Attempt to read coverage report from known locations."""
        if not self.codebase_path:
            return ""
        for path in [
            self.codebase_path / "coverage.txt",
            self.codebase_path / ".animus" / "coverage.txt",
            self.codebase_path / "htmlcov" / "index.html",
        ]:
            if path.exists():
                return path.read_text()
        return ""

    def _read_eval_results(self) -> list[dict]:
        """Attempt to read eval results from known locations."""
        results: list[dict] = []

        # Try Forge eval store first
        try:
            from animus.citizens.eval_evidence import query_eval_runs, read_eval_results_from_memory
            forge_results = query_eval_runs(limit=20)
            if forge_results:
                results.extend(forge_results)
        except Exception:
            pass

        # Fall back to JSON files
        if self.eval_results_dir and self.eval_results_dir.exists():
            for json_file in sorted(self.eval_results_dir.glob("*.json"))[-10:]:
                try:
                    data = json.loads(json_file.read_text())
                    if isinstance(data, list):
                        results.extend(data)
                    elif isinstance(data, dict):
                        results.append(data)
                except Exception:
                    continue

        # Also check memory for eval results
        if self.memory and not results:
            try:
                from animus.citizens.eval_evidence import read_eval_results_from_memory
                mem_results = read_eval_results_from_memory(self.memory, limit=50)
                if mem_results:
                    results.extend(mem_results)
            except Exception:
                pass

        return results

    @staticmethod
    def _suggest_for_regression(regression_type: str) -> str:
        """Generate a suggestion for a given regression type."""
        suggestions = {
            "test_failure": (
                "Investigate root cause of failures. Add minimal reproducing test cases. "
                "Run bisect to identify offending commit."
            ),
            "coverage_drop": (
                "Add tests for uncovered paths. Use mutation testing to find gaps. "
                "Set coverage gate in CI."
            ),
            "eval_drift": (
                "Re-run eval calibration. Check prompt versions and model changes. "
                "Add adversarial examples to suite."
            ),
            "flaky_test": (
                "Isolate flakiness source (timing, state leakage, async race). "
                "Add deterministic ordering or mocking."
            ),
            "missing_coverage": (
                "Create test file for recently modified module. Prioritize happy-path and edge cases."
            ),
        }
        return suggestions.get(regression_type, "Review and improve test coverage.")

    @staticmethod
    def _build_problem_recommendation(regression: QualityRegression) -> tuple[str, str]:
        """Build problem/recommendation pair from regression."""
        if regression.regression_type == "test_failure":
            return (
                f"Test suite has failures: {regression.description[:80]}",
                "Fix failing tests and add regression prevention tests.",
            )
        elif regression.regression_type == "coverage_drop":
            return (
                f"Coverage regression: {regression.description[:80]}",
                "Add tests for uncovered code paths. Set minimum coverage threshold in CI.",
            )
        elif regression.regression_type == "eval_drift":
            return (
                f"Eval drift detected: {regression.description[:80]}",
                "Re-run eval suite, check for model/prompt changes, add adversarial cases.",
            )
        elif regression.regression_type == "flaky_test":
            return (
                f"Flaky tests detected: {regression.description[:80]}",
                "Isolate timing/state dependencies. Use mocks or deterministic fixtures.",
            )
        elif regression.regression_type == "missing_coverage":
            return (
                f"Missing test coverage: {regression.description[:80]}",
                "Create corresponding test files for recently modified modules.",
            )
        else:
            return (
                f"Quality regression: {regression.description[:80]}",
                "Review test and eval health. Add missing coverage.",
            )

    def __repr__(self) -> str:
        return f"TestOracleCitizen(regressions={len(self._regressions)})"