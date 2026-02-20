"""Evaluation runner for executing test suites."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

from .base import (
    EvalCase,
    EvalMetric,
    EvalResult,
    EvalStatus,
    EvalSuite,
    Evaluator,
)

logger = logging.getLogger(__name__)


@dataclass
class SuiteResult:
    """Result of running an evaluation suite.

    Attributes:
        suite: The evaluated suite
        results: Individual case results
        passed: Number of passed cases
        failed: Number of failed cases
        errors: Number of error cases
        skipped: Number of skipped cases
        total_score: Average score across all cases
        duration_ms: Total evaluation time
        timestamp: When evaluation was run
    """

    suite: EvalSuite
    results: list[EvalResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total_score: float = 0.0
    duration_ms: float = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "suite_name": self.suite.name,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "total": self.total,
            "pass_rate": self.pass_rate,
            "total_score": self.total_score,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "results": [r.to_dict() for r in self.results],
        }


class EvalRunner:
    """Runner for executing evaluation suites.

    Supports:
        - Sequential and parallel execution
        - Progress callbacks
        - Result aggregation
        - Multiple reporters
    """

    def __init__(
        self,
        evaluator: Evaluator,
        max_workers: int = 4,
        progress_callback: Callable[[int, int, EvalResult], None] | None = None,
    ):
        """Initialize evaluation runner.

        Args:
            evaluator: Evaluator to use for running cases
            max_workers: Maximum parallel workers
            progress_callback: Called after each case (current, total, result)
        """
        self.evaluator = evaluator
        self.max_workers = max_workers
        self.progress_callback = progress_callback

    def run(
        self,
        suite: EvalSuite,
        parallel: bool = False,
        filter_tags: list[str] | None = None,
    ) -> SuiteResult:
        """Run an evaluation suite.

        Args:
            suite: Suite to evaluate
            parallel: Whether to run cases in parallel
            filter_tags: Only run cases with these tags

        Returns:
            Suite evaluation result
        """
        import time

        start_time = time.time()

        # Filter cases if needed
        cases = suite.cases
        if filter_tags:
            cases = [
                c for c in cases if any(tag in c.metadata.get("tags", []) for tag in filter_tags)
            ]

        # Get metrics
        metrics = suite.metrics

        # Run evaluations
        if parallel and len(cases) > 1:
            results = self._run_parallel(cases, metrics)
        else:
            results = self._run_sequential(cases, metrics)

        # Aggregate results
        suite_result = self._aggregate_results(suite, results)
        suite_result.duration_ms = (time.time() - start_time) * 1000

        return suite_result

    async def run_async(
        self,
        suite: EvalSuite,
        parallel: bool = True,
        filter_tags: list[str] | None = None,
    ) -> SuiteResult:
        """Run an evaluation suite asynchronously.

        Args:
            suite: Suite to evaluate
            parallel: Whether to run cases in parallel
            filter_tags: Only run cases with these tags

        Returns:
            Suite evaluation result
        """
        import time

        start_time = time.time()

        # Filter cases
        cases = suite.cases
        if filter_tags:
            cases = [
                c for c in cases if any(tag in c.metadata.get("tags", []) for tag in filter_tags)
            ]

        metrics = suite.metrics

        if parallel and len(cases) > 1:
            results = await self._run_parallel_async(cases, metrics)
        else:
            results = await self._run_sequential_async(cases, metrics)

        suite_result = self._aggregate_results(suite, results)
        suite_result.duration_ms = (time.time() - start_time) * 1000

        return suite_result

    def _run_sequential(self, cases: list[EvalCase], metrics: list[EvalMetric]) -> list[EvalResult]:
        """Run cases sequentially."""
        results = []
        total = len(cases)

        for i, case in enumerate(cases):
            try:
                result = self.evaluator.evaluate(case, metrics)
            except Exception as e:
                logger.error(f"Error evaluating case {case.id}: {e}")
                result = EvalResult(
                    case=case,
                    status=EvalStatus.ERROR,
                    score=0.0,
                    output=None,
                    error=str(e),
                )

            results.append(result)

            if self.progress_callback:
                self.progress_callback(i + 1, total, result)

        return results

    def _run_parallel(self, cases: list[EvalCase], metrics: list[EvalMetric]) -> list[EvalResult]:
        """Run cases in parallel using threads."""
        results = [None] * len(cases)
        total = len(cases)
        completed = [0]

        def evaluate_case(idx: int, case: EvalCase) -> tuple[int, EvalResult]:
            try:
                result = self.evaluator.evaluate(case, metrics)
            except Exception as e:
                logger.error(f"Error evaluating case {case.id}: {e}")
                result = EvalResult(
                    case=case,
                    status=EvalStatus.ERROR,
                    score=0.0,
                    output=None,
                    error=str(e),
                )
            return idx, result

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(evaluate_case, i, case) for i, case in enumerate(cases)]

            for future in futures:
                idx, result = future.result()
                results[idx] = result
                completed[0] += 1

                if self.progress_callback:
                    self.progress_callback(completed[0], total, result)

        return results

    async def _run_sequential_async(
        self, cases: list[EvalCase], metrics: list[EvalMetric]
    ) -> list[EvalResult]:
        """Run cases sequentially (async)."""
        results = []
        total = len(cases)

        for i, case in enumerate(cases):
            try:
                result = await self.evaluator.evaluate_async(case, metrics)
            except Exception as e:
                logger.error(f"Error evaluating case {case.id}: {e}")
                result = EvalResult(
                    case=case,
                    status=EvalStatus.ERROR,
                    score=0.0,
                    output=None,
                    error=str(e),
                )

            results.append(result)

            if self.progress_callback:
                self.progress_callback(i + 1, total, result)

        return results

    async def _run_parallel_async(
        self, cases: list[EvalCase], metrics: list[EvalMetric]
    ) -> list[EvalResult]:
        """Run cases in parallel (async)."""

        async def evaluate_with_index(idx: int, case: EvalCase) -> tuple[int, EvalResult]:
            try:
                result = await self.evaluator.evaluate_async(case, metrics)
            except Exception as e:
                logger.error(f"Error evaluating case {case.id}: {e}")
                result = EvalResult(
                    case=case,
                    status=EvalStatus.ERROR,
                    score=0.0,
                    output=None,
                    error=str(e),
                )
            return idx, result

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_workers)

        async def bounded_evaluate(idx: int, case: EvalCase):
            async with semaphore:
                return await evaluate_with_index(idx, case)

        # Run all tasks
        tasks = [bounded_evaluate(i, case) for i, case in enumerate(cases)]
        indexed_results = await asyncio.gather(*tasks)

        # Sort by index
        indexed_results.sort(key=lambda x: x[0])
        results = [r for _, r in indexed_results]

        # Call progress callback for each result
        if self.progress_callback:
            for i, result in enumerate(results):
                self.progress_callback(i + 1, len(cases), result)

        return results

    def _aggregate_results(self, suite: EvalSuite, results: list[EvalResult]) -> SuiteResult:
        """Aggregate individual results into suite result."""
        suite_result = SuiteResult(suite=suite, results=results)

        scores = []
        for result in results:
            if result.status == EvalStatus.PASSED:
                suite_result.passed += 1
            elif result.status == EvalStatus.FAILED:
                suite_result.failed += 1
            elif result.status == EvalStatus.ERROR:
                suite_result.errors += 1
            else:
                suite_result.skipped += 1

            scores.append(result.score)

        if scores:
            suite_result.total_score = sum(scores) / len(scores)

        return suite_result

    def run_multiple(
        self,
        suites: list[EvalSuite],
        parallel_suites: bool = False,
    ) -> list[SuiteResult]:
        """Run multiple suites.

        Args:
            suites: Suites to evaluate
            parallel_suites: Whether to run suites in parallel

        Returns:
            List of suite results
        """
        if parallel_suites:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self.run, suites))
            return results

        return [self.run(suite) for suite in suites]


def create_quick_eval(
    agent_fn: Callable[[str], str],
    cases: list[dict],
    metrics: list[str] | None = None,
) -> SuiteResult:
    """Quick helper to run a simple evaluation.

    Args:
        agent_fn: Function that takes input and returns output
        cases: List of {"input": ..., "expected": ...} dicts
        metrics: Metric names to use (default: ["contains"])

    Returns:
        Suite result

    Example:
        result = create_quick_eval(
            agent_fn=lambda x: llm.complete(x),
            cases=[
                {"input": "What is 2+2?", "expected": "4"},
                {"input": "Capital of France?", "expected": "Paris"},
            ]
        )
        print(f"Pass rate: {result.pass_rate:.0%}")
    """
    from .base import AgentEvaluator
    from .metrics import ContainsMetric, ExactMatchMetric

    # Build suite
    suite = EvalSuite(name="quick_eval")
    for case_data in cases:
        suite.add_case(
            input=case_data["input"],
            expected=case_data.get("expected"),
            name=case_data.get("name", ""),
        )

    # Add metrics
    metric_map = {
        "contains": ContainsMetric(),
        "exact": ExactMatchMetric(),
    }
    metric_names = metrics or ["contains"]
    for name in metric_names:
        if name in metric_map:
            suite.add_metric(metric_map[name])

    # Run
    evaluator = AgentEvaluator(agent_fn)
    runner = EvalRunner(evaluator)
    return runner.run(suite)
