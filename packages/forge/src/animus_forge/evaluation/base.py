"""Base classes for agent evaluation."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EvalStatus(Enum):
    """Status of an evaluation."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class EvalCase:
    """A single evaluation test case.

    Attributes:
        input: The input to the agent (prompt, messages, etc.)
        expected: Expected output or criteria
        metadata: Additional metadata (tags, difficulty, etc.)
        id: Unique case identifier
        name: Human-readable name
    """

    input: str | dict[str, Any]
    expected: str | dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"case_{self.id}"


@dataclass
class EvalResult:
    """Result of evaluating a single case.

    Attributes:
        case: The evaluated case
        status: Pass/fail/error status
        score: Numeric score (0-1)
        output: Actual agent output
        metrics: Individual metric scores
        error: Error message if status is ERROR
        latency_ms: Time taken for agent response
        tokens_used: Tokens consumed
        timestamp: When evaluation was run
        failure_mode: Structured failure bucket (FailureMode value) — None
            when the case passed or was skipped. Populated by
            ``FailureClassifier`` after evaluation.
        rubric_band: Letter grade (A/B/C/D/F) assigned by a Rubric. None
            when no rubric was applied.
        rubric_scores: Per-dim scores from the applied rubric, if any.
            Keys are rubric dim names; values are 0-1 floats.
        cost_usd: USD cost of this case, as estimated by CostCalculator.
            Zero for free providers (e.g. Ollama) or when unknown.
    """

    case: EvalCase
    status: EvalStatus
    score: float
    output: str | Any
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0
    tokens_used: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    failure_mode: str | None = None
    rubric_band: str | None = None
    rubric_scores: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == EvalStatus.PASSED

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "case_id": self.case.id,
            "case_name": self.case.name,
            "status": self.status.value,
            "score": self.score,
            "output": str(self.output)[:500],  # Truncate for readability
            "metrics": self.metrics,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp.isoformat(),
            "failure_mode": self.failure_mode,
            "rubric_band": self.rubric_band,
            "rubric_scores": self.rubric_scores,
            "cost_usd": self.cost_usd,
        }


def _unique_metric_key(base_name: str, existing: dict) -> str:
    """Return a dict key that doesn't collide with existing entries.

    Evaluators key ``metric_scores`` by ``metric.name``. Multiple metrics
    of the same class (e.g. three ``RegexMatchMetric`` with different
    patterns) share a name and would clobber each other, corrupting the
    composite average. This helper auto-suffixes duplicates so every
    metric invocation contributes to the composite.
    """
    if base_name not in existing:
        return base_name
    suffix = 1
    while f"{base_name}_{suffix}" in existing:
        suffix += 1
    return f"{base_name}_{suffix}"


def _band_from_score(score: float) -> str:
    """Map a 0-1 score to a letter band (A/B/C/D/F).

    Bands align with the ``personal-quality`` rubric:
        A: >= 0.90
        B: >= 0.80
        C: >= 0.70
        D: >= 0.60
        F: < 0.60
    """
    if score >= 0.90:
        return "A"
    if score >= 0.80:
        return "B"
    if score >= 0.70:
        return "C"
    if score >= 0.60:
        return "D"
    return "F"


class EvalMetric(ABC):
    """Abstract base class for evaluation metrics.

    Metrics score agent outputs on specific dimensions like
    accuracy, relevance, safety, etc.

    Attributes:
        fail_fast: When True, a score < 1.0 from this metric forces the
            case to EvalStatus.FAILED regardless of composite score. Used
            for hard structural invariants (e.g., "rationale must not
            contain raw JSON") that shouldn't be averaged with soft-
            quality rubric dims. Class-level default False; loader sets
            per-instance from YAML `fail_fast: true` in the metric spec.
    """

    fail_fast: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name."""
        pass

    @abstractmethod
    def score(
        self,
        output: str | Any,
        expected: str | Any | None,
        case: EvalCase,
    ) -> float:
        """Score the output.

        Args:
            output: Agent output
            expected: Expected output (if any)
            case: The full evaluation case

        Returns:
            Score between 0 and 1
        """
        pass

    def __str__(self) -> str:
        return self.name


@dataclass
class EvalSuite:
    """A collection of evaluation cases.

    Attributes:
        name: Suite name
        cases: List of evaluation cases
        metrics: Metrics to apply
        description: Suite description
        tags: Tags for categorization
    """

    name: str
    cases: list[EvalCase] = field(default_factory=list)
    metrics: list[EvalMetric] = field(default_factory=list)
    description: str = ""
    tags: list[str] = field(default_factory=list)
    threshold: float = 0.7  # Default passing threshold

    def add_case(
        self,
        input: str | dict,
        expected: str | dict | None = None,
        name: str = "",
        **metadata,
    ) -> EvalCase:
        """Add a case to the suite."""
        case = EvalCase(
            input=input,
            expected=expected,
            name=name,
            metadata=metadata,
        )
        self.cases.append(case)
        return case

    def add_metric(self, metric: EvalMetric) -> None:
        """Add a metric to the suite."""
        self.metrics.append(metric)

    def filter_by_tag(self, tag: str) -> list[EvalCase]:
        """Get cases with a specific tag."""
        return [c for c in self.cases if tag in c.metadata.get("tags", [])]

    @classmethod
    def from_yaml(cls, path: str) -> EvalSuite:
        """Load suite from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        suite = cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            threshold=data.get("threshold", 0.7),
        )

        for case_data in data.get("cases", []):
            suite.add_case(
                input=case_data["input"],
                expected=case_data.get("expected"),
                name=case_data.get("name", ""),
                **case_data.get("metadata", {}),
            )

        return suite

    def to_yaml(self, path: str) -> None:
        """Save suite to YAML file."""
        import yaml

        data = {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "threshold": self.threshold,
            "cases": [
                {
                    "name": c.name,
                    "input": c.input,
                    "expected": c.expected,
                    "metadata": c.metadata,
                }
                for c in self.cases
            ],
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


class Evaluator(ABC):
    """Abstract base class for agent evaluators.

    An evaluator runs an agent on evaluation cases and collects results.
    """

    @abstractmethod
    def evaluate(
        self,
        case: EvalCase,
        metrics: list[EvalMetric],
    ) -> EvalResult:
        """Evaluate a single case.

        Args:
            case: The case to evaluate
            metrics: Metrics to apply

        Returns:
            Evaluation result
        """
        pass

    async def evaluate_async(
        self,
        case: EvalCase,
        metrics: list[EvalMetric],
    ) -> EvalResult:
        """Evaluate a case asynchronously.

        Default implementation wraps sync method.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.evaluate(case, metrics))


class AgentEvaluator(Evaluator):
    """Evaluator that wraps a callable agent function."""

    def __init__(
        self,
        agent_fn: Callable[[str | dict], str | Any],
        threshold: float = 0.7,
    ):
        """Initialize with an agent function.

        Args:
            agent_fn: Callable that takes input and returns output
            threshold: Score threshold for passing
        """
        self.agent_fn = agent_fn
        self.threshold = threshold

    def evaluate(
        self,
        case: EvalCase,
        metrics: list[EvalMetric],
    ) -> EvalResult:
        """Evaluate a single case."""
        import time

        start_time = time.time()
        error = None
        output = None

        try:
            output = self.agent_fn(case.input)
        except Exception as e:
            error = str(e)
            return EvalResult(
                case=case,
                status=EvalStatus.ERROR,
                score=0.0,
                output=None,
                error=error,
                latency_ms=(time.time() - start_time) * 1000,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Calculate metric scores; track any fail_fast metric that didn't score 1.0.
        # Duplicate metric names get auto-suffixed so every invocation contributes
        # to the composite (see _unique_metric_key).
        metric_scores = {}
        fail_fast_tripped = False
        for metric in metrics:
            key = _unique_metric_key(metric.name, metric_scores)
            try:
                score = metric.score(output, case.expected, case)
                metric_scores[key] = score
                if getattr(metric, "fail_fast", False) and score < 1.0:
                    fail_fast_tripped = True
            except Exception as e:
                metric_scores[key] = 0.0
                if getattr(metric, "fail_fast", False):
                    fail_fast_tripped = True
                if error is None:
                    error = f"Metric {metric.name} failed: {e}"

        # Calculate overall score (average of metrics)
        if metric_scores:
            overall_score = sum(metric_scores.values()) / len(metric_scores)
        else:
            overall_score = 1.0 if output else 0.0

        # Determine status. fail_fast metrics are hard gates — any score <1.0
        # forces FAILED regardless of composite.
        if error:
            status = EvalStatus.ERROR
        elif fail_fast_tripped:
            status = EvalStatus.FAILED
        elif overall_score >= self.threshold:
            status = EvalStatus.PASSED
        else:
            status = EvalStatus.FAILED

        # Propagate rubric scores from case metadata (set by RubricJudgeMetric)
        rubric_scores = {}
        rubric_band = None
        if "rubric_criteria" in case.metadata:
            criteria = case.metadata["rubric_criteria"]
            if isinstance(criteria, dict):
                rubric_scores = {
                    name: entry["score"] / 10.0
                    for name, entry in criteria.items()
                    if isinstance(entry, dict) and "score" in entry
                }
                rubric_band = _band_from_score(overall_score)

        return EvalResult(
            case=case,
            status=status,
            score=overall_score,
            output=output,
            metrics=metric_scores,
            error=error,
            latency_ms=latency_ms,
            rubric_scores=rubric_scores,
            rubric_band=rubric_band,
        )


class ProviderEvaluator(Evaluator):
    """Evaluator that uses a Gorgon provider."""

    def __init__(
        self,
        provider: Any,  # Provider from providers module
        system_prompt: str | None = None,
        threshold: float = 0.7,
    ):
        """Initialize with a provider.

        Args:
            provider: Gorgon Provider instance
            system_prompt: Optional system prompt
            threshold: Score threshold for passing
        """
        self.provider = provider
        self.system_prompt = system_prompt
        self.threshold = threshold

    def evaluate(
        self,
        case: EvalCase,
        metrics: list[EvalMetric],
    ) -> EvalResult:
        """Evaluate a single case."""
        import time

        from animus_forge.providers import CompletionRequest

        start_time = time.time()
        error = None
        output = None
        tokens_used = 0

        try:
            # Build request
            if isinstance(case.input, dict):
                prompt = case.input.get("prompt", str(case.input))
            else:
                prompt = str(case.input)

            request = CompletionRequest(
                prompt=prompt,
                system_prompt=self.system_prompt,
            )

            response = self.provider.complete(request)
            output = response.content
            tokens_used = response.tokens_used

        except Exception as e:
            error = str(e)
            return EvalResult(
                case=case,
                status=EvalStatus.ERROR,
                score=0.0,
                output=None,
                error=error,
                latency_ms=(time.time() - start_time) * 1000,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Calculate metric scores; track any fail_fast metric that didn't score 1.0.
        # Duplicate metric names get auto-suffixed so every invocation contributes
        # to the composite (see _unique_metric_key).
        metric_scores = {}
        fail_fast_tripped = False
        for metric in metrics:
            key = _unique_metric_key(metric.name, metric_scores)
            try:
                score = metric.score(output, case.expected, case)
                metric_scores[key] = score
                if getattr(metric, "fail_fast", False) and score < 1.0:
                    fail_fast_tripped = True
            except Exception as e:
                metric_scores[key] = 0.0
                if getattr(metric, "fail_fast", False):
                    fail_fast_tripped = True
                if error is None:
                    error = f"Metric {metric.name} failed: {e}"

        # Calculate overall score
        if metric_scores:
            overall_score = sum(metric_scores.values()) / len(metric_scores)
        else:
            overall_score = 1.0 if output else 0.0

        # Determine status. fail_fast metrics are hard gates — any score <1.0
        # forces FAILED regardless of composite.
        if error:
            status = EvalStatus.ERROR
        elif fail_fast_tripped:
            status = EvalStatus.FAILED
        elif overall_score >= self.threshold:
            status = EvalStatus.PASSED
        else:
            status = EvalStatus.FAILED

        # Propagate rubric scores from case metadata (set by RubricJudgeMetric)
        rubric_scores = {}
        rubric_band = None
        if "rubric_criteria" in case.metadata:
            criteria = case.metadata["rubric_criteria"]
            if isinstance(criteria, dict):
                rubric_scores = {
                    name: entry["score"] / 10.0
                    for name, entry in criteria.items()
                    if isinstance(entry, dict) and "score" in entry
                }
                rubric_band = _band_from_score(overall_score)

        return EvalResult(
            case=case,
            status=status,
            score=overall_score,
            output=output,
            metrics=metric_scores,
            error=error,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            rubric_scores=rubric_scores,
            rubric_band=rubric_band,
        )
