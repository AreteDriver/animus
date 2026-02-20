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
        }


class EvalMetric(ABC):
    """Abstract base class for evaluation metrics.

    Metrics score agent outputs on specific dimensions like
    accuracy, relevance, safety, etc.
    """

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

        # Calculate metric scores
        metric_scores = {}
        for metric in metrics:
            try:
                score = metric.score(output, case.expected, case)
                metric_scores[metric.name] = score
            except Exception as e:
                metric_scores[metric.name] = 0.0
                if error is None:
                    error = f"Metric {metric.name} failed: {e}"

        # Calculate overall score (average of metrics)
        if metric_scores:
            overall_score = sum(metric_scores.values()) / len(metric_scores)
        else:
            overall_score = 1.0 if output else 0.0

        # Determine status
        if error:
            status = EvalStatus.ERROR
        elif overall_score >= self.threshold:
            status = EvalStatus.PASSED
        else:
            status = EvalStatus.FAILED

        return EvalResult(
            case=case,
            status=status,
            score=overall_score,
            output=output,
            metrics=metric_scores,
            error=error,
            latency_ms=latency_ms,
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

        # Calculate metric scores
        metric_scores = {}
        for metric in metrics:
            try:
                score = metric.score(output, case.expected, case)
                metric_scores[metric.name] = score
            except Exception as e:
                metric_scores[metric.name] = 0.0
                if error is None:
                    error = f"Metric {metric.name} failed: {e}"

        # Calculate overall score
        if metric_scores:
            overall_score = sum(metric_scores.values()) / len(metric_scores)
        else:
            overall_score = 1.0 if output else 0.0

        # Determine status
        if error:
            status = EvalStatus.ERROR
        elif overall_score >= self.threshold:
            status = EvalStatus.PASSED
        else:
            status = EvalStatus.FAILED

        return EvalResult(
            case=case,
            status=status,
            score=overall_score,
            output=output,
            metrics=metric_scores,
            error=error,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
        )
