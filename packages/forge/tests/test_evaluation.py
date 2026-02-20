"""Tests for the evaluation module.

Comprehensively tests base classes, metrics, reporters, and runner
with all external dependencies mocked.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

from animus_forge.evaluation.base import (
    AgentEvaluator,
    EvalCase,
    EvalMetric,
    EvalResult,
    EvalStatus,
    EvalSuite,
    Evaluator,
    ProviderEvaluator,
)
from animus_forge.evaluation.metrics import (
    CodeExecutionMetric,
    CompositeMetric,
    ContainsMetric,
    ExactMatchMetric,
    FactualityMetric,
    LengthMetric,
    LLMJudgeMetric,
    RegexMatchMetric,
    SafetyMetric,
    SimilarityMetric,
)
from animus_forge.evaluation.reporters import (
    ConsoleReporter,
    HTMLReporter,
    JSONReporter,
    MarkdownReporter,
)
from animus_forge.evaluation.runner import (
    EvalRunner,
    SuiteResult,
    create_quick_eval,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case(
    input_val: str = "test input",
    expected: str | None = "test expected",
    name: str = "",
    **metadata,
) -> EvalCase:
    """Create an EvalCase with deterministic id."""
    case = EvalCase(
        input=input_val,
        expected=expected,
        name=name,
        metadata=metadata,
    )
    return case


def _make_result(
    status: EvalStatus = EvalStatus.PASSED,
    score: float = 1.0,
    output: str = "output",
    error: str | None = None,
    latency_ms: float = 100.0,
    tokens_used: int = 50,
    metrics: dict | None = None,
    case: EvalCase | None = None,
) -> EvalResult:
    """Create an EvalResult for testing."""
    return EvalResult(
        case=case or _make_case(),
        status=status,
        score=score,
        output=output,
        error=error,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        metrics=metrics or {},
    )


def _make_suite_result(
    suite: EvalSuite | None = None,
    results: list[EvalResult] | None = None,
    passed: int = 2,
    failed: int = 1,
    errors: int = 0,
    skipped: int = 0,
    total_score: float = 0.75,
    duration_ms: float = 500.0,
) -> SuiteResult:
    """Create a SuiteResult for testing reporters."""
    suite = suite or EvalSuite(name="test_suite", threshold=0.7)
    return SuiteResult(
        suite=suite,
        results=results or [],
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        total_score=total_score,
        duration_ms=duration_ms,
    )


class _DummyMetric(EvalMetric):
    """Concrete metric for testing abstract base."""

    @property
    def name(self) -> str:
        return "dummy"

    def score(self, output, expected, case):
        return 1.0


class _FailingMetric(EvalMetric):
    """Metric that always raises."""

    @property
    def name(self) -> str:
        return "failing"

    def score(self, output, expected, case):
        raise ValueError("metric failed")


class _DummyEvaluator(Evaluator):
    """Concrete evaluator for testing abstract base."""

    def evaluate(self, case, metrics):
        return EvalResult(
            case=case,
            status=EvalStatus.PASSED,
            score=1.0,
            output="dummy output",
        )


# ===========================================================================
# base.py Tests
# ===========================================================================


class TestEvalStatus:
    """Tests for EvalStatus enum."""

    def test_all_statuses_exist(self):
        assert EvalStatus.PASSED.value == "passed"
        assert EvalStatus.FAILED.value == "failed"
        assert EvalStatus.ERROR.value == "error"
        assert EvalStatus.SKIPPED.value == "skipped"

    def test_values(self):
        assert EvalStatus.PASSED.value == "passed"
        assert EvalStatus.FAILED.value == "failed"
        assert EvalStatus.ERROR.value == "error"
        assert EvalStatus.SKIPPED.value == "skipped"


class TestEvalCase:
    """Tests for EvalCase dataclass."""

    def test_defaults(self):
        case = EvalCase(input="hello")
        assert case.input == "hello"
        assert case.expected is None
        assert case.metadata == {}
        assert len(case.id) == 8
        assert case.name == f"case_{case.id}"

    def test_custom_values(self):
        case = EvalCase(
            input="prompt",
            expected="answer",
            metadata={"difficulty": "hard"},
            id="abc12345",
            name="my_case",
        )
        assert case.input == "prompt"
        assert case.expected == "answer"
        assert case.metadata == {"difficulty": "hard"}
        assert case.id == "abc12345"
        assert case.name == "my_case"

    def test_auto_name_from_id(self):
        case = EvalCase(input="x", id="deadbeef")
        assert case.name == "case_deadbeef"

    def test_dict_input(self):
        case = EvalCase(input={"prompt": "hello", "context": "world"})
        assert case.input["prompt"] == "hello"

    def test_dict_expected(self):
        case = EvalCase(input="q", expected={"key": "val"})
        assert case.expected["key"] == "val"

    def test_unique_ids(self):
        ids = {EvalCase(input="x").id for _ in range(20)}
        assert len(ids) == 20


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_defaults(self):
        case = _make_case()
        result = EvalResult(case=case, status=EvalStatus.PASSED, score=1.0, output="ok")
        assert result.error is None
        assert result.latency_ms == 0
        assert result.tokens_used == 0
        assert isinstance(result.timestamp, datetime)
        assert result.metadata == {}

    def test_passed_property_true(self):
        result = _make_result(status=EvalStatus.PASSED)
        assert result.passed is True

    def test_passed_property_false(self):
        for status in [EvalStatus.FAILED, EvalStatus.ERROR, EvalStatus.SKIPPED]:
            result = _make_result(status=status)
            assert result.passed is False

    def test_to_dict(self):
        result = _make_result(
            score=0.85,
            output="some output",
            error="some error",
            latency_ms=123.4,
            tokens_used=42,
            metrics={"accuracy": 0.9},
        )
        d = result.to_dict()
        assert d["case_id"] == result.case.id
        assert d["case_name"] == result.case.name
        assert d["status"] == "passed"
        assert d["score"] == 0.85
        assert d["metrics"] == {"accuracy": 0.9}
        assert d["error"] == "some error"
        assert d["latency_ms"] == 123.4
        assert d["tokens_used"] == 42
        assert isinstance(d["timestamp"], str)

    def test_to_dict_truncates_output(self):
        long_output = "a" * 600
        result = _make_result(output=long_output)
        d = result.to_dict()
        assert len(d["output"]) == 500

    def test_to_dict_short_output(self):
        result = _make_result(output="short")
        d = result.to_dict()
        assert d["output"] == "short"


class TestEvalMetric:
    """Tests for abstract EvalMetric."""

    def test_str_returns_name(self):
        metric = _DummyMetric()
        assert str(metric) == "dummy"


class TestEvalSuite:
    """Tests for EvalSuite dataclass."""

    def test_defaults(self):
        suite = EvalSuite(name="my_suite")
        assert suite.name == "my_suite"
        assert suite.cases == []
        assert suite.metrics == []
        assert suite.description == ""
        assert suite.tags == []
        assert suite.threshold == 0.7

    def test_add_case(self):
        suite = EvalSuite(name="s")
        case = suite.add_case(input="hello", expected="world", name="c1")
        assert len(suite.cases) == 1
        assert case.input == "hello"
        assert case.expected == "world"
        assert case.name == "c1"

    def test_add_case_with_metadata(self):
        suite = EvalSuite(name="s")
        case = suite.add_case(input="x", tags=["math"], difficulty="easy")
        assert case.metadata["tags"] == ["math"]
        assert case.metadata["difficulty"] == "easy"

    def test_add_metric(self):
        suite = EvalSuite(name="s")
        metric = _DummyMetric()
        suite.add_metric(metric)
        assert suite.metrics == [metric]

    def test_filter_by_tag(self):
        suite = EvalSuite(name="s")
        suite.add_case(input="a", tags=["math"])
        suite.add_case(input="b", tags=["science"])
        suite.add_case(input="c", tags=["math", "science"])
        math_cases = suite.filter_by_tag("math")
        assert len(math_cases) == 2

    def test_filter_by_tag_no_match(self):
        suite = EvalSuite(name="s")
        suite.add_case(input="a", tags=["math"])
        assert suite.filter_by_tag("history") == []

    def test_filter_by_tag_no_tags_metadata(self):
        suite = EvalSuite(name="s")
        suite.add_case(input="a")
        assert suite.filter_by_tag("math") == []

    def test_from_yaml(self):
        yaml_data = {
            "name": "yaml_suite",
            "description": "A test suite",
            "tags": ["integration"],
            "threshold": 0.8,
            "cases": [
                {"input": "q1", "expected": "a1", "name": "case_1"},
                {"input": "q2", "name": "case_2", "metadata": {"tags": ["hard"]}},
            ],
        }
        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", return_value=yaml_data),
        ):
            suite = EvalSuite.from_yaml("/fake/path.yaml")
        assert suite.name == "yaml_suite"
        assert suite.description == "A test suite"
        assert suite.tags == ["integration"]
        assert suite.threshold == 0.8
        assert len(suite.cases) == 2
        assert suite.cases[0].name == "case_1"
        assert suite.cases[1].metadata.get("tags") == ["hard"]

    def test_from_yaml_minimal(self):
        yaml_data = {}
        with (
            patch("builtins.open", mock_open()),
            patch("yaml.safe_load", return_value=yaml_data),
        ):
            suite = EvalSuite.from_yaml("/fake/path.yaml")
        assert suite.name == "unnamed"
        assert suite.cases == []

    def test_to_yaml(self):
        suite = EvalSuite(name="s", description="d", tags=["t"], threshold=0.9)
        suite.add_case(input="q", expected="a", name="c1")

        written_data = {}

        def mock_dump(data, f, **kwargs):
            written_data.update(data)

        with (
            patch("builtins.open", mock_open()),
            patch("yaml.dump", side_effect=mock_dump),
        ):
            suite.to_yaml("/fake/path.yaml")

        assert written_data["name"] == "s"
        assert written_data["description"] == "d"
        assert written_data["threshold"] == 0.9
        assert len(written_data["cases"]) == 1
        assert written_data["cases"][0]["input"] == "q"


class TestEvaluatorAbstract:
    """Tests for abstract Evaluator base class."""

    def test_evaluate_async_default(self):
        """Default evaluate_async wraps sync evaluate."""
        evaluator = _DummyEvaluator()
        case = _make_case()

        async def _test():
            result = await evaluator.evaluate_async(case, [])
            assert result.status == EvalStatus.PASSED
            assert result.output == "dummy output"

        asyncio.run(_test())


class TestAgentEvaluator:
    """Tests for AgentEvaluator."""

    def test_init(self):
        fn = MagicMock(return_value="output")
        evaluator = AgentEvaluator(fn, threshold=0.8)
        assert evaluator.agent_fn is fn
        assert evaluator.threshold == 0.8

    def test_evaluate_success_with_metrics(self):
        fn = MagicMock(return_value="hello world")
        evaluator = AgentEvaluator(fn, threshold=0.5)
        case = _make_case(expected="hello world")
        metric = _DummyMetric()
        result = evaluator.evaluate(case, [metric])
        assert result.status == EvalStatus.PASSED
        assert result.score == 1.0
        assert result.metrics == {"dummy": 1.0}
        assert result.output == "hello world"
        assert result.latency_ms > 0

    def test_evaluate_agent_exception(self):
        fn = MagicMock(side_effect=RuntimeError("boom"))
        evaluator = AgentEvaluator(fn)
        case = _make_case()
        result = evaluator.evaluate(case, [_DummyMetric()])
        assert result.status == EvalStatus.ERROR
        assert result.score == 0.0
        assert "boom" in result.error
        assert result.output is None

    def test_evaluate_metric_exception(self):
        fn = MagicMock(return_value="output")
        evaluator = AgentEvaluator(fn)
        case = _make_case()
        result = evaluator.evaluate(case, [_FailingMetric()])
        assert result.status == EvalStatus.ERROR
        assert result.metrics == {"failing": 0.0}
        assert "Metric failing failed" in result.error

    def test_evaluate_below_threshold(self):
        """Score below threshold = FAILED."""

        class HalfMetric(EvalMetric):
            @property
            def name(self):
                return "half"

            def score(self, output, expected, case):
                return 0.5

        fn = MagicMock(return_value="output")
        evaluator = AgentEvaluator(fn, threshold=0.8)
        result = evaluator.evaluate(_make_case(), [HalfMetric()])
        assert result.status == EvalStatus.FAILED
        assert result.score == 0.5

    def test_evaluate_no_metrics_with_output(self):
        fn = MagicMock(return_value="something")
        evaluator = AgentEvaluator(fn)
        result = evaluator.evaluate(_make_case(), [])
        assert result.score == 1.0
        assert result.status == EvalStatus.PASSED

    def test_evaluate_no_metrics_no_output(self):
        fn = MagicMock(return_value="")
        evaluator = AgentEvaluator(fn)
        result = evaluator.evaluate(_make_case(), [])
        assert result.score == 0.0
        assert result.status == EvalStatus.FAILED

    def test_evaluate_no_metrics_none_output(self):
        fn = MagicMock(return_value=None)
        evaluator = AgentEvaluator(fn)
        result = evaluator.evaluate(_make_case(), [])
        assert result.score == 0.0

    def test_evaluate_multiple_metrics_averaged(self):
        class ScoreMetric(EvalMetric):
            def __init__(self, val, n):
                self._val = val
                self._n = n

            @property
            def name(self):
                return self._n

            def score(self, output, expected, case):
                return self._val

        fn = MagicMock(return_value="out")
        evaluator = AgentEvaluator(fn, threshold=0.0)
        m1 = ScoreMetric(0.8, "m1")
        m2 = ScoreMetric(0.6, "m2")
        result = evaluator.evaluate(_make_case(), [m1, m2])
        assert abs(result.score - 0.7) < 1e-9

    def test_evaluate_metric_error_sets_error_only_once(self):
        """Only the first metric failure sets the error."""
        fn = MagicMock(return_value="out")
        evaluator = AgentEvaluator(fn)
        result = evaluator.evaluate(_make_case(), [_FailingMetric(), _FailingMetric()])
        # error from first failing metric
        assert "Metric failing failed" in result.error


class TestProviderEvaluator:
    """Tests for ProviderEvaluator."""

    def _mock_provider(self, content="response", tokens=10):
        provider = MagicMock()
        response = MagicMock()
        response.content = content
        response.tokens_used = tokens
        provider.complete.return_value = response
        return provider

    def test_init(self):
        provider = MagicMock()
        evaluator = ProviderEvaluator(provider, system_prompt="sys", threshold=0.9)
        assert evaluator.provider is provider
        assert evaluator.system_prompt == "sys"
        assert evaluator.threshold == 0.9

    @patch("animus_forge.evaluation.base.CompletionRequest", create=True)
    def test_evaluate_success(self, _mock_cr):
        provider = self._mock_provider(content="hello world", tokens=25)
        evaluator = ProviderEvaluator(provider, system_prompt="be helpful")
        case = _make_case(input_val="say hello")
        with patch(
            "animus_forge.providers.CompletionRequest",
            return_value=MagicMock(),
        ):
            result = evaluator.evaluate(case, [_DummyMetric()])
        assert result.status == EvalStatus.PASSED
        assert result.output == "hello world"
        assert result.tokens_used == 25

    @patch("animus_forge.evaluation.base.CompletionRequest", create=True)
    def test_evaluate_dict_input(self, _mock_cr):
        provider = self._mock_provider()
        evaluator = ProviderEvaluator(provider)
        case = _make_case(input_val={"prompt": "hello"})
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            result = evaluator.evaluate(case, [])
        assert result.output == "response"

    @patch("animus_forge.evaluation.base.CompletionRequest", create=True)
    def test_evaluate_dict_input_no_prompt_key(self, _mock_cr):
        provider = self._mock_provider()
        evaluator = ProviderEvaluator(provider)
        case = _make_case(input_val={"context": "world"})
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            result = evaluator.evaluate(case, [])
        assert result.output == "response"

    def test_evaluate_provider_exception(self):
        provider = MagicMock()
        provider.complete.side_effect = RuntimeError("API error")
        evaluator = ProviderEvaluator(provider)
        case = _make_case()
        result = evaluator.evaluate(case, [_DummyMetric()])
        assert result.status == EvalStatus.ERROR
        assert "API error" in result.error
        assert result.output is None

    @patch("animus_forge.evaluation.base.CompletionRequest", create=True)
    def test_evaluate_metric_failure(self, _mock_cr):
        provider = self._mock_provider()
        evaluator = ProviderEvaluator(provider)
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            result = evaluator.evaluate(_make_case(), [_FailingMetric()])
        assert result.status == EvalStatus.ERROR
        assert result.metrics["failing"] == 0.0

    @patch("animus_forge.evaluation.base.CompletionRequest", create=True)
    def test_evaluate_below_threshold(self, _mock_cr):
        class ZeroMetric(EvalMetric):
            @property
            def name(self):
                return "zero"

            def score(self, output, expected, case):
                return 0.0

        provider = self._mock_provider()
        evaluator = ProviderEvaluator(provider, threshold=0.5)
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            result = evaluator.evaluate(_make_case(), [ZeroMetric()])
        assert result.status == EvalStatus.FAILED

    @patch("animus_forge.evaluation.base.CompletionRequest", create=True)
    def test_evaluate_no_metrics_with_output(self, _mock_cr):
        provider = self._mock_provider(content="hello")
        evaluator = ProviderEvaluator(provider)
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            result = evaluator.evaluate(_make_case(), [])
        assert result.score == 1.0

    @patch("animus_forge.evaluation.base.CompletionRequest", create=True)
    def test_evaluate_no_metrics_empty_output(self, _mock_cr):
        provider = self._mock_provider(content="")
        evaluator = ProviderEvaluator(provider)
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            result = evaluator.evaluate(_make_case(), [])
        assert result.score == 0.0


# ===========================================================================
# metrics.py Tests
# ===========================================================================


class TestExactMatchMetric:
    """Tests for ExactMatchMetric."""

    def test_name(self):
        assert ExactMatchMetric().name == "exact_match"

    def test_exact_match_case_insensitive(self):
        m = ExactMatchMetric(case_sensitive=False)
        case = _make_case()
        assert m.score("Hello", "hello", case) == 1.0

    def test_exact_match_case_sensitive(self):
        m = ExactMatchMetric(case_sensitive=True)
        case = _make_case()
        assert m.score("Hello", "hello", case) == 0.0
        assert m.score("hello", "hello", case) == 1.0

    def test_strip_whitespace(self):
        m = ExactMatchMetric(strip=True)
        assert m.score("  hello  ", "hello", _make_case()) == 1.0

    def test_no_strip(self):
        m = ExactMatchMetric(strip=False)
        assert m.score("  hello  ", "hello", _make_case()) == 0.0

    def test_mismatch(self):
        m = ExactMatchMetric()
        assert m.score("foo", "bar", _make_case()) == 0.0

    def test_none_expected_with_output(self):
        m = ExactMatchMetric()
        assert m.score("something", None, _make_case()) == 1.0

    def test_none_expected_empty_output(self):
        m = ExactMatchMetric()
        assert m.score("", None, _make_case()) == 0.0

    def test_non_string_coercion(self):
        m = ExactMatchMetric()
        assert m.score(42, "42", _make_case()) == 1.0


class TestContainsMetric:
    """Tests for ContainsMetric."""

    def test_name(self):
        assert ContainsMetric().name == "contains"

    def test_contains_single_string(self):
        m = ContainsMetric()
        assert m.score("hello world", "hello", _make_case()) == 1.0

    def test_not_contains(self):
        m = ContainsMetric()
        assert m.score("hello", "world", _make_case()) == 0.0

    def test_case_insensitive(self):
        m = ContainsMetric(case_sensitive=False)
        assert m.score("Hello World", "hello", _make_case()) == 1.0

    def test_case_sensitive(self):
        m = ContainsMetric(case_sensitive=True)
        assert m.score("Hello World", "hello", _make_case()) == 0.0

    def test_list_all_required_pass(self):
        m = ContainsMetric(all_required=True)
        assert m.score("hello world foo", ["hello", "world"], _make_case()) == 1.0

    def test_list_all_required_fail(self):
        m = ContainsMetric(all_required=True)
        assert m.score("hello bar", ["hello", "world"], _make_case()) == 0.0

    def test_list_any_required(self):
        m = ContainsMetric(all_required=False)
        assert m.score("hello bar", ["hello", "world"], _make_case()) == 0.5

    def test_list_any_none_match(self):
        m = ContainsMetric(all_required=False)
        assert m.score("bar", ["hello", "world"], _make_case()) == 0.0

    def test_none_expected_with_output(self):
        m = ContainsMetric()
        assert m.score("something", None, _make_case()) == 1.0

    def test_none_expected_empty_output(self):
        m = ContainsMetric()
        assert m.score("", None, _make_case()) == 0.0


class TestRegexMatchMetric:
    """Tests for RegexMatchMetric."""

    def test_name(self):
        assert RegexMatchMetric().name == "regex_match"

    def test_pattern_from_init(self):
        m = RegexMatchMetric(pattern=r"\d+")
        assert m.score("abc123", "anything", _make_case()) == 1.0

    def test_pattern_from_expected(self):
        m = RegexMatchMetric()
        assert m.score("abc123", r"\d+", _make_case()) == 1.0

    def test_no_match(self):
        m = RegexMatchMetric(pattern=r"^\d+$")
        # When expected is provided, pattern = self._pattern or str(expected)
        assert m.score("abc", r"^\d+$", _make_case()) == 0.0

    def test_none_expected_uses_none_pattern(self):
        """Due to operator precedence, expected=None means pattern=None regardless of init."""
        m = RegexMatchMetric(pattern=r"\d+")
        # `(self._pattern or str(expected)) if expected else None` -> None
        assert m.score("something", None, _make_case()) == 1.0

    def test_none_pattern_and_expected_with_output(self):
        m = RegexMatchMetric()
        assert m.score("something", None, _make_case()) == 1.0

    def test_none_pattern_and_expected_empty_output(self):
        m = RegexMatchMetric()
        assert m.score("", None, _make_case()) == 0.0

    def test_invalid_regex_from_expected(self):
        m = RegexMatchMetric()
        assert m.score("test", r"[invalid", _make_case()) == 0.0

    def test_invalid_regex_from_init(self):
        m = RegexMatchMetric(pattern=r"[invalid")
        # init pattern used when expected is truthy
        assert m.score("test", "something", _make_case()) == 0.0

    def test_complex_pattern(self):
        m = RegexMatchMetric(pattern=r"^[A-Z]\w+\s\w+$")
        # Must pass expected to use the init pattern
        assert m.score("Hello World", "dummy", _make_case()) == 1.0
        assert m.score("hello world", "dummy", _make_case()) == 0.0


class TestSimilarityMetric:
    """Tests for SimilarityMetric."""

    def test_name(self):
        assert SimilarityMetric().name == "similarity"

    def test_simple_similarity_identical(self):
        m = SimilarityMetric()
        assert m.score("hello world", "hello world", _make_case()) == 1.0

    def test_simple_similarity_partial(self):
        m = SimilarityMetric()
        score = m.score("hello world foo", "hello world bar", _make_case())
        assert 0.0 < score < 1.0

    def test_simple_similarity_no_overlap(self):
        m = SimilarityMetric()
        assert m.score("aaa bbb", "ccc ddd", _make_case()) == 0.0

    def test_none_expected_with_output(self):
        m = SimilarityMetric()
        assert m.score("something", None, _make_case()) == 1.0

    def test_none_expected_empty_output(self):
        m = SimilarityMetric()
        assert m.score("", None, _make_case()) == 0.0

    def test_simple_similarity_empty_input(self):
        m = SimilarityMetric()
        assert m.score("", "hello", _make_case()) == 0.0

    def test_simple_similarity_empty_expected(self):
        m = SimilarityMetric()
        assert m.score("hello", "", _make_case()) == 0.0

    def test_embedding_provider(self):
        provider = MagicMock()
        provider.embed.side_effect = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
        m = SimilarityMetric(embedding_provider=provider)
        assert m.score("a", "b", _make_case()) == 1.0

    def test_embedding_provider_orthogonal(self):
        provider = MagicMock()
        provider.embed.side_effect = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        m = SimilarityMetric(embedding_provider=provider)
        assert m.score("a", "b", _make_case()) == 0.0

    def test_embedding_provider_failure_falls_back(self):
        provider = MagicMock()
        provider.embed.side_effect = RuntimeError("embed failed")
        m = SimilarityMetric(embedding_provider=provider)
        score = m.score("hello world", "hello world", _make_case())
        assert score == 1.0

    def test_cosine_similarity_zero_norm(self):
        m = SimilarityMetric()
        assert m._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
        assert m._cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0


class TestLLMJudgeMetric:
    """Tests for LLMJudgeMetric."""

    def test_name(self):
        assert LLMJudgeMetric().name == "llm_judge"

    def test_no_provider_returns_half(self):
        m = LLMJudgeMetric()
        assert m.score("output", "expected", _make_case()) == 0.5

    def test_with_provider_numeric_response(self):
        provider = MagicMock()
        response = MagicMock()
        response.content = "8"
        provider.complete.return_value = response
        m = LLMJudgeMetric(judge_provider=provider)
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            score = m.score("output", "expected", _make_case())
        assert score == 0.8

    def test_with_provider_clamps_score(self):
        provider = MagicMock()
        response = MagicMock()
        response.content = "15"
        provider.complete.return_value = response
        m = LLMJudgeMetric(judge_provider=provider)
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            score = m.score("out", "exp", _make_case())
        assert score == 1.0

    def test_with_provider_no_numeric(self):
        provider = MagicMock()
        response = MagicMock()
        response.content = "good"
        provider.complete.return_value = response
        m = LLMJudgeMetric(judge_provider=provider)
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            score = m.score("out", "exp", _make_case())
        assert score == 0.5

    def test_with_provider_exception(self):
        provider = MagicMock()
        provider.complete.side_effect = RuntimeError("fail")
        m = LLMJudgeMetric(judge_provider=provider)
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            score = m.score("out", "exp", _make_case())
        assert score == 0.5

    def test_custom_prompt(self):
        provider = MagicMock()
        response = MagicMock()
        response.content = "7"
        provider.complete.return_value = response
        m = LLMJudgeMetric(
            judge_provider=provider,
            prompt_template="Rate: {input} {expected} {output}",
        )
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            score = m.score("out", "exp", _make_case())
        assert score == 0.7

    def test_custom_criteria(self):
        m = LLMJudgeMetric(criteria=["safety", "accuracy"])
        assert m._criteria == ["safety", "accuracy"]


class TestCodeExecutionMetric:
    """Tests for CodeExecutionMetric."""

    def test_name(self):
        assert CodeExecutionMetric().name == "code_execution"

    def test_non_python_language(self):
        m = CodeExecutionMetric(language="javascript")
        assert m.score("console.log(1)", "1", _make_case()) == 0.5

    def test_no_code_found(self):
        m = CodeExecutionMetric()
        assert m.score("just plain text", None, _make_case()) == 0.0

    def test_extract_code_from_markdown(self):
        m = CodeExecutionMetric()
        text = '```python\nprint("hello")\n```'
        assert m._extract_code(text) == 'print("hello")\n'

    def test_extract_code_no_language(self):
        m = CodeExecutionMetric()
        text = '```\nprint("hello")\n```'
        assert m._extract_code(text) == 'print("hello")\n'

    def test_extract_code_bare_python(self):
        m = CodeExecutionMetric()
        assert "import os" in m._extract_code("import os\nprint(1)")

    def test_extract_code_def_keyword(self):
        m = CodeExecutionMetric()
        code = "def foo():\n    pass"
        assert m._extract_code(code) == code

    def test_extract_code_print_keyword(self):
        m = CodeExecutionMetric()
        code = 'print("hello")'
        assert m._extract_code(code) == code

    def test_extract_code_no_markers(self):
        m = CodeExecutionMetric()
        assert m._extract_code("no code here at all") == ""

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_execute_python_success(self, mock_temp, mock_run):
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.name = "/tmp/test.py"
        mock_temp.return_value = mock_file

        mock_run.return_value = MagicMock(stdout="hello\n", stderr="")

        m = CodeExecutionMetric()
        code = 'print("hello")'
        output = "```python\n" + code + "\n```"
        assert m.score(output, "hello", _make_case()) == 1.0

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_execute_python_no_expected(self, mock_temp, mock_run):
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.name = "/tmp/test.py"
        mock_temp.return_value = mock_file
        mock_run.return_value = MagicMock(stdout="ok\n", stderr="")

        m = CodeExecutionMetric()
        assert m.score("```python\nprint(1)\n```", None, _make_case()) == 1.0

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_execute_python_mismatch(self, mock_temp, mock_run):
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.name = "/tmp/test.py"
        mock_temp.return_value = mock_file
        mock_run.return_value = MagicMock(stdout="wrong\n", stderr="")

        m = CodeExecutionMetric()
        assert m.score("```python\nprint(1)\n```", "hello", _make_case()) == 0.0

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_execute_python_exception(self, mock_temp, mock_run):
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.name = "/tmp/test.py"
        mock_temp.return_value = mock_file
        mock_run.side_effect = OSError("no python")

        m = CodeExecutionMetric()
        assert m.score("```python\nprint(1)\n```", "1", _make_case()) == 0.0

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_execute_python_timeout(self, mock_temp, mock_run):
        import subprocess

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.name = "/tmp/test.py"
        mock_temp.return_value = mock_file
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="python", timeout=10)

        m = CodeExecutionMetric()
        result = m._execute_python("while True: pass")
        assert result == "TIMEOUT"

    def test_expected_output_override(self):
        m = CodeExecutionMetric(expected_output="custom")
        assert m._expected_output == "custom"


class TestFactualityMetric:
    """Tests for FactualityMetric."""

    def test_name(self):
        assert FactualityMetric().name == "factuality"

    def test_no_facts_no_provider(self):
        m = FactualityMetric()
        assert m.score("anything", None, _make_case()) == 0.5

    def test_facts_all_correct(self):
        facts = {"paris": "france", "berlin": "germany"}
        m = FactualityMetric(facts=facts)
        output = "paris is in france, berlin is in germany"
        assert m.score(output, None, _make_case()) == 1.0

    def test_facts_partial(self):
        facts = {"paris": "france", "berlin": "germany"}
        m = FactualityMetric(facts=facts)
        output = "paris is in france, berlin is in italy"
        assert m.score(output, None, _make_case()) == 0.5

    def test_facts_none_mentioned(self):
        facts = {"paris": "france"}
        m = FactualityMetric(facts=facts)
        output = "the weather is nice"
        assert m.score(output, None, _make_case()) == 0.5

    def test_facts_key_mentioned_value_wrong(self):
        facts = {"paris": "france"}
        m = FactualityMetric(facts=facts)
        output = "paris is in spain"
        assert m.score(output, None, _make_case()) == 0.0

    def test_llm_verifier(self):
        provider = MagicMock()
        response = MagicMock()
        response.content = "9"
        provider.complete.return_value = response
        m = FactualityMetric(verifier_provider=provider)
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            score = m.score("output", None, _make_case())
        assert score == 0.9

    def test_llm_verifier_no_numeric(self):
        provider = MagicMock()
        response = MagicMock()
        response.content = "not a number"
        provider.complete.return_value = response
        m = FactualityMetric(verifier_provider=provider)
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            score = m.score("output", None, _make_case())
        assert score == 0.5

    def test_llm_verifier_exception(self):
        provider = MagicMock()
        provider.complete.side_effect = RuntimeError("fail")
        m = FactualityMetric(verifier_provider=provider)
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            score = m.score("output", None, _make_case())
        assert score == 0.5


class TestSafetyMetric:
    """Tests for SafetyMetric."""

    def test_name(self):
        assert SafetyMetric().name == "safety"

    def test_safe_output(self):
        m = SafetyMetric()
        assert m.score("The capital of France is Paris", None, _make_case()) == 1.0

    def test_unsafe_bomb(self):
        m = SafetyMetric()
        assert m.score("how to make a bomb", None, _make_case()) == 0.0

    def test_unsafe_weapon(self):
        m = SafetyMetric()
        assert m.score("how to build a weapon", None, _make_case()) == 0.0

    def test_unsafe_violence(self):
        m = SafetyMetric()
        assert m.score("kill someone", None, _make_case()) == 0.0

    def test_additional_patterns(self):
        m = SafetyMetric(additional_patterns=[r"forbidden_word"])
        assert m.score("this has forbidden_word", None, _make_case()) == 0.0
        assert m.score("this is safe", None, _make_case()) == 1.0

    def test_safety_provider_safe(self):
        provider = MagicMock()
        response = MagicMock()
        response.content = "10"
        provider.complete.return_value = response
        m = SafetyMetric(safety_provider=provider)
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            score = m.score("safe content", None, _make_case())
        assert score == 1.0

    def test_safety_provider_no_numeric(self):
        provider = MagicMock()
        response = MagicMock()
        response.content = "unclear"
        provider.complete.return_value = response
        m = SafetyMetric(safety_provider=provider)
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            score = m.score("content", None, _make_case())
        assert score == 0.5

    def test_safety_provider_exception(self):
        provider = MagicMock()
        provider.complete.side_effect = RuntimeError("fail")
        m = SafetyMetric(safety_provider=provider)
        with patch("animus_forge.providers.CompletionRequest", return_value=MagicMock()):
            score = m.score("content", None, _make_case())
        assert score == 0.5

    def test_unsafe_pattern_overrides_provider(self):
        """Pattern match is checked first; provider is only called if patterns pass."""
        provider = MagicMock()
        m = SafetyMetric(safety_provider=provider)
        score = m.score("how to make a bomb", None, _make_case())
        assert score == 0.0
        provider.complete.assert_not_called()


class TestLengthMetric:
    """Tests for LengthMetric."""

    def test_name(self):
        assert LengthMetric().name == "length"

    def test_in_range(self):
        m = LengthMetric(min_length=5, max_length=20)
        assert m.score("hello world", None, _make_case()) == 1.0

    def test_below_min(self):
        m = LengthMetric(min_length=10)
        score = m.score("hi", None, _make_case())
        assert score == 2 / 10  # len("hi") = 2

    def test_below_min_zero(self):
        m = LengthMetric(min_length=0)
        assert m.score("", None, _make_case()) == 1.0

    def test_above_max(self):
        m = LengthMetric(max_length=5)
        score = m.score("hello world", None, _make_case())
        assert score == 5 / 11  # len("hello world") = 11

    def test_words_unit(self):
        m = LengthMetric(min_length=2, max_length=5, unit="words")
        assert m.score("hello world foo", None, _make_case()) == 1.0

    def test_words_unit_below(self):
        m = LengthMetric(min_length=5, unit="words")
        score = m.score("hello world", None, _make_case())
        assert score == 2 / 5

    def test_sentences_unit(self):
        m = LengthMetric(min_length=2, max_length=5, unit="sentences")
        assert m.score("Hello. World!", None, _make_case()) == 1.0

    def test_unknown_unit_defaults_to_chars(self):
        m = LengthMetric(min_length=1, max_length=100, unit="paragraphs")
        assert m.score("hello", None, _make_case()) == 1.0

    def test_no_max(self):
        m = LengthMetric(min_length=1)
        assert m.score("x" * 10000, None, _make_case()) == 1.0


class TestCompositeMetric:
    """Tests for CompositeMetric."""

    def test_name_default(self):
        m = CompositeMetric(metrics=[], name="composite")
        assert m.name == "composite"

    def test_custom_name(self):
        m = CompositeMetric(metrics=[], name="custom")
        assert m.name == "custom"

    def test_weighted_average(self):
        class FixedMetric(EvalMetric):
            def __init__(self, val, n):
                self._val = val
                self._n = n

            @property
            def name(self):
                return self._n

            def score(self, output, expected, case):
                return self._val

        m = CompositeMetric(
            metrics=[
                (FixedMetric(1.0, "a"), 3.0),
                (FixedMetric(0.0, "b"), 1.0),
            ]
        )
        score = m.score("out", "exp", _make_case())
        assert abs(score - 0.75) < 1e-9

    def test_empty_metrics(self):
        m = CompositeMetric(metrics=[])
        assert m.score("out", "exp", _make_case()) == 0.0

    def test_metric_failure_skipped(self):
        m = CompositeMetric(
            metrics=[
                (_DummyMetric(), 1.0),
                (_FailingMetric(), 1.0),
            ]
        )
        score = m.score("out", "exp", _make_case())
        assert score == 1.0  # only dummy metric counted

    def test_all_metrics_fail(self):
        m = CompositeMetric(
            metrics=[
                (_FailingMetric(), 1.0),
                (_FailingMetric(), 1.0),
            ]
        )
        assert m.score("out", "exp", _make_case()) == 0.0


# ===========================================================================
# runner.py Tests
# ===========================================================================


class TestSuiteResult:
    """Tests for SuiteResult dataclass."""

    def test_defaults(self):
        suite = EvalSuite(name="s")
        sr = SuiteResult(suite=suite)
        assert sr.results == []
        assert sr.passed == 0
        assert sr.failed == 0
        assert sr.errors == 0
        assert sr.skipped == 0
        assert sr.total_score == 0.0
        assert sr.duration_ms == 0
        assert isinstance(sr.timestamp, datetime)

    def test_total(self):
        sr = _make_suite_result(passed=3, failed=2, errors=1, skipped=1)
        assert sr.total == 7

    def test_pass_rate(self):
        sr = _make_suite_result(passed=3, failed=1, errors=0, skipped=0)
        assert sr.pass_rate == 0.75

    def test_pass_rate_zero_total(self):
        sr = _make_suite_result(passed=0, failed=0, errors=0, skipped=0)
        assert sr.pass_rate == 0.0

    def test_to_dict(self):
        suite = EvalSuite(name="my_suite")
        result = _make_result()
        sr = SuiteResult(
            suite=suite,
            results=[result],
            passed=1,
            total_score=0.9,
            duration_ms=100.0,
        )
        d = sr.to_dict()
        assert d["suite_name"] == "my_suite"
        assert d["passed"] == 1
        assert d["total"] == 1
        assert d["total_score"] == 0.9
        assert d["duration_ms"] == 100.0
        assert len(d["results"]) == 1
        assert isinstance(d["timestamp"], str)


class TestEvalRunner:
    """Tests for EvalRunner."""

    def _make_runner(self, evaluator=None, max_workers=2, progress_callback=None):
        evaluator = evaluator or _DummyEvaluator()
        return EvalRunner(
            evaluator=evaluator,
            max_workers=max_workers,
            progress_callback=progress_callback,
        )

    def _make_suite_with_cases(self, n=3):
        suite = EvalSuite(name="test_suite", threshold=0.5)
        suite.add_metric(_DummyMetric())
        for i in range(n):
            suite.add_case(input=f"input_{i}", expected=f"expected_{i}", name=f"case_{i}")
        return suite

    def test_run_sequential(self):
        runner = self._make_runner()
        suite = self._make_suite_with_cases(3)
        result = runner.run(suite, parallel=False)
        assert result.passed == 3
        assert result.failed == 0
        assert result.total == 3
        assert result.total_score == 1.0
        assert result.duration_ms > 0

    def test_run_parallel(self):
        runner = self._make_runner()
        suite = self._make_suite_with_cases(3)
        result = runner.run(suite, parallel=True)
        assert result.passed == 3
        assert result.total == 3

    def test_run_parallel_single_case_uses_sequential(self):
        runner = self._make_runner()
        suite = self._make_suite_with_cases(1)
        result = runner.run(suite, parallel=True)
        assert result.total == 1

    def test_run_with_filter_tags(self):
        suite = EvalSuite(name="s")
        suite.add_metric(_DummyMetric())
        suite.add_case(input="a", tags=["math"])
        suite.add_case(input="b", tags=["science"])
        suite.add_case(input="c", tags=["math"])
        runner = self._make_runner()
        result = runner.run(suite, filter_tags=["math"])
        assert result.total == 2

    def test_run_with_filter_tags_no_match(self):
        suite = EvalSuite(name="s")
        suite.add_case(input="a", tags=["math"])
        runner = self._make_runner()
        result = runner.run(suite, filter_tags=["history"])
        assert result.total == 0

    def test_run_progress_callback(self):
        callback_calls = []
        runner = self._make_runner(
            progress_callback=lambda current, total, result: callback_calls.append((current, total))
        )
        suite = self._make_suite_with_cases(3)
        runner.run(suite, parallel=False)
        assert len(callback_calls) == 3
        assert callback_calls[0] == (1, 3)
        assert callback_calls[2] == (3, 3)

    def test_run_progress_callback_parallel(self):
        callback_calls = []
        runner = self._make_runner(
            progress_callback=lambda current, total, result: callback_calls.append((current, total))
        )
        suite = self._make_suite_with_cases(3)
        runner.run(suite, parallel=True)
        assert len(callback_calls) == 3

    def test_run_evaluator_exception_in_sequential(self):
        evaluator = MagicMock()
        evaluator.evaluate.side_effect = RuntimeError("evaluator error")
        runner = self._make_runner(evaluator=evaluator)
        suite = self._make_suite_with_cases(2)
        result = runner.run(suite, parallel=False)
        assert result.errors == 2
        assert result.total == 2

    def test_run_evaluator_exception_in_parallel(self):
        evaluator = MagicMock()
        evaluator.evaluate.side_effect = RuntimeError("evaluator error")
        runner = self._make_runner(evaluator=evaluator)
        suite = self._make_suite_with_cases(2)
        result = runner.run(suite, parallel=True)
        assert result.errors == 2

    def test_aggregate_results_mixed(self):
        suite = EvalSuite(name="s")
        results = [
            _make_result(status=EvalStatus.PASSED, score=1.0),
            _make_result(status=EvalStatus.FAILED, score=0.3),
            _make_result(status=EvalStatus.ERROR, score=0.0),
            _make_result(status=EvalStatus.SKIPPED, score=0.0),
        ]
        runner = self._make_runner()
        sr = runner._aggregate_results(suite, results)
        assert sr.passed == 1
        assert sr.failed == 1
        assert sr.errors == 1
        assert sr.skipped == 1
        assert sr.total == 4
        assert abs(sr.total_score - 0.325) < 1e-9

    def test_aggregate_results_empty(self):
        suite = EvalSuite(name="s")
        runner = self._make_runner()
        sr = runner._aggregate_results(suite, [])
        assert sr.total == 0
        assert sr.total_score == 0.0

    def test_run_async_sequential(self):
        runner = self._make_runner()
        suite = self._make_suite_with_cases(2)

        async def _test():
            return await runner.run_async(suite, parallel=False)

        result = asyncio.run(_test())
        assert result.total == 2
        assert result.passed == 2

    def test_run_async_parallel(self):
        runner = self._make_runner()
        suite = self._make_suite_with_cases(3)

        async def _test():
            return await runner.run_async(suite, parallel=True)

        result = asyncio.run(_test())
        assert result.total == 3
        assert result.passed == 3

    def test_run_async_with_filter_tags(self):
        suite = EvalSuite(name="s")
        suite.add_metric(_DummyMetric())
        suite.add_case(input="a", tags=["math"])
        suite.add_case(input="b", tags=["science"])
        runner = self._make_runner()

        async def _test():
            return await runner.run_async(suite, filter_tags=["science"])

        result = asyncio.run(_test())
        assert result.total == 1

    def test_run_async_parallel_single_case_uses_sequential(self):
        runner = self._make_runner()
        suite = self._make_suite_with_cases(1)

        async def _test():
            return await runner.run_async(suite, parallel=True)

        result = asyncio.run(_test())
        assert result.total == 1

    def test_run_async_sequential_exception(self):
        evaluator = MagicMock()
        evaluator.evaluate_async = AsyncMock(side_effect=RuntimeError("async boom"))
        runner = self._make_runner(evaluator=evaluator)
        suite = self._make_suite_with_cases(2)

        async def _test():
            return await runner.run_async(suite, parallel=False)

        result = asyncio.run(_test())
        assert result.errors == 2

    def test_run_async_parallel_exception(self):
        evaluator = MagicMock()
        evaluator.evaluate_async = AsyncMock(side_effect=RuntimeError("async boom"))
        runner = self._make_runner(evaluator=evaluator)
        suite = self._make_suite_with_cases(3)

        async def _test():
            return await runner.run_async(suite, parallel=True)

        result = asyncio.run(_test())
        assert result.errors == 3

    def test_run_async_progress_callback(self):
        callback_calls = []
        runner = self._make_runner(progress_callback=lambda c, t, r: callback_calls.append((c, t)))
        suite = self._make_suite_with_cases(2)

        async def _test():
            return await runner.run_async(suite, parallel=False)

        asyncio.run(_test())
        assert len(callback_calls) == 2

    def test_run_async_parallel_progress_callback(self):
        callback_calls = []
        runner = self._make_runner(progress_callback=lambda c, t, r: callback_calls.append((c, t)))
        suite = self._make_suite_with_cases(3)

        async def _test():
            return await runner.run_async(suite, parallel=True)

        asyncio.run(_test())
        assert len(callback_calls) == 3

    def test_run_multiple_sequential(self):
        runner = self._make_runner()
        s1 = self._make_suite_with_cases(2)
        s2 = self._make_suite_with_cases(3)
        results = runner.run_multiple([s1, s2], parallel_suites=False)
        assert len(results) == 2
        assert results[0].total == 2
        assert results[1].total == 3

    def test_run_multiple_parallel(self):
        runner = self._make_runner()
        s1 = self._make_suite_with_cases(2)
        s2 = self._make_suite_with_cases(3)
        results = runner.run_multiple([s1, s2], parallel_suites=True)
        assert len(results) == 2

    def test_run_multiple_empty(self):
        runner = self._make_runner()
        results = runner.run_multiple([], parallel_suites=False)
        assert results == []


class TestCreateQuickEval:
    """Tests for create_quick_eval helper."""

    def test_basic(self):
        def agent_fn(x):
            return f"answer: {x}"

        result = create_quick_eval(
            agent_fn=agent_fn,
            cases=[
                {"input": "hello", "expected": "hello"},
            ],
        )
        assert isinstance(result, SuiteResult)
        assert result.total == 1

    def test_with_exact_metric(self):
        def agent_fn(x):
            return "Paris"

        result = create_quick_eval(
            agent_fn=agent_fn,
            cases=[{"input": "Capital of France?", "expected": "Paris"}],
            metrics=["exact"],
        )
        assert result.total == 1
        assert result.passed == 1

    def test_with_contains_metric(self):
        def agent_fn(x):
            return "The capital of France is Paris"

        result = create_quick_eval(
            agent_fn=agent_fn,
            cases=[{"input": "Capital of France?", "expected": "Paris"}],
            metrics=["contains"],
        )
        assert result.passed == 1

    def test_multiple_cases(self):
        def agent_fn(x):
            return x.upper()

        result = create_quick_eval(
            agent_fn=agent_fn,
            cases=[
                {"input": "hello", "expected": "HELLO", "name": "c1"},
                {"input": "world", "expected": "WORLD", "name": "c2"},
            ],
            metrics=["exact"],
        )
        assert result.passed == 2

    def test_unknown_metric_ignored(self):
        def agent_fn(x):
            return "output"

        result = create_quick_eval(
            agent_fn=agent_fn,
            cases=[{"input": "q", "expected": "output"}],
            metrics=["nonexistent"],
        )
        # No metrics applied, so score is based on output truthiness
        assert result.total == 1

    def test_default_metric_is_contains(self):
        def agent_fn(x):
            return "my answer includes hello and world"

        result = create_quick_eval(
            agent_fn=agent_fn,
            cases=[{"input": "q", "expected": "hello"}],
        )
        assert result.passed == 1


# ===========================================================================
# reporters.py Tests
# ===========================================================================


def _make_reporter_suite_result(
    pass_rate_above_threshold: bool = True,
    include_errors: bool = False,
    include_metrics: bool = False,
) -> SuiteResult:
    """Build a SuiteResult with case results for reporter testing."""
    suite = EvalSuite(name="reporter_test", threshold=0.7)
    results = []

    case1 = _make_case(input_val="q1", name="passed_case")
    r1 = _make_result(
        case=case1,
        status=EvalStatus.PASSED,
        score=1.0,
        output="correct answer",
        latency_ms=50.0,
        metrics={"accuracy": 1.0} if include_metrics else {},
    )
    results.append(r1)

    case2 = _make_case(input_val="q2", name="failed_case")
    r2 = _make_result(
        case=case2,
        status=EvalStatus.FAILED,
        score=0.3,
        output="wrong answer",
        latency_ms=75.0,
        metrics={"accuracy": 0.3} if include_metrics else {},
    )
    results.append(r2)

    passed = 1
    failed = 1
    errors = 0

    if include_errors:
        case3 = _make_case(input_val="q3", name="error_case")
        r3 = _make_result(
            case=case3,
            status=EvalStatus.ERROR,
            score=0.0,
            output=None,
            error="Something went wrong",
            latency_ms=10.0,
        )
        results.append(r3)
        errors = 1

    total_score = 0.65 if not include_errors else 0.43
    if pass_rate_above_threshold:
        passed = 2
        failed = 0
        total_score = 0.9

    return SuiteResult(
        suite=suite,
        results=results,
        passed=passed,
        failed=failed,
        errors=errors,
        total_score=total_score,
        duration_ms=250.0,
    )


class TestConsoleReporter:
    """Tests for ConsoleReporter."""

    def test_basic_report(self):
        sr = _make_reporter_suite_result()
        reporter = ConsoleReporter()
        report = reporter.report(sr)
        assert "reporter_test" in report
        assert "Summary:" in report
        assert "Total Cases:" in report
        assert "250ms" in report

    def test_verbose_shows_details(self):
        sr = _make_reporter_suite_result(include_metrics=True)
        reporter = ConsoleReporter(verbose=True)
        report = reporter.report(sr)
        assert "Case Details:" in report
        assert "passed_case" in report

    def test_show_output(self):
        sr = _make_reporter_suite_result()
        reporter = ConsoleReporter(verbose=True, show_output=True)
        report = reporter.report(sr)
        assert "Output:" in report

    def test_show_output_truncation(self):
        suite = EvalSuite(name="s", threshold=0.7)
        case = _make_case(name="long_output_case")
        result = _make_result(
            case=case,
            status=EvalStatus.FAILED,
            score=0.5,
            output="x" * 300,
        )
        sr = SuiteResult(
            suite=suite,
            results=[result],
            failed=1,
            total_score=0.5,
            duration_ms=10.0,
        )
        reporter = ConsoleReporter(verbose=True, show_output=True)
        report = reporter.report(sr)
        assert "..." in report

    def test_error_displayed(self):
        sr = _make_reporter_suite_result(include_errors=True)
        reporter = ConsoleReporter()
        report = reporter.report(sr)
        assert "Something went wrong" in report

    def test_metrics_displayed(self):
        sr = _make_reporter_suite_result(include_metrics=True, pass_rate_above_threshold=False)
        reporter = ConsoleReporter(verbose=True)
        report = reporter.report(sr)
        assert "accuracy" in report

    def test_passed_verdict(self):
        sr = _make_reporter_suite_result(pass_rate_above_threshold=True)
        reporter = ConsoleReporter()
        report = reporter.report(sr)
        assert "PASSED" in report

    def test_failed_verdict(self):
        sr = _make_reporter_suite_result(pass_rate_above_threshold=False)
        reporter = ConsoleReporter()
        report = reporter.report(sr)
        assert "FAILED" in report

    def test_pct_zero_total(self):
        reporter = ConsoleReporter()
        assert reporter._pct(0, 0) == "0%"

    def test_pct_normal(self):
        reporter = ConsoleReporter()
        assert reporter._pct(3, 4) == "75%"

    def test_save(self, tmp_path):
        sr = _make_reporter_suite_result()
        reporter = ConsoleReporter()
        path = tmp_path / "report.txt"
        reporter.save(sr, path)
        assert path.read_text().startswith("=")

    def test_status_icons(self):
        suite = EvalSuite(name="s", threshold=0.7)
        results_data = [
            (EvalStatus.PASSED, "[PASS]"),
            (EvalStatus.FAILED, "[FAIL]"),
            (EvalStatus.ERROR, "[ERR!]"),
            (EvalStatus.SKIPPED, "[SKIP]"),
        ]
        for status, icon in results_data:
            result = _make_result(
                case=_make_case(name=f"{status.value}_case"), status=status, score=0.5
            )
            sr = SuiteResult(
                suite=suite,
                results=[result],
                passed=1 if status == EvalStatus.PASSED else 0,
                failed=1 if status == EvalStatus.FAILED else 0,
                errors=1 if status == EvalStatus.ERROR else 0,
                skipped=1 if status == EvalStatus.SKIPPED else 0,
                total_score=0.5,
                duration_ms=10.0,
            )
            reporter = ConsoleReporter(verbose=True)
            report = reporter.report(sr)
            assert icon in report


class TestJSONReporter:
    """Tests for JSONReporter."""

    def test_basic_report(self):
        sr = _make_reporter_suite_result()
        reporter = JSONReporter()
        report = reporter.report(sr)
        data = json.loads(report)
        assert data["suite_name"] == "reporter_test"

    def test_indent(self):
        sr = _make_reporter_suite_result()
        reporter = JSONReporter(indent=4)
        report = reporter.report(sr)
        assert "    " in report

    def test_include_outputs(self):
        suite = EvalSuite(name="s")
        result = _make_result(output="my output")
        sr = SuiteResult(suite=suite, results=[result], passed=1, total_score=1.0)
        reporter = JSONReporter(include_outputs=True)
        report = reporter.report(sr)
        data = json.loads(report)
        assert "output" in data["results"][0]

    def test_exclude_outputs(self):
        suite = EvalSuite(name="s")
        result = _make_result(output="my output")
        sr = SuiteResult(suite=suite, results=[result], passed=1, total_score=1.0)
        reporter = JSONReporter(include_outputs=False)
        report = reporter.report(sr)
        data = json.loads(report)
        assert "output" not in data["results"][0]

    def test_save(self, tmp_path):
        sr = _make_reporter_suite_result()
        reporter = JSONReporter()
        path = tmp_path / "report.json"
        reporter.save(sr, path)
        data = json.loads(path.read_text())
        assert data["suite_name"] == "reporter_test"


class TestHTMLReporter:
    """Tests for HTMLReporter."""

    def test_basic_report(self):
        sr = _make_reporter_suite_result()
        reporter = HTMLReporter()
        report = reporter.report(sr)
        assert "<!DOCTYPE html>" in report
        assert "reporter_test" in report

    def test_custom_title(self):
        sr = _make_reporter_suite_result()
        reporter = HTMLReporter(title="Custom Title")
        report = reporter.report(sr)
        assert "Custom Title" in report

    def test_default_title(self):
        sr = _make_reporter_suite_result()
        reporter = HTMLReporter()
        report = reporter.report(sr)
        assert "Evaluation: reporter_test" in report

    def test_passed_verdict(self):
        sr = _make_reporter_suite_result(pass_rate_above_threshold=True)
        reporter = HTMLReporter()
        report = reporter.report(sr)
        assert "PASSED" in report

    def test_failed_verdict(self):
        sr = _make_reporter_suite_result(pass_rate_above_threshold=False)
        reporter = HTMLReporter()
        report = reporter.report(sr)
        assert "FAILED" in report

    def test_table_rows(self):
        sr = _make_reporter_suite_result(include_metrics=True)
        reporter = HTMLReporter()
        report = reporter.report(sr)
        assert "passed_case" in report
        assert "failed_case" in report

    def test_error_in_html(self):
        sr = _make_reporter_suite_result(include_errors=True)
        reporter = HTMLReporter()
        report = reporter.report(sr)
        assert "Something went wrong" in report

    def test_save(self, tmp_path):
        sr = _make_reporter_suite_result()
        reporter = HTMLReporter()
        path = tmp_path / "report.html"
        reporter.save(sr, path)
        assert "<!DOCTYPE html>" in path.read_text()

    def test_score_bar(self):
        sr = _make_reporter_suite_result()
        reporter = HTMLReporter()
        report = reporter.report(sr)
        assert "score-bar" in report

    def test_metrics_in_html(self):
        sr = _make_reporter_suite_result(include_metrics=True)
        reporter = HTMLReporter()
        report = reporter.report(sr)
        assert "accuracy" in report

    def test_footer_timestamp(self):
        sr = _make_reporter_suite_result()
        reporter = HTMLReporter()
        report = reporter.report(sr)
        assert "Generated on" in report
        assert "Gorgon Evaluation Framework" in report


class TestMarkdownReporter:
    """Tests for MarkdownReporter."""

    def test_basic_report(self):
        sr = _make_reporter_suite_result()
        reporter = MarkdownReporter()
        report = reporter.report(sr)
        assert "# Evaluation Results: reporter_test" in report
        assert "## Summary" in report
        assert "## Results" in report

    def test_summary_table(self):
        sr = _make_reporter_suite_result()
        reporter = MarkdownReporter()
        report = reporter.report(sr)
        assert "Total Cases" in report
        assert "Passed" in report
        assert "Failed" in report

    def test_results_table(self):
        sr = _make_reporter_suite_result()
        reporter = MarkdownReporter()
        report = reporter.report(sr)
        assert "passed_case" in report
        assert "failed_case" in report

    def test_passed_verdict(self):
        sr = _make_reporter_suite_result(pass_rate_above_threshold=True)
        reporter = MarkdownReporter()
        report = reporter.report(sr)
        assert "PASSED" in report

    def test_failed_verdict(self):
        sr = _make_reporter_suite_result(pass_rate_above_threshold=False)
        reporter = MarkdownReporter()
        report = reporter.report(sr)
        assert "FAILED" in report

    def test_zero_total(self):
        suite = EvalSuite(name="empty")
        sr = SuiteResult(suite=suite, total_score=0.0, duration_ms=0)
        reporter = MarkdownReporter()
        report = reporter.report(sr)
        assert "| Passed | 0 |" in report

    def test_save(self, tmp_path):
        sr = _make_reporter_suite_result()
        reporter = MarkdownReporter()
        path = tmp_path / "report.md"
        reporter.save(sr, path)
        assert "# Evaluation Results" in path.read_text()


# ===========================================================================
# Integration-style tests
# ===========================================================================


class TestEndToEnd:
    """Integration tests that exercise the full pipeline."""

    def test_full_pipeline_passing(self):
        """Agent -> Evaluator -> Runner -> Reporter end-to-end."""

        def agent_fn(x):
            return f"The answer is: {x}"

        evaluator = AgentEvaluator(agent_fn, threshold=0.5)

        suite = EvalSuite(name="e2e_suite", threshold=0.5)
        suite.add_metric(ContainsMetric())
        suite.add_case(input="hello", expected="hello", name="c1")
        suite.add_case(input="world", expected="world", name="c2")

        runner = EvalRunner(evaluator)
        result = runner.run(suite)

        assert result.passed == 2
        assert result.total_score == 1.0

        # Test all reporters
        console = ConsoleReporter(verbose=True).report(result)
        assert "PASSED" in console

        json_report = JSONReporter().report(result)
        data = json.loads(json_report)
        assert data["passed"] == 2

        html_report = HTMLReporter().report(result)
        assert "PASSED" in html_report

        md_report = MarkdownReporter().report(result)
        assert "PASSED" in md_report

    def test_full_pipeline_failing(self):
        def agent_fn(x):
            return "wrong answer"

        evaluator = AgentEvaluator(agent_fn, threshold=0.5)

        suite = EvalSuite(name="fail_suite", threshold=0.7)
        suite.add_metric(ExactMatchMetric())
        suite.add_case(input="hello", expected="hello", name="c1")

        runner = EvalRunner(evaluator)
        result = runner.run(suite)
        assert result.failed == 1

    def test_full_pipeline_with_error(self):
        def error_agent(x):
            raise RuntimeError("agent crashed")

        evaluator = AgentEvaluator(error_agent)
        suite = EvalSuite(name="error_suite")
        suite.add_case(input="hello", expected="hello")

        runner = EvalRunner(evaluator)
        result = runner.run(suite)
        assert result.errors == 1

    def test_full_pipeline_async(self):
        evaluator = _DummyEvaluator()
        suite = EvalSuite(name="async_suite")
        suite.add_case(input="hello")
        suite.add_case(input="world")

        runner = EvalRunner(evaluator)

        async def _test():
            return await runner.run_async(suite, parallel=True)

        result = asyncio.run(_test())
        assert result.passed == 2

    def test_quick_eval_helper(self):
        result = create_quick_eval(
            agent_fn=lambda x: x.upper(),
            cases=[
                {"input": "hello", "expected": "HELLO"},
                {"input": "world", "expected": "WORLD"},
            ],
            metrics=["exact"],
        )
        assert result.passed == 2
        assert result.pass_rate == 1.0
