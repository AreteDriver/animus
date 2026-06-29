"""Tests for the adversarial evaluation harness."""

from __future__ import annotations

import pytest

from animus_forge.evaluation.adversarial import (
    AdversarialCaseGenerator,
    FaultInjectingEvaluator,
    FaultProfile,
    PropertyBasedEvaluator,
    run_adversarial_suite,
)
from animus_forge.evaluation.base import AgentEvaluator, EvalCase, EvalResult, EvalStatus
from animus_forge.evaluation.metrics import ExactMatchMetric


def _dummy_agent(inp: str) -> str:
    return f"processed: {inp[:20]}"


class TestAdversarialCaseGenerator:
    def test_generates_requested_count(self):
        gen = AdversarialCaseGenerator(seed=1)
        cases = gen.generate_cases(n=20)
        assert len(cases) == 20

    def test_cases_have_adversarial_tags(self):
        gen = AdversarialCaseGenerator(seed=1)
        cases = gen.generate_cases(n=10)
        for case in cases:
            assert "adversarial" in case.metadata.get("tags", [])
            assert case.metadata.get("adversarial_type") in {
                "empty_whitespace",
                "extreme_length",
                "unicode_bomb",
                "prompt_injection",
                "malformed_json",
                "mixed_language",
                "random_noise",
            }

    def test_suite_has_name_and_tags(self):
        gen = AdversarialCaseGenerator(seed=1)
        suite = gen.generate_suite(name="my_suite", n=5)
        assert suite.name == "my_suite"
        assert "adversarial" in suite.tags
        assert len(suite.cases) == 5


class TestFaultInjectingEvaluator:
    def test_network_error_injection(self):
        base = AgentEvaluator(_dummy_agent)
        # Force network error every time
        profile = FaultProfile(network_error_rate=1.0, timeout_rate=0.0, corrupt_output_rate=0.0)
        evaluator = FaultInjectingEvaluator(base, profile=profile, seed=1)

        case = EvalCase(input="hello", expected="processed: hello")
        result = evaluator.evaluate(case, [ExactMatchMetric()])

        assert result.status == EvalStatus.ERROR
        assert "network" in result.error.lower()
        assert result.metadata.get("injected_fault") == "network_error"

    def test_timeout_injection(self):
        base = AgentEvaluator(_dummy_agent)
        profile = FaultProfile(network_error_rate=0.0, timeout_rate=1.0, corrupt_output_rate=0.0)
        evaluator = FaultInjectingEvaluator(base, profile=profile, seed=1)

        case = EvalCase(input="hello", expected="processed: hello")
        result = evaluator.evaluate(case, [ExactMatchMetric()])

        assert result.status == EvalStatus.ERROR
        assert "timeout" in result.error.lower()
        assert result.metadata.get("injected_fault") == "timeout"

    def test_corrupt_output_injection(self):
        base = AgentEvaluator(_dummy_agent)
        profile = FaultProfile(network_error_rate=0.0, timeout_rate=0.0, corrupt_output_rate=1.0)
        evaluator = FaultInjectingEvaluator(base, profile=profile, seed=1)

        case = EvalCase(input="hello", expected="processed: hello")
        result = evaluator.evaluate(case, [ExactMatchMetric()])

        assert result.metadata.get("injected_fault") == "corrupt_output"
        # Output should differ from normal processing
        assert result.output != "processed: hello"

    def test_no_fault_when_rates_zero(self):
        base = AgentEvaluator(_dummy_agent)
        profile = FaultProfile(network_error_rate=0.0, timeout_rate=0.0, corrupt_output_rate=0.0, delay_ms=0.0)
        evaluator = FaultInjectingEvaluator(base, profile=profile, seed=1)

        case = EvalCase(input="hello", expected="processed: hello")
        result = evaluator.evaluate(case, [ExactMatchMetric()])

        assert result.status == EvalStatus.PASSED
        assert result.output == "processed: hello"


class TestPropertyBasedEvaluator:
    def test_fails_on_empty_output(self):
        base = AgentEvaluator(lambda x: "")
        evaluator = PropertyBasedEvaluator(base)

        case = EvalCase(input="hello", expected="something")
        result = evaluator.evaluate(case, [ExactMatchMetric()])

        assert result.status == EvalStatus.FAILED
        assert "empty" in result.error.lower()

    def test_fails_on_untransformed_output(self):
        base = AgentEvaluator(lambda x: x)
        evaluator = PropertyBasedEvaluator(base)

        case = EvalCase(input="hello", expected="something", metadata={"expect_transform": True})
        result = evaluator.evaluate(case, [ExactMatchMetric()])

        assert result.status == EvalStatus.FAILED
        assert "identical" in result.error.lower()

    def test_passes_when_properties_satisfied(self):
        base = AgentEvaluator(lambda x: "transformed output here")
        evaluator = PropertyBasedEvaluator(base)

        case = EvalCase(input="hello", expected="transformed output here")
        result = evaluator.evaluate(case, [ExactMatchMetric()])

        assert result.status == EvalStatus.PASSED


class TestRunAdversarialSuite:
    def test_runs_without_crashing(self):
        result = run_adversarial_suite(_dummy_agent, n_cases=5)
        assert result.total >= 5
        # Some cases will likely fail due to adversarial inputs, that's expected
        assert result.pass_rate >= 0.0
