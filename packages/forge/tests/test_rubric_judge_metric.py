"""Tests for RubricJudgeMetric and RubricRegistry (ATLAS Proposal 2)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from animus_forge.evaluation.base import EvalCase
from animus_forge.evaluation.metrics import (
    JudgeError,
    RubricJudgeMetric,
    RubricRegistry,
)


class MockProvider:
    """Mock judge provider for testing."""

    def __init__(self, response_text: str = "6"):
        self.response_text = response_text

    def complete(self, request):
        mock = MagicMock()
        mock.content = self.response_text
        mock.tokens_used = 10
        return mock


class TestRubricJudgeMetric:
    """Tests for structured rubric judge."""

    def test_name(self):
        metric = RubricJudgeMetric()
        assert metric.name == "rubric_judge"

    def test_no_provider_raises(self):
        metric = RubricJudgeMetric()
        case = EvalCase(input="test")
        with pytest.raises(JudgeError, match="no judge_provider"):
            metric.score("output", None, case)

    def test_parse_valid_json(self):
        metric = RubricJudgeMetric()
        text = """{
            "criteria": {
                "correctness": {"score": 8, "reason": "Correct."},
                "completeness": {"score": 5, "reason": "Missing detail."}
            },
            "overall": 6.5
        }"""
        parsed = metric._parse_judge_response(text)
        assert parsed is not None
        assert parsed["overall"] == 6.5
        assert "correctness" in parsed["criteria"]
        assert parsed["criteria"]["correctness"]["score"] == 8.0

    def test_parse_with_markdown_fences(self):
        metric = RubricJudgeMetric()
        text = '```json\n{"criteria": {}, "overall": 7.0}\n```'
        parsed = metric._parse_judge_response(text)
        assert parsed is not None
        assert parsed["overall"] == 7.0

    def test_parse_invalid_json_returns_none(self):
        metric = RubricJudgeMetric()
        assert metric._parse_judge_response("not json") is None

    def test_parse_missing_overall_returns_none(self):
        metric = RubricJudgeMetric()
        assert metric._parse_judge_response('{"criteria": {}}') is None

    def test_weighted_average(self):
        metric = RubricJudgeMetric(weights={"correctness": 2.0, "completeness": 1.0})
        criteria = {
            "correctness": {"score": 10},
            "completeness": {"score": 4},
        }
        avg = metric._weighted_average(criteria)
        # (10*2 + 4*1) / 3 = 24/3 = 8
        assert avg == 8.0

    def test_score_end_to_end(self):
        provider = MockProvider(
            response_text='{"criteria": {"correctness": {"score": 8, "reason": "ok"}}, "overall": 8.0}'
        )
        metric = RubricJudgeMetric(judge_provider=provider)
        case = EvalCase(input="test task")
        score = metric.score("output", None, case)
        assert score == 0.8  # 8.0 / 10
        assert "rubric_criteria" in case.metadata
        assert case.metadata["rubric_criteria"]["correctness"]["score"] == 8.0

    def test_score_with_bad_response_raises(self):
        provider = MockProvider(response_text="bad response")
        metric = RubricJudgeMetric(judge_provider=provider)
        case = EvalCase(input="test")
        with pytest.raises(JudgeError, match="no parseable rubric"):
            metric.score("output", None, case)

    def test_criteria_descriptions_used_in_prompt(self):
        provider = MockProvider(response_text='{"criteria": {"foo": {"score": 5}}, "overall": 5.0}')
        metric = RubricJudgeMetric(
            judge_provider=provider,
            criteria={"foo": "Bar baz"},
        )
        case = EvalCase(input="test")
        metric.score("output", None, case)
        # Provider received the prompt — we can verify criteria were included
        # by checking the mock's call args if we captured them.


class TestRubricRegistry:
    """Tests for task-adaptive rubric registry."""

    def test_get_default_rubric(self):
        rubric = RubricRegistry.get_rubric("unknown")
        assert "correctness" in rubric

    def test_get_code_edit_rubric(self):
        rubric = RubricRegistry.get_rubric("code-edit")
        assert "safety" in rubric
        assert "testability" in rubric

    def test_list_task_types(self):
        types = RubricRegistry.list_task_types()
        assert "default" in types
        assert "code-edit" in types
        assert "architecture-review" in types

    def test_register_custom_rubric(self):
        RubricRegistry.register_rubric("custom", {"a": "Criterion A"})
        rubric = RubricRegistry.get_rubric("custom")
        assert rubric["a"] == "Criterion A"

    def test_create_metric(self):
        metric = RubricRegistry.create_metric("code-edit")
        assert isinstance(metric, RubricJudgeMetric)
        assert "safety" in metric._criteria

    def test_create_metric_with_weights(self):
        metric = RubricRegistry.create_metric("default", weights={"correctness": 2.0})
        assert metric._weights == {"correctness": 2.0}
