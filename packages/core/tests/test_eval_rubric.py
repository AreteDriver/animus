"""Tests for rubric-based evaluation framework.

Validates P-20260706-004: Rubric-Based Evaluation Rewards.
"""

import pytest

from animus.eval.rubric import (
    Dimension,
    Rubric,
    Score,
    ScoreLevel,
    DimensionScore,
)
from animus.eval.judge import (
    RubricJudge,
    JudgeConfig,
    _level_from_value,
    _normalize_score,
)
from animus.eval.rewards import (
    RewardAggregator,
    RewardConfig,
)
from animus.eval.rubrics.personal_quality import create_personal_quality_rubric


class TestScoreLevel:
    """ScoreLevel enum behavior."""

    def test_level_ordering(self):
        assert ScoreLevel.CRITICAL.value < ScoreLevel.POOR.value
        assert ScoreLevel.POOR.value < ScoreLevel.ACCEPTABLE.value
        assert ScoreLevel.ACCEPTABLE.value < ScoreLevel.GOOD.value
        assert ScoreLevel.GOOD.value < ScoreLevel.EXCELLENT.value

    def test_level_from_value(self):
        assert _level_from_value(1) == ScoreLevel.CRITICAL
        assert _level_from_value(2) == ScoreLevel.POOR
        assert _level_from_value(3) == ScoreLevel.ACCEPTABLE
        assert _level_from_value(4) == ScoreLevel.GOOD
        assert _level_from_value(5) == ScoreLevel.EXCELLENT

    def test_normalize_score(self):
        assert _normalize_score(ScoreLevel.CRITICAL) == 0.0
        assert _normalize_score(ScoreLevel.POOR) == 0.25
        assert _normalize_score(ScoreLevel.ACCEPTABLE) == 0.5
        assert _normalize_score(ScoreLevel.GOOD) == 0.75
        assert _normalize_score(ScoreLevel.EXCELLENT) == 1.0


class TestDimension:
    """Dimension creation and formatting."""

    def test_dimension_creation(self):
        dim = Dimension(
            name="correctness",
            description="Is the answer correct?",
            weight=1.5,
            criteria={
                ScoreLevel.EXCELLENT: "Completely correct",
                ScoreLevel.CRITICAL: "Completely wrong",
            },
        )
        assert dim.name == "correctness"
        assert dim.weight == 1.5
        # Auto-populated missing criteria
        assert ScoreLevel.ACCEPTABLE in dim.criteria

    def test_format_criteria(self):
        dim = Dimension(
            name="test",
            description="test dim",
            criteria={ScoreLevel.GOOD: "Good job"},
        )
        text = dim.format_criteria()
        assert "test" in text
        assert "Good job" in text


class TestRubric:
    """Rubric creation and serialization."""

    def test_rubric_building(self):
        rubric = (
            Rubric(name="test", description="test rubric")
            .add_dimension(
                Dimension(name="dim1", description="first", weight=1.0)
            )
            .add_dimension(
                Dimension(name="dim2", description="second", weight=2.0)
            )
        )
        assert len(rubric.dimensions) == 2
        assert rubric.total_weight == 3.0

    def test_rubric_serialization(self):
        rubric = create_personal_quality_rubric()
        data = rubric.to_dict()
        restored = Rubric.from_dict(data)
        assert restored.name == rubric.name
        assert len(restored.dimensions) == len(rubric.dimensions)
        assert restored.dimensions[0].name == rubric.dimensions[0].name

    def test_rubric_format_for_judge(self):
        rubric = create_personal_quality_rubric()
        text = rubric.format_for_judge()
        assert "relevance" in text.lower()
        assert "precision" in text.lower()
        assert "1 —" in text  # Score level indicator


class TestScore:
    """Score computation and properties."""

    def test_weighted_score(self):
        score = Score(
            rubric_name="test",
            dimension_scores=[
                DimensionScore("dim1", ScoreLevel.GOOD, 0.75),
                DimensionScore("dim2", ScoreLevel.EXCELLENT, 1.0),
            ],
            metadata={"weights": {"dim1": 1.0, "dim2": 2.0}},
        )
        # (0.75 * 1 + 1.0 * 2) / 3 = 0.9167
        assert pytest.approx(score.weighted_score, 0.01) == 0.9167

    def test_overall_level(self):
        score = Score(
            rubric_name="test",
            dimension_scores=[
                DimensionScore("d1", ScoreLevel.GOOD, 0.75),
                DimensionScore("d2", ScoreLevel.GOOD, 0.75),
            ],
        )
        assert score.overall_level == ScoreLevel.GOOD

    def test_critical_failure(self):
        score = Score(
            rubric_name="test",
            dimension_scores=[
                DimensionScore("d1", ScoreLevel.CRITICAL, 0.0),
                DimensionScore("d2", ScoreLevel.EXCELLENT, 1.0),
            ],
            metadata={"required_dims": {"d1": True}},
        )
        assert score.has_critical_failure is True

    def test_failures_and_strengths(self):
        score = Score(
            rubric_name="test",
            dimension_scores=[
                DimensionScore("d1", ScoreLevel.CRITICAL, 0.0),
                DimensionScore("d2", ScoreLevel.POOR, 0.25),
                DimensionScore("d3", ScoreLevel.GOOD, 0.75),
                DimensionScore("d4", ScoreLevel.EXCELLENT, 1.0),
            ],
        )
        assert len(score.failures()) == 2
        assert len(score.strengths()) == 2

    def test_summary_output(self):
        score = Score(
            rubric_name="test",
            dimension_scores=[
                DimensionScore("d1", ScoreLevel.GOOD, 0.75, "solid work"),
            ],
        )
        summary = score.summary()
        assert "test" in summary
        assert "0.75" in summary
        assert "solid work" in summary


class TestRubricJudge:
    """RubricJudge scoring behavior."""

    def test_mock_judge_scoring(self):
        judge = RubricJudge(JudgeConfig(provider="mock"))
        rubric = (
            Rubric(name="simple", description="test")
            .add_dimension(Dimension(name="quality", description="how good"))
        )
        score = judge.judge(rubric, "This is a test output")
        assert score.rubric_name == "simple"
        assert len(score.dimension_scores) == 1
        # Mock returns 4 (GOOD) for non-error outputs
        assert score.dimension_scores[0].level == ScoreLevel.GOOD

    def test_mock_judge_detects_failure(self):
        judge = RubricJudge(JudgeConfig(provider="mock"))
        rubric = (
            Rubric(name="simple", description="test")
            .add_dimension(Dimension(name="quality", description="how good"))
        )
        score = judge.judge(rubric, "This output contains an error and fails")
        # Mock detects "error"/"fail" and returns POOR (2)
        assert score.dimension_scores[0].level == ScoreLevel.POOR

    def test_batch_vs_per_dimension(self):
        rubric = (
            Rubric(name="batch", description="test")
            .add_dimension(Dimension(name="a", description="first"))
            .add_dimension(Dimension(name="b", description="second"))
        )

        # Per-dimension
        judge_per = RubricJudge(JudgeConfig(provider="mock", score_per_dimension=True))
        score_per = judge_per.judge(rubric, "test output")
        assert len(score_per.dimension_scores) == 2

        # Batch
        judge_batch = RubricJudge(JudgeConfig(provider="mock", score_per_dimension=False))
        score_batch = judge_batch.judge(rubric, "test output")
        assert len(score_batch.dimension_scores) == 2

    def test_parse_dimension_score_valid(self):
        judge = RubricJudge()
        raw = '{"score": 4, "justification": "Good output"}'
        dim = Dimension(name="test", description="test")
        ds = judge._parse_dimension_score(raw, dim)
        assert ds.level == ScoreLevel.GOOD
        assert ds.raw_score == 0.75
        assert ds.justification == "Good output"

    def test_parse_dimension_score_invalid(self):
        judge = RubricJudge()
        raw = "not json"
        dim = Dimension(name="test", description="test")
        ds = judge._parse_dimension_score(raw, dim)
        # Falls back to ACCEPTABLE
        assert ds.level == ScoreLevel.ACCEPTABLE
        assert ds.confidence == 0.0


class TestRewardAggregator:
    """Reward aggregation from rubric scores."""

    def test_linear_reward(self):
        agg = RewardAggregator(RewardConfig(shaping="linear"))
        score = Score(
            rubric_name="test",
            dimension_scores=[
                DimensionScore("d1", ScoreLevel.GOOD, 0.75),
                DimensionScore("d2", ScoreLevel.GOOD, 0.75),
            ],
        )
        reward = agg.aggregate(score)
        assert reward == pytest.approx(0.75, 0.01)

    def test_critical_failure_penalty(self):
        agg = RewardAggregator(RewardConfig(critical_penalty=-2.0))
        score = Score(
            rubric_name="test",
            dimension_scores=[
                DimensionScore("d1", ScoreLevel.CRITICAL, 0.0),
            ],
            metadata={"required_dims": {"d1": True}},
        )
        reward = agg.aggregate(score)
        assert reward == -2.0

    def test_below_threshold_zero(self):
        agg = RewardAggregator(RewardConfig(min_threshold=0.6))
        score = Score(
            rubric_name="test",
            dimension_scores=[
                DimensionScore("d1", ScoreLevel.POOR, 0.25),
            ],
        )
        reward = agg.aggregate(score)
        assert reward == 0.0

    def test_sigmoid_shaping(self):
        agg = RewardAggregator(RewardConfig(shaping="sigmoid"))
        score = Score(
            rubric_name="test",
            dimension_scores=[
                DimensionScore("d1", ScoreLevel.ACCEPTABLE, 0.5),
            ],
        )
        reward = agg.aggregate(score)
        # Sigmoid at 0.5 should be ~0.5
        assert 0.4 <= reward <= 0.6

    def test_step_shaping(self):
        agg = RewardAggregator(RewardConfig(shaping="step"))
        score = Score(
            rubric_name="test",
            dimension_scores=[
                DimensionScore("d1", ScoreLevel.GOOD, 0.75),
            ],
        )
        reward = agg.aggregate(score)
        # GOOD (0.75) = step threshold for 1.0
        assert reward == 1.0

    def test_selective_reward(self):
        agg = RewardAggregator(
            RewardConfig(selective_reward=True, selective_threshold=0.6)
        )
        score = Score(
            rubric_name="test",
            dimension_scores=[
                DimensionScore("d1", ScoreLevel.GOOD, 0.75),
                DimensionScore("d2", ScoreLevel.POOR, 0.25),
            ],
        )
        reward = agg.aggregate(score)
        # d2 below threshold, selective_reward requires ALL above
        assert reward == 0.0

    def test_per_dimension_rewards(self):
        agg = RewardAggregator()
        score = Score(
            rubric_name="test",
            dimension_scores=[
                DimensionScore("d1", ScoreLevel.EXCELLENT, 1.0),
                DimensionScore("d2", ScoreLevel.POOR, 0.25),
            ],
        )
        rewards = agg.per_dimension_rewards(score)
        assert rewards["d1"] == 1.0
        assert rewards["d2"] == 0.0  # Below default threshold

    def test_improvement_direction(self):
        agg = RewardAggregator()
        score = Score(
            rubric_name="test",
            dimension_scores=[
                DimensionScore("d1", ScoreLevel.CRITICAL, 0.0, "completely wrong"),
                DimensionScore("d2", ScoreLevel.POOR, 0.25, "needs work"),
                DimensionScore("d3", ScoreLevel.GOOD, 0.75, "fine"),
            ],
        )
        directions = agg.improvement_direction(score)
        assert len(directions) == 2
        assert "CRITICAL" in directions[0]

    def test_no_failures_direction(self):
        agg = RewardAggregator()
        score = Score(
            rubric_name="test",
            dimension_scores=[
                DimensionScore("d1", ScoreLevel.GOOD, 0.75),
            ],
        )
        directions = agg.improvement_direction(score)
        assert len(directions) == 1
        assert "GOOD to EXCELLENT" in directions[0]


class TestPersonalQualityRubric:
    """Built-in personal quality rubric."""

    def test_rubric_has_six_dimensions(self):
        rubric = create_personal_quality_rubric()
        assert len(rubric.dimensions) == 6
        names = [d.name for d in rubric.dimensions]
        assert "relevance" in names
        assert "precision" in names
        assert "actionability" in names
        assert "evidence_quality" in names
        assert "format_compliance" in names
        assert "hallucination_safety" in names

    def test_required_dimensions(self):
        rubric = create_personal_quality_rubric()
        required = [d for d in rubric.dimensions if d.required]
        assert len(required) == 2
        assert required[0].name == "precision"
        assert required[1].name == "hallucination_safety"

    def test_precision_has_highest_weight(self):
        rubric = create_personal_quality_rubric()
        weights = {d.name: d.weight for d in rubric.dimensions}
        assert weights["precision"] == 1.5
        assert weights["hallucination_safety"] == 1.5
        assert weights["relevance"] == 1.2
