"""Rubric-based evaluation framework for Animus.

Implements P-20260706-004: Replace scalar LLM-as-judge rewards with
structured task-level rubrics. Inspired by ATLAS (arXiv 2603.06713).

Key design:
- Dimensions are scored independently (no scalar collapse)
- Small-model judges (4B via Ollama) can handle individual dimensions
- Task-type-specific weights enable targeted optimization
- Few-shot examples calibrate judges per dimension
"""

from animus.eval.rubric import (
    Dimension,
    Rubric,
    Score,
    ScoreLevel,
)
from animus.eval.judge import (
    RubricJudge,
    JudgeConfig,
    JudgmentResult,
)
from animus.eval.rewards import (
    RewardAggregator,
    RewardConfig,
)

__all__ = [
    "Dimension",
    "Rubric",
    "Score",
    "ScoreLevel",
    "RubricJudge",
    "JudgeConfig",
    "JudgmentResult",
    "RewardAggregator",
    "RewardConfig",
]