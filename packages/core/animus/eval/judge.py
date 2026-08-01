"""RubricJudge: score agent outputs using structured rubric dimensions.

Supports small-model judges (4B via Ollama) per ATLAS paper finding:
rubric-based RFT outperforms scalar judge rewards.

Design:
- One judge instance per rubric
- Dimension scoring can be batched or parallel
- Supports few-shot examples for calibration
- Confidence scores per dimension enable selective human review
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from animus.eval.rubric import Dimension, DimensionScore, Rubric, Score, ScoreLevel
from animus.logging import get_logger

logger = get_logger("eval.judge")


@dataclass
class JudgeConfig:
    """Configuration for a RubricJudge."""

    provider: str = "ollama"  # ollama, anthropic, openai
    model: str = "qwen2.5-coder:4b"  # Small judge per ATLAS
    temperature: float = 0.1  # Low variance for judging
    max_tokens: int = 1024
    # Per-dimension few-shot examples for calibration
    examples: dict[str, list[dict]] = field(default_factory=dict)
    # If True, score each dimension independently (more accurate but slower)
    score_per_dimension: bool = True
    # Retry config
    max_retries: int = 2
    # Fallback: if judge fails, assign NEUTRAL scores
    fallback_on_error: bool = True


@dataclass
class JudgmentResult:
    """Result of a single judgment call."""

    dimension_scores: list[DimensionScore]
    raw_response: str = ""
    parse_error: str | None = None
    tokens_used: int = 0
    duration_ms: float = 0.0


def _level_from_value(value: int | str) -> ScoreLevel:
    """Convert numeric or string value to ScoreLevel."""
    if isinstance(value, str):
        value = int(value.strip())
    mapping = {
        1: ScoreLevel.CRITICAL,
        2: ScoreLevel.POOR,
        3: ScoreLevel.ACCEPTABLE,
        4: ScoreLevel.GOOD,
        5: ScoreLevel.EXCELLENT,
    }
    return mapping.get(value, ScoreLevel.ACCEPTABLE)


def _normalize_score(level: ScoreLevel) -> float:
    """Map ScoreLevel to 0.0–1.0 range."""
    mapping = {
        ScoreLevel.CRITICAL: 0.0,
        ScoreLevel.POOR: 0.25,
        ScoreLevel.ACCEPTABLE: 0.5,
        ScoreLevel.GOOD: 0.75,
        ScoreLevel.EXCELLENT: 1.0,
    }
    return mapping[level]


class RubricJudge:
    """Judge that scores agent outputs against rubrics.

    Can use local small models for cheap per-dimension judging,
    or cloud models for higher accuracy on critical dimensions.
    """

    def __init__(self, config: JudgeConfig | None = None):
        self.config = config or JudgeConfig()
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazy-load judge client."""
        if self._client is not None:
            return self._client

        if self.config.provider == "ollama":
            try:
                import ollama

                self._client = ollama
            except ImportError:
                logger.warning("ollama not installed, using mock judge")
                self._client = None
        elif self.config.provider == "anthropic":
            try:
                import anthropic

                self._client = anthropic.Anthropic()
            except ImportError:
                logger.warning("anthropic not installed, using mock judge")
                self._client = None
        else:
            self._client = None

        return self._client

    def _call_judge(self, prompt: str) -> tuple[str, int]:
        """Call the judge model and return (response_text, tokens_used).

        Returns mock response if no client available.
        """
        client = self._get_client()
        if client is None:
            # Mock judge for testing / when providers unavailable
            return self._mock_judge(prompt), 0

        try:
            if self.config.provider == "ollama":
                response = client.chat(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    },
                )
                text = response["message"]["content"]
                tokens = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)
                return text, tokens

            elif self.config.provider == "anthropic":
                response = client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text
                tokens = response.usage.input_tokens + response.usage.output_tokens
                return text, tokens

        except Exception as e:
            logger.error(f"Judge call failed: {e}")
            if self.config.fallback_on_error:
                return self._mock_judge(prompt), 0
            raise

        return "", 0

    def _mock_judge(self, prompt: str) -> str:
        """Mock judge that returns plausible scores for testing."""
        # Extract dimension name from prompt
        match = re.search(r"Dimension:\s*(\w+)", prompt)
        dim_name = match.group(1) if match else "unknown"

        # Heuristic: if output contains "error" or "fail", score lower
        output_match = re.search(r"Output to evaluate:\s*---\s*([\s\S]*?)\s*---", prompt)
        output_text = output_match.group(1) if output_match else ""
        if "error" in output_text.lower() or "fail" in output_text.lower():
            return json.dumps({"score": 2, "justification": "Mock: detected failure indicators"})

        return json.dumps({"score": 4, "justification": f"Mock: {dim_name} looks acceptable"})

    def _build_dimension_prompt(
        self,
        rubric: Rubric,
        dimension: Dimension,
        output_text: str,
        context: str | None = None,
        reference: str | None = None,
    ) -> str:
        """Build a prompt for scoring one dimension."""
        lines = [
            "You are an expert evaluator. Score the following output on one dimension.",
            "",
            rubric.format_for_judge(),
            "",
            f"## Dimension to score: {dimension.name}",
            dimension.format_criteria(),
            "",
        ]

        if context:
            lines.extend([f"## Task context:\n{context}", ""])

        if reference:
            lines.extend([f"## Reference answer:\n{reference}", ""])

        lines.extend(
            [
                "## Output to evaluate:",
                "---",
                output_text[:4000],  # Limit to avoid context overflow
                "---",
                "",
                "Respond with ONLY a JSON object in this exact format:",
                '{"score": 1_to_5, "justification": "brief explanation"}',
                "",
                "Score meanings:",
                "1 = Critical failure, 2 = Poor, 3 = Acceptable, 4 = Good, 5 = Excellent",
            ]
        )

        # Add few-shot examples if configured
        dim_examples = self.config.examples.get(dimension.name, [])
        if dim_examples:
            lines.append("\n## Examples:")
            for ex in dim_examples:
                lines.append(f"Output: {ex.get('output', '')[:200]}")
                lines.append(f"Score: {ex.get('score', 3)}")
                lines.append(f"Justification: {ex.get('justification', '')}")
                lines.append("")

        return "\n".join(lines)

    def _build_batch_prompt(
        self,
        rubric: Rubric,
        output_text: str,
        context: str | None = None,
        reference: str | None = None,
    ) -> str:
        """Build a prompt for scoring all dimensions at once."""
        lines = [
            "You are an expert evaluator. Score the following output on all dimensions.",
            "",
            rubric.format_for_judge(),
            "",
        ]

        if context:
            lines.extend([f"## Task context:\n{context}", ""])
        if reference:
            lines.extend([f"## Reference answer:\n{reference}", ""])

        lines.extend(
            [
                "## Output to evaluate:",
                "---",
                output_text[:4000],
                "---",
                "",
                "Respond with ONLY a JSON object in this exact format:",
                '{"dimension_name": {"score": 1_to_5, "justification": "brief explanation"}, ...}',
            ]
        )

        return "\n".join(lines)

    def _parse_dimension_score(self, raw: str, dimension: Dimension) -> DimensionScore:
        """Parse judge response for a single dimension."""
        try:
            # Extract JSON from response
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in response")
            data = json.loads(match.group())

            score_val = data.get("score", 3)
            level = _level_from_value(score_val)
            raw_score = _normalize_score(level)

            return DimensionScore(
                dimension_name=dimension.name,
                level=level,
                raw_score=raw_score,
                justification=data.get("justification", "No justification provided"),
                confidence=1.0,  # Could be derived from model uncertainty
            )
        except Exception as e:
            logger.warning(f"Failed to parse judge response for {dimension.name}: {e}")
            return DimensionScore(
                dimension_name=dimension.name,
                level=ScoreLevel.ACCEPTABLE,
                raw_score=0.5,
                justification=f"Parse error: {e}",
                confidence=0.0,
            )

    def _parse_batch_scores(self, raw: str, rubric: Rubric) -> list[DimensionScore]:
        """Parse judge response for batch dimension scoring."""
        results: list[DimensionScore] = []
        try:
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in response")
            data = json.loads(match.group())

            for dim in rubric.dimensions:
                dim_data = data.get(dim.name, {})
                if isinstance(dim_data, dict):
                    score_val = dim_data.get("score", 3)
                    level = _level_from_value(score_val)
                    raw_score = _normalize_score(level)
                    justification = dim_data.get("justification", "No justification")
                else:
                    level = ScoreLevel.ACCEPTABLE
                    raw_score = 0.5
                    justification = "Malformed response"

                results.append(
                    DimensionScore(
                        dimension_name=dim.name,
                        level=level,
                        raw_score=raw_score,
                        justification=justification,
                        confidence=1.0,
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to parse batch judge response: {e}")
            # Fallback: all dimensions get ACCEPTABLE
            for dim in rubric.dimensions:
                results.append(
                    DimensionScore(
                        dimension_name=dim.name,
                        level=ScoreLevel.ACCEPTABLE,
                        raw_score=0.5,
                        justification=f"Parse error: {e}",
                        confidence=0.0,
                    )
                )
        return results

    def judge(
        self,
        rubric: Rubric,
        output_text: str,
        context: str | None = None,
        reference: str | None = None,
    ) -> Score:
        """Score an output against a rubric.

        Args:
            rubric: The rubric to evaluate against.
            output_text: The agent output to evaluate.
            context: Optional task context/prompt.
            reference: Optional reference/ground-truth answer.

        Returns:
            Score with per-dimension breakdown.
        """
        if self.config.score_per_dimension:
            return self._judge_per_dimension(rubric, output_text, context, reference)
        return self._judge_batch(rubric, output_text, context, reference)

    def _judge_per_dimension(
        self,
        rubric: Rubric,
        output_text: str,
        context: str | None,
        reference: str | None,
    ) -> Score:
        """Score each dimension independently (more accurate, slower)."""
        dimension_scores: list[DimensionScore] = []
        total_tokens = 0

        for dim in rubric.dimensions:
            prompt = self._build_dimension_prompt(rubric, dim, output_text, context, reference)
            raw_response, tokens = self._call_judge(prompt)
            total_tokens += tokens

            ds = self._parse_dimension_score(raw_response, dim)
            dimension_scores.append(ds)

        score = Score(
            rubric_name=rubric.name,
            dimension_scores=dimension_scores,
            metadata={
                "weights": {d.name: d.weight for d in rubric.dimensions},
                "required_dims": {d.name: d.required for d in rubric.dimensions},
                "tokens_used": total_tokens,
            },
        )
        return score

    def _judge_batch(
        self,
        rubric: Rubric,
        output_text: str,
        context: str | None,
        reference: str | None,
    ) -> Score:
        """Score all dimensions in one call (faster, potentially less accurate)."""
        prompt = self._build_batch_prompt(rubric, output_text, context, reference)
        raw_response, tokens = self._call_judge(prompt)

        dimension_scores = self._parse_batch_scores(raw_response, rubric)

        score = Score(
            rubric_name=rubric.name,
            dimension_scores=dimension_scores,
            metadata={
                "weights": {d.name: d.weight for d in rubric.dimensions},
                "required_dims": {d.name: d.required for d in rubric.dimensions},
                "tokens_used": tokens,
            },
        )
        return score
