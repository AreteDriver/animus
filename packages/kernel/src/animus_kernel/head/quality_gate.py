"""Quality gates for Animus Head — lightweight evaluation of local model outputs.

Scores responses without requiring another model call, using heuristics
based on tool-call validity, response structure, and error patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from animus_kernel.providers.base import CompletionResponse

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Score breakdown for a model response."""

    overall: int = 0  # 0-100 composite score
    tool_call_quality: int = 0  # 0-100: valid tools, correct args
    response_completeness: int = 0  # 0-100: non-empty, addresses user intent
    structure_quality: int = 0  # 0-100: proper JSON, no malformed blocks
    failure_streak: int = 0  # Consecutive low-quality turns
    reason: str = ""


class HeadQualityGate:
    """Lightweight quality evaluator for local model outputs.

    Uses deterministic heuristics — no secondary model call required.
    """

    def __init__(self, max_failure_streak: int = 3) -> None:
        self.max_failure_streak = max_failure_streak
        self._failure_streak = 0

    def evaluate(
        self,
        user_input: str,
        response: CompletionResponse,
        valid_tool_calls: list,
        invalid_tool_calls: list,
    ) -> QualityScore:
        """Evaluate a model response and return a composite score.

        Args:
            user_input: The original user message
            response: Model's CompletionResponse
            valid_tool_calls: Tool calls that passed validation
            invalid_tool_calls: ToolCallValidationResult objects that failed

        Returns:
            QualityScore with breakdown and composite score
        """
        # Tool call quality (0-40 points)
        tq = self._score_tool_calls(valid_tool_calls, invalid_tool_calls)

        # Response completeness (0-40 points)
        rc = self._score_completeness(user_input, response)

        # Structure quality (0-20 points)
        sq = self._score_structure(response, invalid_tool_calls)

        overall = min(100, tq + rc + sq)

        # Completely empty response (no content, no tool calls) is a hard failure
        if not (response.content or "").strip() and not valid_tool_calls and not invalid_tool_calls:
            overall = min(overall, 30)

        # Update failure streak
        if overall < 40:
            self._failure_streak += 1
            reason = f"Low quality score ({overall}): "
            if tq < 20:
                reason += "poor tool calls; "
            if rc < 20:
                reason += "empty or incomplete response; "
            if sq < 10:
                reason += "structural issues; "
            reason = reason.rstrip("; ")
        else:
            self._failure_streak = 0
            reason = "pass"

        return QualityScore(
            overall=overall,
            tool_call_quality=tq,
            response_completeness=rc,
            structure_quality=sq,
            failure_streak=self._failure_streak,
            reason=reason,
        )

    def should_fallback(self, score: QualityScore, threshold: int = 40) -> bool:
        """Determine if cloud fallback is warranted.

        Fallback triggers when:
        - Overall score is below threshold, OR
        - Failure streak exceeds max_failure_streak
        """
        if score.overall < threshold:
            return True
        if score.failure_streak >= self.max_failure_streak:
            return True
        return False

    # ------------------------------------------------------------------
    # Scoring heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_tool_calls(valid: list, invalid: list) -> int:
        """Score tool call quality (0-40)."""
        if not valid and not invalid:
            # No tool calls attempted — neutral for non-tool turns
            return 25

        total = len(valid) + len(invalid)
        if total == 0:
            return 25

        valid_ratio = len(valid) / total
        # Full validity = 40, half = 20, none = 0
        return int(valid_ratio * 40)

    @staticmethod
    def _score_completeness(user_input: str, response: CompletionResponse) -> int:
        """Score response completeness (0-40)."""
        content = (response.content or "").strip()

        if not content:
            # Empty content is a strong negative signal
            return 0

        # Length heuristic: very short responses are often refusals or errors
        if len(content) < 10:
            return 10

        # Check for refusal patterns
        refusal_phrases = [
            "i cannot",
            "i can't",
            "i'm unable",
            "i am unable",
            "i don't know",
            "i do not know",
            "not sure",
            "cannot help",
        ]
        content_lower = content.lower()
        if any(p in content_lower for p in refusal_phrases):
            return 5

        # Good response
        return 40

    @staticmethod
    def _score_structure(response: CompletionResponse, invalid_calls: list) -> int:
        """Score structural quality (0-20)."""
        score = 20

        # Penalize invalid tool call structures
        if invalid_calls:
            score -= min(15, len(invalid_calls) * 5)

        # Penalize empty finish_reason (sometimes indicates truncation)
        if response.finish_reason is None:
            score -= 5

        return max(0, score)

    def reset(self) -> None:
        """Reset failure streak."""
        self._failure_streak = 0
