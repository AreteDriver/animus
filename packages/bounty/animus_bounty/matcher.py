"""LLM-scored skill matching for bounties."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from animus_bounty.models import Bounty, Difficulty, SkillMatch

if TYPE_CHECKING:
    from animus_bounty.config import BountyConfig

logger = logging.getLogger("animus_bounty.matcher")


def _keyword_match_score(bounty: Bounty, skills: list[str]) -> tuple[list[str], list[str], float]:
    """Fast keyword-based skill matching (no LLM needed).

    Returns (matching_skills, missing_skills, score).
    """
    bounty_text = (
        f"{bounty.title} {bounty.description_snippet} "
        f"{' '.join(bounty.skills_required)} {' '.join(bounty.labels)}"
    ).lower()

    matching = []
    missing = []

    for skill in skills:
        if skill.lower() in bounty_text:
            matching.append(skill)

    # Skills required by the bounty that we don't have
    for req in bounty.skills_required:
        req_lower = req.lower()
        if not any(s.lower() in req_lower or req_lower in s.lower() for s in skills):
            missing.append(req)

    if not bounty.skills_required and matching:
        # No specific requirements and we have some matches — good sign
        score = min(1.0, len(matching) / max(1, len(skills)) * 3)
    elif bounty.skills_required:
        total = len(bounty.skills_required)
        matched = total - len(missing)
        score = matched / total if total > 0 else 0.0
    else:
        score = 0.5  # Unknown — neutral

    return matching, missing, round(score, 3)


def _estimate_hours(bounty: Bounty) -> float:
    """Estimate hours to complete based on difficulty and description length."""
    base = {
        Difficulty.TRIVIAL: 0.5,
        Difficulty.EASY: 2.0,
        Difficulty.MEDIUM: 8.0,
        Difficulty.HARD: 24.0,
        Difficulty.EXPERT: 60.0,
    }
    hours = base.get(bounty.difficulty, 8.0)

    # Adjust by description length as a rough complexity signal
    desc_len = len(bounty.description_snippet)
    if desc_len > 200:
        hours *= 1.2
    elif desc_len < 50:
        hours *= 0.8

    return round(hours, 1)


def _parse_llm_response(text: str, bounty_id: str) -> SkillMatch | None:
    """Parse LLM response into a SkillMatch.

    Expects JSON with: match_score, matching_skills, missing_skills,
    estimated_hours, reasoning.
    """
    # Try direct JSON parse
    try:
        data = json.loads(text)
        return SkillMatch(
            bounty_id=bounty_id,
            match_score=float(data.get("match_score", 0.0)),
            matching_skills=data.get("matching_skills", []),
            missing_skills=data.get("missing_skills", []),
            estimated_hours=float(data.get("estimated_hours", 0.0)),
            reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return SkillMatch(
                bounty_id=bounty_id,
                match_score=float(data.get("match_score", 0.0)),
                matching_skills=data.get("matching_skills", []),
                missing_skills=data.get("missing_skills", []),
                estimated_hours=float(data.get("estimated_hours", 0.0)),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("Failed to parse LLM response for bounty %s", bounty_id)
    return None


class SkillMatcher:
    """Matches bounties against a developer skill profile.

    Uses keyword matching as a fast path and optionally LLM
    for deeper analysis.
    """

    def __init__(self, config: BountyConfig):
        self.config = config
        self._skills = config.skills.all_skills()

    def quick_match(self, bounty: Bounty) -> SkillMatch:
        """Fast keyword-based matching (no LLM call)."""
        matching, missing, score = _keyword_match_score(bounty, self._skills)
        hours = _estimate_hours(bounty)

        return SkillMatch(
            bounty_id=bounty.bounty_id,
            match_score=score,
            matching_skills=matching,
            missing_skills=missing,
            estimated_hours=hours,
            reasoning=f"Keyword match: {len(matching)} skills matched, {len(missing)} missing",
        )

    def build_prompt(self, bounty: Bounty) -> str:
        """Build the LLM prompt for deep skill matching."""
        return f"""Evaluate if a developer can complete this bounty based on their skills.

Developer skills: {", ".join(self._skills)}

Bounty:
- Title: {bounty.title}
- Platform: {bounty.platform.value}
- Description: {bounty.description_snippet}
- Required skills: {", ".join(bounty.skills_required) or "not specified"}
- Difficulty: {bounty.difficulty.value}
- Reward: ${bounty.reward_usd}
- Labels: {", ".join(bounty.labels)}

Respond with JSON only:
{{
    "match_score": 0.0-1.0,
    "matching_skills": ["skill1", "skill2"],
    "missing_skills": ["skill3"],
    "estimated_hours": 5.0,
    "reasoning": "Brief explanation"
}}"""

    def parse_response(self, text: str, bounty_id: str) -> SkillMatch | None:
        """Parse an LLM response into a SkillMatch."""
        return _parse_llm_response(text, bounty_id)

    def should_auto_claim(self, bounty: Bounty, match: SkillMatch) -> bool:
        """Determine if a bounty should be auto-claimed.

        Auto-claim if:
        - Config allows it
        - Bounty is low effort (docs, typo, good-first-issue)
        - Match score >= 0.7
        - Reward is within auto-claim threshold
        """
        if not self.config.filters.auto_claim_low_effort:
            return False
        if not bounty.is_low_effort:
            return False
        if match.match_score < 0.7:
            return False
        if bounty.reward_usd > self.config.filters.auto_claim_max_reward_usd:
            return False
        return True
