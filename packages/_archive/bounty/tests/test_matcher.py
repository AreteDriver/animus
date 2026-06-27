"""Tests for animus_bounty.matcher."""

from __future__ import annotations

import json

from animus_bounty.config import BountyConfig, SkillProfile
from animus_bounty.matcher import (
    SkillMatcher,
    _estimate_hours,
    _keyword_match_score,
    _parse_llm_response,
)
from animus_bounty.models import Bounty, Difficulty, Platform


class TestKeywordMatchScore:
    def test_exact_skill_match(self):
        bounty = Bounty(
            title="Python REST API",
            url="https://example.com/1",
            platform=Platform.GITHUB,
            skills_required=["python", "fastapi"],
        )
        matching, missing, score = _keyword_match_score(bounty, ["python", "fastapi", "rust"])
        assert "python" in matching
        assert "fastapi" in matching
        assert missing == []
        assert score == 1.0

    def test_partial_match(self):
        bounty = Bounty(
            title="Solidity smart contract",
            url="https://example.com/2",
            platform=Platform.GITCOIN,
            skills_required=["solidity", "ethereum", "hardhat"],
        )
        matching, missing, score = _keyword_match_score(bounty, ["solidity", "python"])
        assert "solidity" in matching
        assert "ethereum" in missing or "hardhat" in missing
        assert 0 < score < 1

    def test_no_match(self):
        bounty = Bounty(
            title="Java Spring Boot",
            url="https://example.com/3",
            platform=Platform.GITHUB,
            skills_required=["java", "spring", "maven"],
        )
        matching, missing, score = _keyword_match_score(bounty, ["python", "rust"])
        assert matching == []
        assert len(missing) == 3
        assert score == 0.0

    def test_no_required_skills_with_matches(self):
        bounty = Bounty(
            title="Python documentation fix",
            url="https://example.com/4",
            platform=Platform.GITHUB,
            skills_required=[],
            labels=["python", "docs"],
        )
        matching, missing, score = _keyword_match_score(bounty, ["python", "rust"])
        assert "python" in matching
        assert score > 0

    def test_no_required_skills_no_matches(self):
        bounty = Bounty(
            title="Unknown task",
            url="https://example.com/5",
            platform=Platform.GENERIC,
            skills_required=[],
        )
        _, _, score = _keyword_match_score(bounty, ["python"])
        assert score == 0.5  # Neutral when unknown

    def test_case_insensitive(self):
        bounty = Bounty(
            title="PYTHON API",
            url="https://example.com/6",
            platform=Platform.GITHUB,
            skills_required=["Python"],
        )
        matching, missing, score = _keyword_match_score(bounty, ["python"])
        assert len(missing) == 0
        assert score == 1.0

    def test_description_snippet_searched(self):
        bounty = Bounty(
            title="Task",
            url="https://example.com/7",
            platform=Platform.GITHUB,
            description_snippet="This requires knowledge of rust and wasm",
            skills_required=[],
        )
        matching, _, _ = _keyword_match_score(bounty, ["rust"])
        assert "rust" in matching


class TestEstimateHours:
    def test_trivial(self):
        b = Bounty(title="T", url="u", platform=Platform.GITHUB, difficulty=Difficulty.TRIVIAL)
        assert _estimate_hours(b) < 1.0

    def test_easy(self):
        b = Bounty(title="T", url="u", platform=Platform.GITHUB, difficulty=Difficulty.EASY)
        hours = _estimate_hours(b)
        assert 1.0 <= hours <= 3.0

    def test_medium(self):
        b = Bounty(title="T", url="u", platform=Platform.GITHUB, difficulty=Difficulty.MEDIUM)
        hours = _estimate_hours(b)
        assert 5.0 <= hours <= 12.0

    def test_hard(self):
        b = Bounty(title="T", url="u", platform=Platform.GITHUB, difficulty=Difficulty.HARD)
        hours = _estimate_hours(b)
        assert hours >= 15.0

    def test_expert(self):
        b = Bounty(title="T", url="u", platform=Platform.GITHUB, difficulty=Difficulty.EXPERT)
        hours = _estimate_hours(b)
        assert hours >= 40.0

    def test_long_description_increases(self):
        short = Bounty(title="T", url="u", platform=Platform.GITHUB, description_snippet="x")
        long = Bounty(
            title="T",
            url="u2",
            platform=Platform.GITHUB,
            description_snippet="x" * 300,
        )
        assert _estimate_hours(long) > _estimate_hours(short)


class TestParseLLMResponse:
    def test_valid_json(self):
        response = json.dumps(
            {
                "match_score": 0.9,
                "matching_skills": ["python"],
                "missing_skills": [],
                "estimated_hours": 2.0,
                "reasoning": "Good match",
            }
        )
        result = _parse_llm_response(response, "test_id")
        assert result is not None
        assert result.match_score == 0.9
        assert result.bounty_id == "test_id"

    def test_json_in_code_block(self):
        response = """Here is my analysis:
```json
{
    "match_score": 0.7,
    "matching_skills": ["rust"],
    "missing_skills": ["solidity"],
    "estimated_hours": 10.0,
    "reasoning": "Partial match"
}
```"""
        result = _parse_llm_response(response, "test2")
        assert result is not None
        assert result.match_score == 0.7

    def test_invalid_json(self):
        result = _parse_llm_response("This is not JSON at all", "test3")
        assert result is None

    def test_missing_fields_defaults(self):
        response = json.dumps({"match_score": 0.5})
        result = _parse_llm_response(response, "test4")
        assert result is not None
        assert result.matching_skills == []
        assert result.estimated_hours == 0.0

    def test_code_block_invalid_json(self):
        """Code block exists but JSON inside is invalid."""
        response = """```json
{not valid json}
```"""
        result = _parse_llm_response(response, "test_invalid_block")
        assert result is None

    def test_code_block_without_json_tag(self):
        response = """```
{"match_score": 0.6, "matching_skills": [], "missing_skills": [], "estimated_hours": 1.0, "reasoning": "ok"}
```"""
        result = _parse_llm_response(response, "test5")
        assert result is not None
        assert result.match_score == 0.6


class TestSkillMatcher:
    def test_quick_match(self, sample_config, sample_bounty):
        matcher = SkillMatcher(sample_config)
        result = matcher.quick_match(sample_bounty)
        assert result.bounty_id == sample_bounty.bounty_id
        assert result.match_score >= 0
        assert result.estimated_hours > 0

    def test_quick_match_high_score_for_matching(self, sample_config):
        bounty = Bounty(
            title="Python FastAPI endpoint",
            url="https://example.com/python",
            platform=Platform.GITHUB,
            skills_required=["python", "fastapi"],
        )
        matcher = SkillMatcher(sample_config)
        result = matcher.quick_match(bounty)
        assert result.match_score >= 0.5

    def test_build_prompt(self, sample_config, sample_bounty):
        matcher = SkillMatcher(sample_config)
        prompt = matcher.build_prompt(sample_bounty)
        assert "python" in prompt
        assert sample_bounty.title in prompt
        assert "match_score" in prompt

    def test_parse_response(self, sample_config, sample_bounty):
        matcher = SkillMatcher(sample_config)
        response = json.dumps(
            {
                "match_score": 0.8,
                "matching_skills": ["python"],
                "missing_skills": [],
                "estimated_hours": 1.0,
                "reasoning": "Good match",
            }
        )
        result = matcher.parse_response(response, sample_bounty.bounty_id)
        assert result is not None
        assert result.match_score == 0.8

    def test_should_auto_claim_true(self, sample_config, sample_bounty):
        matcher = SkillMatcher(sample_config)
        match = matcher.quick_match(sample_bounty)
        match.match_score = 0.9  # Force high score
        # sample_bounty has "good first issue" label and is Difficulty.EASY
        assert matcher.should_auto_claim(sample_bounty, match)

    def test_should_auto_claim_disabled(self, tmp_data_dir):
        config = BountyConfig(data_dir=tmp_data_dir)
        config.filters.auto_claim_low_effort = False
        matcher = SkillMatcher(config)
        bounty = Bounty(
            title="Easy",
            url="https://example.com/e",
            platform=Platform.GITHUB,
            labels=["good first issue"],
            difficulty=Difficulty.EASY,
        )
        match = matcher.quick_match(bounty)
        match.match_score = 0.9
        assert not matcher.should_auto_claim(bounty, match)

    def test_should_auto_claim_hard_bounty(self, sample_config, hard_bounty):
        matcher = SkillMatcher(sample_config)
        match = matcher.quick_match(hard_bounty)
        match.match_score = 0.9
        assert not matcher.should_auto_claim(hard_bounty, match)

    def test_should_auto_claim_low_score(self, sample_config, sample_bounty):
        matcher = SkillMatcher(sample_config)
        match = matcher.quick_match(sample_bounty)
        match.match_score = 0.3
        assert not matcher.should_auto_claim(sample_bounty, match)

    def test_should_auto_claim_over_threshold(self, tmp_data_dir, sample_bounty):
        config = BountyConfig(data_dir=tmp_data_dir)
        config.filters.auto_claim_max_reward_usd = 10.0
        matcher = SkillMatcher(config)
        sample_bounty.reward_usd = 50.0
        match = matcher.quick_match(sample_bounty)
        match.match_score = 0.9
        assert not matcher.should_auto_claim(sample_bounty, match)

    def test_custom_skills(self, tmp_data_dir):
        config = BountyConfig(
            data_dir=tmp_data_dir,
            skills=SkillProfile(languages=["go"], frameworks=["gin"], domains=["microservices"]),
        )
        matcher = SkillMatcher(config)
        bounty = Bounty(
            title="Go microservice",
            url="https://example.com/go",
            platform=Platform.GITHUB,
            skills_required=["go", "gin"],
        )
        result = matcher.quick_match(bounty)
        assert "go" in result.matching_skills
