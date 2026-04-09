"""Tests for the LLM evaluator."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_prospector.evaluator import Evaluator
from animus_prospector.models import (
    Action,
    Opportunity,
    OpportunityType,
    SourceType,
)


class TestHeuristicCheck:
    def test_detects_scam_keywords(self, sample_config, scam_opportunity):
        evaluator = Evaluator(sample_config)
        result = evaluator._heuristic_check(scam_opportunity)
        assert result is not None
        assert result.recommended_action == Action.SKIP
        assert result.scam_probability >= 0.8

    def test_passes_clean_opportunity(self, sample_config, sample_opportunity):
        evaluator = Evaluator(sample_config)
        result = evaluator._heuristic_check(sample_opportunity)
        assert result is None  # passes to LLM

    def test_detects_expired(self, sample_config):
        opp = Opportunity(
            title="Old airdrop",
            url="http://old",
            opportunity_type=OpportunityType.AIRDROP,
            source_type=SourceType.MANUAL,
            expiry=datetime.now(UTC) - timedelta(days=1),
        )
        evaluator = Evaluator(sample_config)
        result = evaluator._heuristic_check(opp)
        assert result is not None
        assert result.recommended_action == Action.EXPIRED

    def test_scam_in_raw_data(self, sample_config):
        opp = Opportunity(
            title="Free tokens",
            url="http://legit-looking",
            opportunity_type=OpportunityType.GIVEAWAY,
            source_type=SourceType.TWITTER,
            raw_data={"snippet": "send eth to receive 10x returns"},
        )
        evaluator = Evaluator(sample_config)
        result = evaluator._heuristic_check(opp)
        assert result is not None
        assert result.recommended_action == Action.SKIP

    def test_private_key_scam(self, sample_config):
        opp = Opportunity(
            title="Enter your private key to claim rewards",
            url="http://phish",
            opportunity_type=OpportunityType.AIRDROP,
            source_type=SourceType.TELEGRAM,
        )
        evaluator = Evaluator(sample_config)
        result = evaluator._heuristic_check(opp)
        assert result is not None
        assert result.recommended_action == Action.SKIP

    def test_seed_phrase_scam(self, sample_config):
        opp = Opportunity(
            title="Import seed phrase to verify wallet",
            url="http://phish2",
            opportunity_type=OpportunityType.AIRDROP,
            source_type=SourceType.DISCORD,
        )
        evaluator = Evaluator(sample_config)
        result = evaluator._heuristic_check(opp)
        assert result is not None


class TestDecideAction:
    def test_auto_execute(self, sample_config):
        evaluator = Evaluator(sample_config)
        opp = Opportunity(
            title="Quest",
            url="http://quest",
            opportunity_type=OpportunityType.QUEST,
            source_type=SourceType.AGGREGATOR,
            estimated_value_usd=10.0,
            effort_minutes=5,
            auto_actionable=True,
        )
        action = evaluator._decide_action(opp, scam_prob=0.05)
        assert action == Action.AUTO_EXECUTE

    def test_add_to_faucet(self, sample_config):
        evaluator = Evaluator(sample_config)
        opp = Opportunity(
            title="New faucet",
            url="http://faucet",
            opportunity_type=OpportunityType.FAUCET,
            source_type=SourceType.AGGREGATOR,
            estimated_value_usd=5.0,
            effort_minutes=1,
            auto_actionable=True,
        )
        action = evaluator._decide_action(opp, scam_prob=0.05)
        assert action == Action.ADD_TO_FAUCET

    def test_skip_high_risk(self, sample_config):
        evaluator = Evaluator(sample_config)
        opp = Opportunity(
            title="Risky",
            url="http://risky",
            opportunity_type=OpportunityType.AIRDROP,
            source_type=SourceType.TWITTER,
            estimated_value_usd=100.0,
            effort_minutes=5,
            auto_actionable=True,
        )
        action = evaluator._decide_action(opp, scam_prob=0.9)
        assert action == Action.SKIP

    def test_skip_too_much_effort(self, sample_config):
        evaluator = Evaluator(sample_config)
        opp = Opportunity(
            title="Long task",
            url="http://long",
            opportunity_type=OpportunityType.BOUNTY,
            source_type=SourceType.WEB_SCRAPE,
            estimated_value_usd=10.0,
            effort_minutes=200,
        )
        action = evaluator._decide_action(opp, scam_prob=0.1)
        assert action == Action.SKIP

    def test_skip_low_value(self, sample_config):
        evaluator = Evaluator(sample_config)
        opp = Opportunity(
            title="Tiny",
            url="http://tiny",
            opportunity_type=OpportunityType.FAUCET,
            source_type=SourceType.AGGREGATOR,
            estimated_value_usd=0.001,
            effort_minutes=5,
        )
        action = evaluator._decide_action(opp, scam_prob=0.1)
        assert action == Action.SKIP

    def test_flag_for_review(self, sample_config):
        evaluator = Evaluator(sample_config)
        opp = Opportunity(
            title="Interesting",
            url="http://interesting",
            opportunity_type=OpportunityType.GRANT,
            source_type=SourceType.WEB_SCRAPE,
            estimated_value_usd=50.0,
            effort_minutes=60,
            auto_actionable=False,
        )
        action = evaluator._decide_action(opp, scam_prob=0.1)
        assert action == Action.FLAG_FOR_REVIEW


class TestFallbackEvaluation:
    def test_returns_flag_for_review(self, sample_config, sample_opportunity):
        evaluator = Evaluator(sample_config)
        result = evaluator._fallback_evaluation(sample_opportunity)
        assert result.recommended_action == Action.FLAG_FOR_REVIEW
        assert result.scam_probability == 0.5
        assert result.confidence == 0.2


class TestParseLLMResponse:
    def test_valid_json(self, sample_config, sample_opportunity):
        evaluator = Evaluator(sample_config)
        response = '{"scam_probability": 0.1, "estimated_value_usd": 5.0, "effort_minutes": 10, "auto_actionable": true, "reasoning": "Legit", "confidence": 0.8}'
        result = evaluator._parse_llm_response(sample_opportunity, response)
        assert result.scam_probability == 0.1
        assert result.confidence == 0.8

    def test_json_in_text(self, sample_config, sample_opportunity):
        evaluator = Evaluator(sample_config)
        response = 'Here is my analysis: {"scam_probability": 0.3, "estimated_value_usd": 2.0, "effort_minutes": 5, "auto_actionable": false, "reasoning": "OK", "confidence": 0.6} Done.'
        result = evaluator._parse_llm_response(sample_opportunity, response)
        assert result.scam_probability == 0.3

    def test_invalid_json_fallback(self, sample_config, sample_opportunity):
        evaluator = Evaluator(sample_config)
        result = evaluator._parse_llm_response(sample_opportunity, "not json at all")
        assert result.recommended_action == Action.FLAG_FOR_REVIEW
        assert result.confidence == 0.2

    def test_partial_json(self, sample_config, sample_opportunity):
        evaluator = Evaluator(sample_config)
        result = evaluator._parse_llm_response(sample_opportunity, '{"broken')
        assert result.recommended_action == Action.FLAG_FOR_REVIEW


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_evaluate_scam_short_circuits(self, sample_config, scam_opportunity):
        evaluator = Evaluator(sample_config)
        result = await evaluator.evaluate(scam_opportunity)
        assert result.recommended_action == Action.SKIP

    @pytest.mark.asyncio
    async def test_evaluate_calls_llm(self, sample_config, sample_opportunity):
        evaluator = Evaluator(sample_config)
        evaluator._call_llm = AsyncMock(
            return_value='{"scam_probability": 0.05, "estimated_value_usd": 5.0, "effort_minutes": 10, "auto_actionable": true, "reasoning": "Good", "confidence": 0.9}'
        )
        result = await evaluator.evaluate(sample_opportunity)
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_evaluate_llm_failure_fallback(self, sample_config, sample_opportunity):
        evaluator = Evaluator(sample_config)
        evaluator._call_llm = AsyncMock(side_effect=Exception("LLM down"))
        result = await evaluator.evaluate(sample_opportunity)
        assert result.recommended_action == Action.FLAG_FOR_REVIEW
        assert result.confidence == 0.2

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, sample_config, sample_opportunity, scam_opportunity):
        evaluator = Evaluator(sample_config)
        evaluator._call_llm = AsyncMock(
            return_value='{"scam_probability": 0.1, "estimated_value_usd": 3.0, "effort_minutes": 5, "auto_actionable": true, "reasoning": "ok", "confidence": 0.7}'
        )
        results = await evaluator.evaluate_batch([sample_opportunity, scam_opportunity])
        assert len(results) == 2


class TestCallLLM:
    @pytest.mark.asyncio
    async def test_unsupported_provider(self, sample_config):
        sample_config.llm.provider = "unsupported"
        evaluator = Evaluator(sample_config)
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            await evaluator._call_llm("test prompt")

    @pytest.mark.asyncio
    async def test_ollama_call(self, sample_config):
        evaluator = Evaluator(sample_config)
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": '{"test": true}'}
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await evaluator._call_llm("test")
            assert result == '{"test": true}'
