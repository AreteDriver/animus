"""Tests for data models."""

from datetime import UTC, datetime, timedelta

from animus_prospector.models import (
    Action,
    Chain,
    EvaluationResult,
    HunterResult,
    Opportunity,
    OpportunityStatus,
    OpportunityType,
    SourceType,
)


class TestOpportunityType:
    def test_values(self):
        assert OpportunityType.FAUCET.value == "faucet"
        assert OpportunityType.AIRDROP.value == "airdrop"
        assert OpportunityType.GIVEAWAY.value == "giveaway"
        assert OpportunityType.QUEST.value == "quest"
        assert OpportunityType.BOUNTY.value == "bounty"
        assert OpportunityType.GRANT.value == "grant"
        assert OpportunityType.TESTNET.value == "testnet"
        assert OpportunityType.REFERRAL.value == "referral"
        assert OpportunityType.DEFI_YIELD.value == "defi_yield"


class TestChain:
    def test_values(self):
        assert Chain.BTC.value == "btc"
        assert Chain.SUI.value == "sui"
        assert Chain.BASE.value == "base"
        assert Chain.MULTI.value == "multi"
        assert Chain.UNKNOWN.value == "unknown"
        assert Chain.ARB.value == "arb"
        assert Chain.OP.value == "op"
        assert Chain.MATIC.value == "matic"
        assert Chain.SOL.value == "sol"
        assert Chain.ETH.value == "eth"
        assert Chain.BASE_SEPOLIA.value == "base_sepolia"
        assert Chain.SUI_TESTNET.value == "sui_testnet"


class TestAction:
    def test_values(self):
        assert Action.AUTO_EXECUTE.value == "auto_execute"
        assert Action.ADD_TO_FAUCET.value == "add_to_faucet"
        assert Action.FLAG_FOR_REVIEW.value == "flag_for_review"
        assert Action.SKIP.value == "skip"
        assert Action.EXPIRED.value == "expired"


class TestSourceType:
    def test_values(self):
        assert SourceType.TWITTER.value == "twitter"
        assert SourceType.REDDIT.value == "reddit"
        assert SourceType.AGGREGATOR.value == "aggregator"
        assert SourceType.WEB_SCRAPE.value == "web_scrape"
        assert SourceType.API.value == "api"
        assert SourceType.RSS.value == "rss"
        assert SourceType.DISCORD.value == "discord"
        assert SourceType.TELEGRAM.value == "telegram"
        assert SourceType.MANUAL.value == "manual"


class TestOpportunityStatus:
    def test_values(self):
        assert OpportunityStatus.DISCOVERED.value == "discovered"
        assert OpportunityStatus.EVALUATING.value == "evaluating"
        assert OpportunityStatus.APPROVED.value == "approved"
        assert OpportunityStatus.COMPLETED.value == "completed"
        assert OpportunityStatus.FAILED.value == "failed"
        assert OpportunityStatus.SKIPPED.value == "skipped"
        assert OpportunityStatus.EXPIRED.value == "expired"
        assert OpportunityStatus.EXECUTING.value == "executing"


class TestOpportunity:
    def test_creation(self, sample_opportunity):
        assert sample_opportunity.title == "Free SUI Airdrop — Testnet Interaction"
        assert sample_opportunity.chain == Chain.SUI
        assert sample_opportunity.opportunity_type == OpportunityType.AIRDROP

    def test_opportunity_id_deterministic(self, sample_opportunity):
        id1 = sample_opportunity.opportunity_id
        id2 = sample_opportunity.opportunity_id
        assert id1 == id2
        assert len(id1) == 16

    def test_opportunity_id_unique(self, sample_opportunity, scam_opportunity):
        assert sample_opportunity.opportunity_id != scam_opportunity.opportunity_id

    def test_roi_score(self, sample_opportunity):
        assert sample_opportunity.roi_score == 0.5  # 5.0 / 10

    def test_roi_score_zero_effort(self):
        opp = Opportunity(
            title="x",
            url="http://x",
            opportunity_type=OpportunityType.FAUCET,
            source_type=SourceType.MANUAL,
            estimated_value_usd=5.0,
            effort_minutes=0,
        )
        assert opp.roi_score == 500.0  # 5.0 * 100

    def test_roi_score_zero_value_zero_effort(self):
        opp = Opportunity(
            title="x",
            url="http://x",
            opportunity_type=OpportunityType.FAUCET,
            source_type=SourceType.MANUAL,
            estimated_value_usd=0.0,
            effort_minutes=0,
        )
        assert opp.roi_score == 0.0

    def test_is_expired_true(self):
        opp = Opportunity(
            title="x",
            url="http://x",
            opportunity_type=OpportunityType.AIRDROP,
            source_type=SourceType.MANUAL,
            expiry=datetime.now(UTC) - timedelta(days=1),
        )
        assert opp.is_expired is True

    def test_is_expired_false(self):
        opp = Opportunity(
            title="x",
            url="http://x",
            opportunity_type=OpportunityType.AIRDROP,
            source_type=SourceType.MANUAL,
            expiry=datetime.now(UTC) + timedelta(days=1),
        )
        assert opp.is_expired is False

    def test_is_expired_none(self):
        opp = Opportunity(
            title="x",
            url="http://x",
            opportunity_type=OpportunityType.AIRDROP,
            source_type=SourceType.MANUAL,
        )
        assert opp.is_expired is False

    def test_is_expired_naive_datetime(self):
        opp = Opportunity(
            title="x",
            url="http://x",
            opportunity_type=OpportunityType.AIRDROP,
            source_type=SourceType.MANUAL,
            expiry=datetime(2020, 1, 1),  # naive datetime
        )
        assert opp.is_expired is False  # can't compare, returns False

    def test_to_dict(self, sample_opportunity):
        d = sample_opportunity.to_dict()
        assert d["title"] == "Free SUI Airdrop — Testnet Interaction"
        assert d["chain"] == "sui"
        assert d["opportunity_type"] == "airdrop"
        assert d["source_type"] == "aggregator"
        assert d["opportunity_id"] == sample_opportunity.opportunity_id
        assert d["tags"] == ["sui", "testnet"]

    def test_from_dict(self, sample_opportunity):
        d = sample_opportunity.to_dict()
        restored = Opportunity.from_dict(d)
        assert restored.title == sample_opportunity.title
        assert restored.chain == sample_opportunity.chain
        assert restored.opportunity_id == sample_opportunity.opportunity_id

    def test_from_dict_minimal(self):
        d = {
            "title": "test",
            "url": "http://test",
            "opportunity_type": "faucet",
            "source_type": "manual",
        }
        opp = Opportunity.from_dict(d)
        assert opp.chain == Chain.UNKNOWN
        assert opp.estimated_value_usd == 0.0
        assert opp.status == OpportunityStatus.DISCOVERED

    def test_from_dict_with_expiry(self):
        d = {
            "title": "test",
            "url": "http://test",
            "opportunity_type": "airdrop",
            "source_type": "api",
            "expiry": "2026-12-01T00:00:00+00:00",
        }
        opp = Opportunity.from_dict(d)
        assert opp.expiry is not None

    def test_defaults(self):
        opp = Opportunity(
            title="x",
            url="http://x",
            opportunity_type=OpportunityType.FAUCET,
            source_type=SourceType.MANUAL,
        )
        assert opp.status == OpportunityStatus.DISCOVERED
        assert opp.risk_score == 0.0
        assert opp.requirements == []
        assert opp.raw_data == {}
        assert opp.tags == []
        assert opp.source_url == ""
        assert opp.auto_actionable is False


class TestEvaluationResult:
    def test_creation(self, sample_evaluation):
        assert sample_evaluation.recommended_action == Action.AUTO_EXECUTE
        assert sample_evaluation.scam_probability == 0.1
        assert sample_evaluation.confidence == 0.8

    def test_to_dict(self, sample_evaluation):
        d = sample_evaluation.to_dict()
        assert d["recommended_action"] == "auto_execute"
        assert d["scam_probability"] == 0.1
        assert "evaluated_at" in d

    def test_from_dict(self, sample_evaluation):
        d = sample_evaluation.to_dict()
        restored = EvaluationResult.from_dict(d)
        assert restored.opportunity_id == sample_evaluation.opportunity_id
        assert restored.recommended_action == sample_evaluation.recommended_action

    def test_from_dict_minimal(self):
        d = {
            "opportunity_id": "abc123",
            "recommended_action": "skip",
        }
        ev = EvaluationResult.from_dict(d)
        assert ev.scam_probability == 0.5
        assert ev.confidence == 0.0
        assert ev.reasoning == ""

    def test_defaults(self):
        ev = EvaluationResult(
            opportunity_id="x",
            roi_score=1.0,
            scam_probability=0.0,
            recommended_action=Action.AUTO_EXECUTE,
            reasoning="test",
        )
        assert ev.confidence == 0.0
        assert isinstance(ev.evaluated_at, datetime)


class TestHunterResult:
    def test_empty(self):
        hr = HunterResult(hunter_name="test")
        assert hr.count == 0
        assert hr.new_count == 0
        assert hr.errors == []

    def test_with_opportunities(self, sample_opportunity):
        hr = HunterResult(
            hunter_name="test",
            opportunities=[sample_opportunity],
        )
        assert hr.count == 1
        assert hr.new_count == 1  # status is DISCOVERED

    def test_new_count_mixed(self, sample_opportunity):
        completed = Opportunity(
            title="done",
            url="http://done",
            opportunity_type=OpportunityType.FAUCET,
            source_type=SourceType.MANUAL,
            status=OpportunityStatus.COMPLETED,
        )
        hr = HunterResult(
            hunter_name="test",
            opportunities=[sample_opportunity, completed],
        )
        assert hr.count == 2
        assert hr.new_count == 1
