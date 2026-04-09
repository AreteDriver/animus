"""Tests for the opportunity feed."""

from animus_prospector.feed import OpportunityFeed
from animus_prospector.models import (
    Action,
    EvaluationResult,
    Opportunity,
    OpportunityType,
    SourceType,
)


class TestOpportunityFeed:
    def test_record_and_get_actionable(self, tmp_data_dir, sample_opportunity, sample_evaluation):
        feed = OpportunityFeed(tmp_data_dir)
        feed.record_opportunity(sample_opportunity)
        feed.record_evaluation(sample_evaluation)

        actionable = feed.get_actionable()
        assert len(actionable) == 1
        opp, ev = actionable[0]
        assert ev.recommended_action == Action.AUTO_EXECUTE

    def test_get_flagged(self, tmp_data_dir, sample_opportunity):
        feed = OpportunityFeed(tmp_data_dir)
        feed.record_opportunity(sample_opportunity)
        feed.record_evaluation(
            EvaluationResult(
                opportunity_id=sample_opportunity.opportunity_id,
                roi_score=0.3,
                scam_probability=0.15,
                recommended_action=Action.FLAG_FOR_REVIEW,
                reasoning="Needs review",
            )
        )

        flagged = feed.get_flagged()
        assert len(flagged) == 1

    def test_get_new_faucets(self, tmp_data_dir):
        feed = OpportunityFeed(tmp_data_dir)
        faucet = Opportunity(
            title="New faucet",
            url="http://faucet.new",
            opportunity_type=OpportunityType.FAUCET,
            source_type=SourceType.AGGREGATOR,
        )
        feed.record_opportunity(faucet)
        feed.record_evaluation(
            EvaluationResult(
                opportunity_id=faucet.opportunity_id,
                roi_score=1.0,
                scam_probability=0.05,
                recommended_action=Action.ADD_TO_FAUCET,
                reasoning="Good faucet",
            )
        )

        faucets = feed.get_new_faucets()
        assert len(faucets) == 1
        assert faucets[0].title == "New faucet"

    def test_get_summary(self, tmp_data_dir, sample_opportunity, sample_evaluation):
        feed = OpportunityFeed(tmp_data_dir)
        feed.record_opportunity(sample_opportunity)
        feed.record_evaluation(sample_evaluation)

        summary = feed.get_summary()
        assert summary["total_opportunities"] == 1
        assert summary["total_evaluations"] == 1
        assert "airdrop" in summary["by_type"]
        assert "auto_execute" in summary["by_action"]

    def test_empty_feed(self, tmp_data_dir):
        feed = OpportunityFeed(tmp_data_dir)
        assert feed.get_actionable() == []
        assert feed.get_flagged() == []
        assert feed.get_new_faucets() == []
        summary = feed.get_summary()
        assert summary["total_opportunities"] == 0

    def test_dedup(self, tmp_data_dir, sample_opportunity):
        feed = OpportunityFeed(tmp_data_dir)
        assert feed.record_opportunity(sample_opportunity) is True
        assert feed.record_opportunity(sample_opportunity) is False

    def test_sorted_by_roi(self, tmp_data_dir):
        feed = OpportunityFeed(tmp_data_dir)
        opp1 = Opportunity(
            title="Low",
            url="http://low",
            opportunity_type=OpportunityType.QUEST,
            source_type=SourceType.MANUAL,
        )
        opp2 = Opportunity(
            title="High",
            url="http://high",
            opportunity_type=OpportunityType.QUEST,
            source_type=SourceType.MANUAL,
        )
        feed.record_opportunity(opp1)
        feed.record_opportunity(opp2)
        feed.record_evaluation(
            EvaluationResult(
                opportunity_id=opp1.opportunity_id,
                roi_score=0.1,
                scam_probability=0.0,
                recommended_action=Action.AUTO_EXECUTE,
                reasoning="low",
            )
        )
        feed.record_evaluation(
            EvaluationResult(
                opportunity_id=opp2.opportunity_id,
                roi_score=10.0,
                scam_probability=0.0,
                recommended_action=Action.AUTO_EXECUTE,
                reasoning="high",
            )
        )

        actionable = feed.get_actionable()
        assert len(actionable) == 2
        assert actionable[0][1].roi_score > actionable[1][1].roi_score
