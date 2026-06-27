"""Shared test fixtures for animus_prospector."""

from __future__ import annotations

import pytest

from animus_prospector.config import ProspectorConfig
from animus_prospector.models import (
    Action,
    Chain,
    EvaluationResult,
    Opportunity,
    OpportunityType,
    SourceType,
)


@pytest.fixture
def tmp_data_dir(tmp_path):
    data_dir = tmp_path / "prospector_data"
    data_dir.mkdir()
    (data_dir / "opportunities").mkdir()
    (data_dir / "evaluations").mkdir()
    return data_dir


@pytest.fixture
def sample_config(tmp_data_dir):
    return ProspectorConfig(data_dir=tmp_data_dir)


@pytest.fixture
def sample_opportunity():
    return Opportunity(
        title="Free SUI Airdrop — Testnet Interaction",
        url="https://example.com/sui-airdrop",
        opportunity_type=OpportunityType.AIRDROP,
        source_type=SourceType.AGGREGATOR,
        source_url="https://airdrops.io/",
        chain=Chain.SUI,
        estimated_value_usd=5.0,
        effort_minutes=10,
        risk_score=0.1,
        auto_actionable=True,
        tags=["sui", "testnet"],
    )


@pytest.fixture
def scam_opportunity():
    return Opportunity(
        title="SEND ETH TO RECEIVE 10X — GUARANTEED PROFIT",
        url="https://scam.example.com/double",
        opportunity_type=OpportunityType.GIVEAWAY,
        source_type=SourceType.TWITTER,
        source_url="https://twitter.com/scammer",
        chain=Chain.ETH,
        estimated_value_usd=100.0,
        tags=["suspicious"],
    )


@pytest.fixture
def sample_evaluation(sample_opportunity):
    return EvaluationResult(
        opportunity_id=sample_opportunity.opportunity_id,
        roi_score=0.5,
        scam_probability=0.1,
        recommended_action=Action.AUTO_EXECUTE,
        reasoning="Known platform, reasonable reward",
        confidence=0.8,
    )
