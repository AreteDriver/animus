"""Tests for the prospector orchestrator."""

from unittest.mock import AsyncMock

import pytest

from animus_prospector.hunters.base import BaseHunter
from animus_prospector.models import (
    Chain,
    HunterResult,
    Opportunity,
    OpportunityType,
    SourceType,
)
from animus_prospector.orchestrator import Prospector, ScanReport


class TestScanReport:
    def test_creation(self):
        report = ScanReport(total_scanned=10, new_discovered=3, auto_execute=1)
        assert report.total_scanned == 10
        assert report.new_discovered == 3
        assert report.auto_execute == 1

    def test_defaults(self):
        report = ScanReport()
        assert report.total_scanned == 0
        assert report.hunter_results == []
        assert report.evaluations == []

    def test_to_dict(self):
        hr = HunterResult(hunter_name="test", errors=["err1"], scan_duration_seconds=1.5)
        report = ScanReport(
            total_scanned=5,
            new_discovered=2,
            auto_execute=1,
            add_to_faucet=0,
            flagged_for_review=1,
            skipped=0,
            hunter_results=[hr],
        )
        d = report.to_dict()
        assert d["total_scanned"] == 5
        assert d["new_discovered"] == 2
        assert len(d["hunters"]) == 1
        assert d["hunters"][0]["name"] == "test"
        assert d["hunters"][0]["errors"] == ["err1"]


class TestProspector:
    def test_init(self, sample_config):
        sample_config.ensure_dirs()
        p = Prospector(sample_config)
        assert len(p._hunters) == 3  # aggregator, web, social

    def test_register_hunter(self, sample_config):
        sample_config.ensure_dirs()
        p = Prospector(sample_config)

        class CustomHunter(BaseHunter):
            name = "custom"

            async def _scan(self):
                return HunterResult(hunter_name="custom")

        p.register_hunter(CustomHunter(sample_config))
        assert len(p._hunters) == 4

    @pytest.mark.asyncio
    async def test_run_scan(self, sample_config):
        sample_config.ensure_dirs()
        p = Prospector(sample_config)

        opp = Opportunity(
            title="Test opp",
            url="http://unique-test-url",
            opportunity_type=OpportunityType.AIRDROP,
            source_type=SourceType.AGGREGATOR,
            chain=Chain.ETH,
        )

        # Mock all hunters to return one opportunity
        for hunter in p._hunters:
            hunter._scan = AsyncMock(
                return_value=HunterResult(
                    hunter_name=hunter.name,
                    opportunities=[opp] if hunter.name == "aggregator" else [],
                )
            )

        # Mock evaluator
        p.evaluator._call_llm = AsyncMock(
            return_value='{"scam_probability": 0.1, "estimated_value_usd": 5.0, "effort_minutes": 10, "auto_actionable": true, "reasoning": "good", "confidence": 0.8}'
        )

        report = await p.run_scan()
        assert report.total_scanned >= 1
        assert report.new_discovered >= 1

    @pytest.mark.asyncio
    async def test_run_scan_dedup(self, sample_config):
        sample_config.ensure_dirs()
        p = Prospector(sample_config)

        opp = Opportunity(
            title="Dupe test",
            url="http://same-url",
            opportunity_type=OpportunityType.FAUCET,
            source_type=SourceType.AGGREGATOR,
        )

        # Both hunters return same opportunity
        for hunter in p._hunters:
            hunter._scan = AsyncMock(
                return_value=HunterResult(
                    hunter_name=hunter.name,
                    opportunities=[opp],
                )
            )

        p.evaluator._call_llm = AsyncMock(
            return_value='{"scam_probability": 0.5, "estimated_value_usd": 1.0, "effort_minutes": 5, "auto_actionable": false, "reasoning": "ok", "confidence": 0.5}'
        )

        report = await p.run_scan()
        assert report.new_discovered == 1  # deduped

    @pytest.mark.asyncio
    async def test_health_check(self, sample_config):
        sample_config.ensure_dirs()
        p = Prospector(sample_config)

        for hunter in p._hunters:
            hunter.health_check = AsyncMock(return_value=True)

        health = await p.health_check()
        assert len(health) == 3
        assert all(v is True for v in health.values())

    @pytest.mark.asyncio
    async def test_run_scan_with_hunter_errors(self, sample_config):
        sample_config.ensure_dirs()
        p = Prospector(sample_config)

        for hunter in p._hunters:
            hunter._scan = AsyncMock(
                return_value=HunterResult(
                    hunter_name=hunter.name,
                    opportunities=[],
                    errors=["test error"],
                )
            )

        report = await p.run_scan()
        assert report.total_scanned == 0
        assert report.new_discovered == 0


class TestTools:
    def test_tool_definitions(self):
        from animus_prospector.tools.prospector_tools import PROSPECTOR_TOOLS

        assert len(PROSPECTOR_TOOLS) == 3
        names = {t["name"] for t in PROSPECTOR_TOOLS}
        assert "prospector_scan" in names
        assert "prospector_feed" in names
        assert "prospector_health" in names
        for tool in PROSPECTOR_TOOLS:
            assert tool["category"] == "prospector"
