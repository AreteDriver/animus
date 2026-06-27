"""Prospector orchestrator — ties hunters, evaluator, and feed together."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from animus_prospector.evaluator import Evaluator
from animus_prospector.feed import OpportunityFeed
from animus_prospector.hunters.aggregator_hunter import AggregatorHunter
from animus_prospector.hunters.base import BaseHunter
from animus_prospector.hunters.social_hunter import SocialHunter
from animus_prospector.hunters.web_hunter import WebHunter
from animus_prospector.models import Action, HunterResult, Opportunity

if TYPE_CHECKING:
    from animus_prospector.config import ProspectorConfig

logger = logging.getLogger("animus_prospector.orchestrator")


class Prospector:
    """Main orchestrator that runs the full hunt → evaluate → act pipeline.

    Usage:
        prospector = Prospector(config)
        report = await prospector.run_scan()
    """

    def __init__(self, config: ProspectorConfig):
        self.config = config
        self.feed = OpportunityFeed(config.data_dir)
        self.evaluator = Evaluator(config)
        self._hunters: list[BaseHunter] = []
        self._register_default_hunters()

    def _register_default_hunters(self) -> None:
        """Register all built-in hunters."""
        self._hunters = [
            AggregatorHunter(self.config),
            WebHunter(self.config),
            SocialHunter(self.config),
        ]

    def register_hunter(self, hunter: BaseHunter) -> None:
        """Register a custom hunter."""
        self._hunters.append(hunter)

    async def run_scan(self) -> ScanReport:
        """Run a full scan: hunt → deduplicate → evaluate → report."""
        logger.info(f"Starting scan with {len(self._hunters)} hunters")

        # Phase 1: Hunt
        all_results: list[HunterResult] = []
        total_found = 0
        for hunter in self._hunters:
            result = await hunter.scan()
            all_results.append(result)
            total_found += result.count
            if result.errors:
                for err in result.errors:
                    logger.warning(f"[{hunter.name}] {err}")

        # Phase 2: Deduplicate and store
        new_opportunities: list[Opportunity] = []
        for result in all_results:
            for opp in result.opportunities:
                if self.feed.record_opportunity(opp):
                    new_opportunities.append(opp)

        logger.info(f"Found {total_found} total, {len(new_opportunities)} new")

        # Phase 3: Evaluate new opportunities
        evaluations = await self.evaluator.evaluate_batch(new_opportunities)
        for ev in evaluations:
            self.feed.record_evaluation(ev)

        # Phase 4: Categorize results
        auto_execute = [e for e in evaluations if e.recommended_action == Action.AUTO_EXECUTE]
        add_to_faucet = [e for e in evaluations if e.recommended_action == Action.ADD_TO_FAUCET]
        flagged = [e for e in evaluations if e.recommended_action == Action.FLAG_FOR_REVIEW]
        skipped = [e for e in evaluations if e.recommended_action == Action.SKIP]

        report = ScanReport(
            total_scanned=total_found,
            new_discovered=len(new_opportunities),
            auto_execute=len(auto_execute),
            add_to_faucet=len(add_to_faucet),
            flagged_for_review=len(flagged),
            skipped=len(skipped),
            hunter_results=all_results,
            evaluations=evaluations,
        )

        logger.info(
            f"Scan complete: {report.new_discovered} new, "
            f"{report.auto_execute} auto-execute, "
            f"{report.add_to_faucet} new faucets, "
            f"{report.flagged_for_review} flagged"
        )

        return report

    async def health_check(self) -> dict[str, bool]:
        """Check health of all hunters."""
        results = {}
        for hunter in self._hunters:
            results[hunter.name] = await hunter.health_check()
        return results


class ScanReport:
    """Summary of a prospector scan run."""

    def __init__(
        self,
        total_scanned: int = 0,
        new_discovered: int = 0,
        auto_execute: int = 0,
        add_to_faucet: int = 0,
        flagged_for_review: int = 0,
        skipped: int = 0,
        hunter_results: list[HunterResult] | None = None,
        evaluations: list | None = None,
    ):
        self.total_scanned = total_scanned
        self.new_discovered = new_discovered
        self.auto_execute = auto_execute
        self.add_to_faucet = add_to_faucet
        self.flagged_for_review = flagged_for_review
        self.skipped = skipped
        self.hunter_results = hunter_results or []
        self.evaluations = evaluations or []

    def to_dict(self) -> dict:
        return {
            "total_scanned": self.total_scanned,
            "new_discovered": self.new_discovered,
            "auto_execute": self.auto_execute,
            "add_to_faucet": self.add_to_faucet,
            "flagged_for_review": self.flagged_for_review,
            "skipped": self.skipped,
            "hunters": [
                {
                    "name": hr.hunter_name,
                    "found": hr.count,
                    "errors": hr.errors,
                    "duration_s": hr.scan_duration_seconds,
                }
                for hr in self.hunter_results
            ],
        }
