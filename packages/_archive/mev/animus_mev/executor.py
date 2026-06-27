"""MEV bundle submission — DRY_RUN by default."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from animus_mev.config import MEVConfig
from animus_mev.models import ExtractionResult, MEVOpportunity, MEVType
from animus_mev.risk import RiskManager
from animus_mev.simulator import SimulationResult, Simulator

logger = logging.getLogger("animus_mev.executor")


class Executor:
    """Execute MEV opportunities.

    CRITICAL: Defaults to DRY_RUN mode. Live execution requires
    `config.execution.live = True` explicitly.

    Sandwich execution is ALWAYS blocked regardless of config.
    """

    def __init__(self, config: MEVConfig, risk_manager: RiskManager | None = None):
        self.config = config
        self.risk_manager = risk_manager or RiskManager(config)
        self.simulator = Simulator(config)
        self._execution_count = 0
        self._dry_run_count = 0

    @property
    def is_live(self) -> bool:
        return self.config.execution.live

    @property
    def execution_count(self) -> int:
        return self._execution_count

    @property
    def dry_run_count(self) -> int:
        return self._dry_run_count

    async def execute(self, opportunity: MEVOpportunity) -> ExtractionResult:
        """Execute an MEV opportunity.

        Flow:
        1. Block sandwich execution (ethical constraint)
        2. Check risk limits (gas budget, confidence, etc.)
        3. Simulate trade
        4. If DRY_RUN, return simulated result
        5. If LIVE, submit via private tx relay
        """
        # HARD BLOCK: Never execute sandwich attacks
        if opportunity.mev_type == MEVType.SANDWICH_DETECT:
            logger.warning(
                "Blocked sandwich execution for %s — monitoring only",
                opportunity.opportunity_id,
            )
            return ExtractionResult(
                opportunity_id=opportunity.opportunity_id,
                executed=False,
                error="sandwich_execution_blocked",
            )

        # Check risk limits
        risk_check = self.risk_manager.check(opportunity)
        if not risk_check.approved:
            logger.info(
                "Risk check failed for %s: %s",
                opportunity.opportunity_id,
                risk_check.reason,
            )
            return ExtractionResult(
                opportunity_id=opportunity.opportunity_id,
                executed=False,
                error=f"risk_rejected: {risk_check.reason}",
            )

        # Simulate
        sim_result = await self.simulator.simulate(opportunity)
        if not sim_result.profitable:
            return ExtractionResult(
                opportunity_id=opportunity.opportunity_id,
                executed=False,
                error=f"simulation_unprofitable: net={sim_result.net_profit_usd:.4f}",
            )

        # DRY RUN — default mode
        if not self.is_live:
            self._dry_run_count += 1
            logger.info(
                "[DRY RUN] Would execute %s on %s — est. profit: $%.4f",
                opportunity.mev_type.value,
                opportunity.chain.value,
                sim_result.net_profit_usd,
            )
            return _dry_run_result(opportunity, sim_result)

        # LIVE execution
        return await self._execute_live(opportunity, sim_result)

    async def _execute_live(
        self, opportunity: MEVOpportunity, sim_result: SimulationResult
    ) -> ExtractionResult:
        """Submit transaction via private relay.

        In production, this would use Flashbots Protect or similar
        private transaction submission to avoid being frontrun.
        """
        self._execution_count += 1
        self.risk_manager.record_gas_spend(sim_result.gas_estimate_usd)

        logger.info(
            "[LIVE] Executing %s on %s — est. profit: $%.4f",
            opportunity.mev_type.value,
            opportunity.chain.value,
            sim_result.net_profit_usd,
        )

        # Placeholder: actual implementation would use web3 + Flashbots
        # For now, return a simulated live result
        return ExtractionResult(
            opportunity_id=opportunity.opportunity_id,
            executed=True,
            tx_hash=None,  # Would be set by actual submission
            actual_profit_usd=sim_result.estimated_profit_usd * 0.8,  # Conservative
            gas_used_usd=sim_result.gas_estimate_usd,
            block_number=opportunity.trigger_tx.block_number,
            timestamp=datetime.now(UTC),
        )


def _dry_run_result(opportunity: MEVOpportunity, sim_result: SimulationResult) -> ExtractionResult:
    """Create a dry-run extraction result."""
    return ExtractionResult(
        opportunity_id=opportunity.opportunity_id,
        executed=False,
        actual_profit_usd=sim_result.estimated_profit_usd,
        gas_used_usd=sim_result.gas_estimate_usd,
        error="dry_run",
        timestamp=datetime.now(UTC),
    )
