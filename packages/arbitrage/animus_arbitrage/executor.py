"""Trade execution engine with dry-run safety gate."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from animus_arbitrage.config import ArbitrageConfig
from animus_arbitrage.models import (
    ArbitrageOpportunity,
    ExecutionMode,
    TradeResult,
)
from animus_arbitrage.risk import RiskManager

logger = logging.getLogger("animus_arbitrage.executor")


class ExecutionBlockedError(Exception):
    """Raised when execution is blocked by safety checks."""


class TradeExecutor:
    """Executes arbitrage trades. DRY_RUN by default.

    Live execution requires explicit config: execution.mode = "live".
    All trades pass through the RiskManager before execution.
    """

    def __init__(self, config: ArbitrageConfig, risk_manager: RiskManager | None = None):
        self.config = config
        self.risk = risk_manager or RiskManager(config)

    @property
    def mode(self) -> ExecutionMode:
        return self.config.execution.mode

    @property
    def is_live(self) -> bool:
        return self.config.execution.is_live

    async def execute(self, opportunity: ArbitrageOpportunity) -> TradeResult:
        """Execute an arbitrage trade (or simulate in dry-run mode).

        Args:
            opportunity: The detected arbitrage opportunity to act on.

        Returns:
            TradeResult with execution details.

        Raises:
            ExecutionBlockedError: If risk checks fail and mode is live.
        """
        # Risk check
        risk_ok, risk_reason = self.risk.check_trade(opportunity)
        if not risk_ok:
            if self.is_live:
                raise ExecutionBlockedError(f"Risk check failed: {risk_reason}")
            return TradeResult(
                opportunity_id=opportunity.opportunity_id,
                executed=False,
                mode=self.mode,
                error=f"Risk blocked: {risk_reason}",
            )

        # Check expiration
        if opportunity.expires_at and datetime.now(UTC) > opportunity.expires_at:
            return TradeResult(
                opportunity_id=opportunity.opportunity_id,
                executed=False,
                mode=self.mode,
                error="Opportunity expired",
            )

        if not self.is_live:
            return self._dry_run(opportunity)

        return await self._live_execute(opportunity)

    def _dry_run(self, opportunity: ArbitrageOpportunity) -> TradeResult:
        """Simulate trade execution without touching any chain."""
        logger.info(
            "DRY RUN: %s buy@%s(%.4f) -> sell@%s(%.4f) | spread=%.2f%% profit=$%.2f",
            opportunity.buy_quote.pair.symbol,
            opportunity.buy_quote.dex.value,
            opportunity.buy_quote.price,
            opportunity.sell_quote.dex.value,
            opportunity.sell_quote.price,
            opportunity.spread_pct,
            opportunity.profit_after_gas_usd,
        )

        # Record the simulated trade for tracking
        self.risk.record_trade(opportunity)

        return TradeResult(
            opportunity_id=opportunity.opportunity_id,
            executed=False,
            mode=ExecutionMode.DRY_RUN,
            actual_profit_usd=opportunity.profit_after_gas_usd,
            slippage_pct=0.0,
        )

    async def _live_execute(self, opportunity: ArbitrageOpportunity) -> TradeResult:
        """Execute a real trade on-chain.

        This is a framework method. Actual chain interaction requires
        chain-specific adapters (web3 for EVM, pysui for Sui, etc).
        """
        logger.warning(
            "LIVE EXECUTION: %s buy@%s -> sell@%s | profit=$%.2f",
            opportunity.buy_quote.pair.symbol,
            opportunity.buy_quote.dex.value,
            opportunity.sell_quote.dex.value,
            opportunity.profit_after_gas_usd,
        )

        # Record the trade attempt
        self.risk.record_trade(opportunity)

        # Placeholder: actual chain interaction would go here
        # For now, return a result indicating the framework is ready
        # but no chain adapter is configured
        return TradeResult(
            opportunity_id=opportunity.opportunity_id,
            executed=False,
            mode=ExecutionMode.LIVE,
            error="No chain adapter configured. Install web3 (EVM) or pysui (Sui).",
        )
