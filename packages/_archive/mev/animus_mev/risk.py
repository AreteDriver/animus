"""Risk management for MEV execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from animus_mev.config import MEVConfig
from animus_mev.models import MEVOpportunity, MEVType

logger = logging.getLogger("animus_mev.risk")


@dataclass
class RiskCheck:
    """Result of a risk assessment."""

    approved: bool
    reason: str
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


class RiskManager:
    """Enforce risk limits on MEV execution.

    Checks:
    - Sandwich execution blocked (always)
    - Minimum confidence threshold
    - Minimum profit threshold
    - Maximum gas per transaction
    - Daily gas budget
    - Maximum concurrent transactions
    """

    def __init__(self, config: MEVConfig):
        self.config = config
        self._daily_gas_spent: float = 0.0
        self._daily_reset: datetime = datetime.now(UTC)
        self._active_txs: int = 0

    @property
    def daily_gas_spent(self) -> float:
        self._maybe_reset_daily()
        return self._daily_gas_spent

    @property
    def daily_gas_remaining(self) -> float:
        return max(0.0, self.config.risk.daily_gas_budget_usd - self.daily_gas_spent)

    @property
    def active_txs(self) -> int:
        return self._active_txs

    def check(self, opportunity: MEVOpportunity) -> RiskCheck:
        """Run all risk checks on an opportunity."""
        self._maybe_reset_daily()

        passed: list[str] = []
        failed: list[str] = []

        # 1. Sandwich execution is ALWAYS blocked
        if opportunity.mev_type == MEVType.SANDWICH_DETECT:
            failed.append("sandwich_execution_blocked")
            return RiskCheck(
                approved=False,
                reason="sandwich execution is blocked by design",
                checks_passed=passed,
                checks_failed=failed,
            )
        passed.append("not_sandwich")

        # 2. Minimum confidence
        if opportunity.confidence < self.config.risk.min_confidence:
            failed.append(
                f"confidence_too_low: {opportunity.confidence:.2f} < {self.config.risk.min_confidence}"
            )
        else:
            passed.append("confidence_ok")

        # 3. Minimum profit
        if opportunity.net_profit_usd < self.config.risk.min_profit_usd:
            failed.append(
                f"profit_too_low: ${opportunity.net_profit_usd:.4f} < ${self.config.risk.min_profit_usd}"
            )
        else:
            passed.append("profit_ok")

        # 4. Max gas per tx
        if opportunity.gas_cost_usd > self.config.risk.max_gas_per_tx_usd:
            failed.append(
                f"gas_too_high: ${opportunity.gas_cost_usd:.4f} > ${self.config.risk.max_gas_per_tx_usd}"
            )
        else:
            passed.append("gas_per_tx_ok")

        # 5. Daily gas budget
        if self._daily_gas_spent + opportunity.gas_cost_usd > self.config.risk.daily_gas_budget_usd:
            failed.append(
                f"daily_budget_exceeded: ${self._daily_gas_spent:.2f} + ${opportunity.gas_cost_usd:.4f} > ${self.config.risk.daily_gas_budget_usd}"
            )
        else:
            passed.append("daily_budget_ok")

        # 6. Concurrent tx limit
        if self._active_txs >= self.config.risk.max_concurrent_txs:
            failed.append(
                f"too_many_concurrent: {self._active_txs} >= {self.config.risk.max_concurrent_txs}"
            )
        else:
            passed.append("concurrent_ok")

        if failed:
            return RiskCheck(
                approved=False,
                reason=failed[0],
                checks_passed=passed,
                checks_failed=failed,
            )

        return RiskCheck(
            approved=True,
            reason="all_checks_passed",
            checks_passed=passed,
            checks_failed=failed,
        )

    def record_gas_spend(self, amount_usd: float) -> None:
        """Record gas expenditure."""
        self._maybe_reset_daily()
        self._daily_gas_spent += amount_usd
        logger.debug("Gas spent: +$%.4f (daily total: $%.2f)", amount_usd, self._daily_gas_spent)

    def increment_active(self) -> None:
        """Track a new active transaction."""
        self._active_txs += 1

    def decrement_active(self) -> None:
        """Mark a transaction as completed."""
        self._active_txs = max(0, self._active_txs - 1)

    def _maybe_reset_daily(self) -> None:
        """Reset daily counters if a new day has started."""
        now = datetime.now(UTC)
        if now - self._daily_reset > timedelta(hours=24):
            self._daily_gas_spent = 0.0
            self._daily_reset = now
            logger.info("Daily gas budget reset")

    def get_status(self) -> dict:
        """Get current risk status."""
        self._maybe_reset_daily()
        return {
            "daily_gas_spent_usd": self._daily_gas_spent,
            "daily_gas_remaining_usd": self.daily_gas_remaining,
            "daily_gas_budget_usd": self.config.risk.daily_gas_budget_usd,
            "active_txs": self._active_txs,
            "max_concurrent_txs": self.config.risk.max_concurrent_txs,
            "max_gas_per_tx_usd": self.config.risk.max_gas_per_tx_usd,
            "min_profit_usd": self.config.risk.min_profit_usd,
            "min_confidence": self.config.risk.min_confidence,
            "sandwich_blocked": self.config.risk.sandwich_execution_blocked,
        }
