"""Sandwich detection — MONITORING ONLY.

Detects sandwich attack patterns in the mempool for analysis.
Execution is BLOCKED by design — this is an ethical constraint.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from animus_mev.config import MEVConfig
from animus_mev.models import MempoolTx, MEVOpportunity, MEVType
from animus_mev.strategies.base import BaseStrategy

logger = logging.getLogger("animus_mev.strategies.sandwich")


class SandwichDetector(BaseStrategy):
    """Detect sandwich attack opportunities — MONITOR ONLY.

    This strategy identifies transactions that could be sandwiched
    (frontrun + backrun), but NEVER generates executable opportunities.
    All detected opportunities are flagged as SANDWICH_DETECT type,
    which the executor refuses to execute.

    Purpose: analytics, academic research, and monitoring the MEV
    landscape on L2 chains.
    """

    def __init__(self, config: MEVConfig):
        super().__init__(config)
        self._recent_txs: list[MempoolTx] = []
        self._max_window = 50

    @property
    def name(self) -> str:
        return "sandwich_detector"

    def is_applicable(self, tx: MempoolTx) -> bool:
        """Any DEX swap is potentially sandwichable."""
        return tx.dex_detected and tx.swap_amount_usd > 0

    async def evaluate(self, tx: MempoolTx) -> MEVOpportunity | None:
        """Detect if a transaction is vulnerable to sandwich attack.

        IMPORTANT: Returns SANDWICH_DETECT type, which the executor
        will REFUSE to execute. This is monitoring only.
        """
        if not self.is_applicable(tx):
            return None

        # Track recent txs for pattern analysis
        self._recent_txs.append(tx)
        if len(self._recent_txs) > self._max_window:
            self._recent_txs = self._recent_txs[-self._max_window :]

        vulnerability = _assess_vulnerability(tx)
        if vulnerability < 0.3:
            return None

        # Estimate theoretical sandwich profit (for monitoring)
        theoretical_profit = _estimate_sandwich_profit(tx)
        gas_cost = _estimate_sandwich_gas(tx)

        return MEVOpportunity(
            mev_type=MEVType.SANDWICH_DETECT,  # NEVER BACKRUN or ARBITRAGE
            chain=tx.chain,
            trigger_tx=tx,
            estimated_profit_usd=theoretical_profit,
            gas_cost_usd=gas_cost,
            confidence=vulnerability,
            expires_at_block=(tx.block_number + 1) if tx.block_number else None,
            detected_at=datetime.now(UTC),
            metadata={
                "vulnerability_score": vulnerability,
                "monitoring_only": True,
                "reason": "sandwich_detection_monitor",
                "swap_amount_usd": tx.swap_amount_usd,
            },
        )


def _assess_vulnerability(tx: MempoolTx) -> float:
    """Score how vulnerable a transaction is to sandwich attacks.

    Factors:
    - Large swap size (more price impact to exploit)
    - Low gas price (easier to frontrun)
    - Known DEX router (predictable execution)
    """
    score = 0.0

    # Swap size vulnerability
    if tx.swap_amount_usd >= 100_000:
        score += 0.4
    elif tx.swap_amount_usd >= 10_000:
        score += 0.3
    elif tx.swap_amount_usd >= 1_000:
        score += 0.15

    # Known swap function (predictable)
    if tx.swap_selector:
        score += 0.2

    # DEX detected
    if tx.dex_detected:
        score += 0.1

    # Low gas = easier to frontrun
    gas_gwei = tx.gas_price / 1e9
    if gas_gwei < 5:
        score += 0.15
    elif gas_gwei < 20:
        score += 0.05

    return min(score, 0.95)


def _estimate_sandwich_profit(tx: MempoolTx) -> float:
    """Estimate theoretical sandwich profit (for monitoring only)."""
    # Sandwich typically captures 0.5-2% of swap value
    return tx.swap_amount_usd * 0.01


def _estimate_sandwich_gas(tx: MempoolTx) -> float:
    """Estimate gas cost for a sandwich (2 txs: frontrun + backrun)."""
    from animus_mev.models import Chain

    # Two transactions needed
    gas_units = 300_000  # 150k per tx

    defaults = {
        Chain.BASE: 0.01,
        Chain.ARB: 0.1,
        Chain.OP: 0.01,
        Chain.ETH: 30.0,
    }

    if tx.gas_price > 0:
        gas_eth = (tx.gas_price * gas_units) / 1e18
    else:
        gas_gwei = defaults.get(tx.chain, 1.0)
        gas_eth = (gas_gwei * 1e9 * gas_units) / 1e18

    return gas_eth * 3000.0
