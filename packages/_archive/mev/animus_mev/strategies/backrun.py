"""Backrun strategy — execute after large DEX swaps to capture price impact."""

from __future__ import annotations

import logging

from animus_mev.models import Chain, MempoolTx, MEVOpportunity, MEVType
from animus_mev.strategies.base import BaseStrategy

logger = logging.getLogger("animus_mev.strategies.backrun")

# Typical price impact percentages by swap size
# Larger swaps create more price impact to capture
PRICE_IMPACT_ESTIMATE: dict[str, float] = {
    "small": 0.001,  # < $5k: 0.1% impact
    "medium": 0.003,  # $5k-$50k: 0.3% impact
    "large": 0.008,  # $50k-$500k: 0.8% impact
    "whale": 0.015,  # > $500k: 1.5% impact
}


class BackrunStrategy(BaseStrategy):
    """Detect backrun opportunities from large DEX swaps.

    When a large swap moves the price, we can place a trade right after
    to capture the price impact as it reverts. This is the most common
    and lowest-risk form of MEV on L2s.
    """

    @property
    def name(self) -> str:
        return "backrun"

    def is_applicable(self, tx: MempoolTx) -> bool:
        """Check if tx is a DEX swap above the minimum threshold."""
        if not tx.dex_detected:
            return False
        chain_config = self.config.get_chain(tx.chain)
        if not chain_config:
            return False
        min_swap = chain_config.min_swap_usd
        return tx.swap_amount_usd >= min_swap

    async def evaluate(self, tx: MempoolTx) -> MEVOpportunity | None:
        """Evaluate a DEX swap for backrun profit potential."""
        if not self.is_applicable(tx):
            return None

        swap_usd = tx.swap_amount_usd
        impact_rate = _estimate_price_impact(swap_usd)
        estimated_profit = swap_usd * impact_rate

        # Gas cost estimate (L2 gas is cheap)
        gas_cost = _estimate_gas_cost(tx.chain, tx.gas_price)

        net_profit = estimated_profit - gas_cost
        if net_profit <= 0:
            return None

        # Confidence based on swap size and whether we detected the selector
        confidence = _calculate_confidence(tx, swap_usd)

        if confidence < self.config.risk.min_confidence:
            return None

        return MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=tx.chain,
            trigger_tx=tx,
            estimated_profit_usd=estimated_profit,
            gas_cost_usd=gas_cost,
            confidence=confidence,
            expires_at_block=(tx.block_number + 1) if tx.block_number else None,
            metadata={
                "swap_amount_usd": swap_usd,
                "price_impact_rate": impact_rate,
                "swap_function": tx.swap_selector or "unknown",
            },
        )


def _estimate_price_impact(swap_usd: float) -> float:
    """Estimate the price impact percentage based on swap size."""
    if swap_usd >= 500_000:
        return PRICE_IMPACT_ESTIMATE["whale"]
    elif swap_usd >= 50_000:
        return PRICE_IMPACT_ESTIMATE["large"]
    elif swap_usd >= 5_000:
        return PRICE_IMPACT_ESTIMATE["medium"]
    return PRICE_IMPACT_ESTIMATE["small"]


def _estimate_gas_cost(chain: Chain, gas_price: int) -> float:
    """Estimate gas cost in USD for a backrun transaction."""

    # Typical gas for a swap: ~150k gas
    gas_units = 150_000

    if gas_price > 0:
        gas_eth = (gas_price * gas_units) / 1e18
    else:
        # Default gas estimates per chain (in gwei)
        defaults = {
            Chain.BASE: 0.01,  # Base is very cheap
            Chain.ARB: 0.1,
            Chain.OP: 0.01,
            Chain.ETH: 30.0,
        }
        gas_gwei = defaults.get(chain, 1.0)
        gas_eth = (gas_gwei * 1e9 * gas_units) / 1e18

    return gas_eth * 3000.0  # approximate ETH price


def _calculate_confidence(tx: MempoolTx, swap_usd: float) -> float:
    """Calculate confidence score for a backrun opportunity."""
    confidence = 0.5

    # Higher confidence if we identified the swap function
    if tx.swap_selector:
        confidence += 0.2

    # Higher confidence for larger swaps (more reliable price impact)
    if swap_usd >= 50_000:
        confidence += 0.2
    elif swap_usd >= 10_000:
        confidence += 0.1

    # Cap at 0.95 — never fully confident
    return min(confidence, 0.95)
