"""Trade simulation to estimate profit before execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from animus_mev.config import MEVConfig
from animus_mev.models import MEVOpportunity, MEVType

logger = logging.getLogger("animus_mev.simulator")


@dataclass
class SimulationResult:
    """Result of a trade simulation."""

    opportunity_id: str
    profitable: bool
    estimated_profit_usd: float = 0.0
    gas_estimate_usd: float = 0.0
    net_profit_usd: float = 0.0
    revert_risk: float = 0.0  # 0-1
    simulated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "opportunity_id": self.opportunity_id,
            "profitable": self.profitable,
            "estimated_profit_usd": self.estimated_profit_usd,
            "gas_estimate_usd": self.gas_estimate_usd,
            "net_profit_usd": self.net_profit_usd,
            "revert_risk": self.revert_risk,
            "simulated_at": self.simulated_at.isoformat(),
            "error": self.error,
        }


class Simulator:
    """Simulate MEV trades before execution.

    Uses static analysis and heuristics to estimate profitability.
    In production, this would call eth_call or a simulation API
    (Tenderly, Flashbots simulate).
    """

    def __init__(self, config: MEVConfig):
        self.config = config
        self._simulation_count = 0

    @property
    def simulation_count(self) -> int:
        return self._simulation_count

    async def simulate(self, opportunity: MEVOpportunity) -> SimulationResult:
        """Simulate an MEV opportunity and estimate actual profit."""
        self._simulation_count += 1

        # Sandwich detection is never simulated for execution
        if opportunity.mev_type == MEVType.SANDWICH_DETECT:
            return SimulationResult(
                opportunity_id=opportunity.opportunity_id,
                profitable=False,
                estimated_profit_usd=opportunity.estimated_profit_usd,
                gas_estimate_usd=opportunity.gas_cost_usd,
                net_profit_usd=0.0,
                revert_risk=1.0,
                error="sandwich_detection_only",
            )

        # Estimate revert risk based on opportunity age and confidence
        revert_risk = _estimate_revert_risk(opportunity)

        # Adjust profit estimate with simulation discount
        # Real simulation would use eth_call — this is a conservative heuristic
        discount_factor = 1.0 - (revert_risk * 0.5)
        simulated_profit = opportunity.estimated_profit_usd * discount_factor

        # Re-estimate gas with more accurate model
        gas_estimate = _refine_gas_estimate(opportunity)

        net = simulated_profit - gas_estimate
        profitable = net > self.config.risk.min_profit_usd

        return SimulationResult(
            opportunity_id=opportunity.opportunity_id,
            profitable=profitable,
            estimated_profit_usd=simulated_profit,
            gas_estimate_usd=gas_estimate,
            net_profit_usd=net,
            revert_risk=revert_risk,
        )


def _estimate_revert_risk(opportunity: MEVOpportunity) -> float:
    """Estimate the risk that the transaction will revert."""
    risk = 0.1  # Base 10% revert risk

    # Lower confidence = higher revert risk
    risk += (1.0 - opportunity.confidence) * 0.3

    # Backruns are lower risk than arbitrage
    if opportunity.mev_type == MEVType.BACKRUN:
        risk -= 0.05
    elif opportunity.mev_type == MEVType.ARBITRAGE:
        risk += 0.1

    # ETH mainnet is more competitive = higher revert risk
    from animus_mev.models import Chain

    if opportunity.chain == Chain.ETH:
        risk += 0.2

    return max(0.0, min(risk, 0.95))


def _refine_gas_estimate(opportunity: MEVOpportunity) -> float:
    """Refine gas cost estimate with simulation-level precision."""
    base_gas = opportunity.gas_cost_usd

    # Add 20% buffer for gas price volatility
    return base_gas * 1.2
