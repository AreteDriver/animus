"""MEV P&L tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from animus_mev.models import Chain, ExtractionResult, MEVType

logger = logging.getLogger("animus_mev.tracker")


@dataclass
class PnLSummary:
    """Profit and loss summary."""

    total_opportunities: int = 0
    total_executed: int = 0
    total_dry_runs: int = 0
    total_profit_usd: float = 0.0
    total_gas_usd: float = 0.0
    total_errors: int = 0
    by_chain: dict[str, float] = field(default_factory=dict)
    by_type: dict[str, float] = field(default_factory=dict)
    best_trade_usd: float = 0.0
    worst_trade_usd: float = 0.0
    win_rate: float = 0.0

    @property
    def net_profit_usd(self) -> float:
        return self.total_profit_usd - self.total_gas_usd

    def to_dict(self) -> dict:
        return {
            "total_opportunities": self.total_opportunities,
            "total_executed": self.total_executed,
            "total_dry_runs": self.total_dry_runs,
            "total_profit_usd": self.total_profit_usd,
            "total_gas_usd": self.total_gas_usd,
            "net_profit_usd": self.net_profit_usd,
            "total_errors": self.total_errors,
            "by_chain": self.by_chain,
            "by_type": self.by_type,
            "best_trade_usd": self.best_trade_usd,
            "worst_trade_usd": self.worst_trade_usd,
            "win_rate": self.win_rate,
        }


class PnLTracker:
    """Track MEV extraction performance."""

    def __init__(self):
        self._results: list[ExtractionResult] = []
        self._opportunities_seen: int = 0
        self._chain_profit: dict[str, float] = {}
        self._type_profit: dict[str, float] = {}

    @property
    def results(self) -> list[ExtractionResult]:
        return list(self._results)

    def record_opportunity_seen(self) -> None:
        """Record that an opportunity was detected."""
        self._opportunities_seen += 1

    def record_result(
        self,
        result: ExtractionResult,
        chain: Chain | None = None,
        mev_type: MEVType | None = None,
    ) -> None:
        """Record an extraction result."""
        self._results.append(result)

        if chain:
            chain_key = chain.value
            self._chain_profit[chain_key] = (
                self._chain_profit.get(chain_key, 0.0) + result.net_profit_usd
            )

        if mev_type:
            type_key = mev_type.value
            self._type_profit[type_key] = (
                self._type_profit.get(type_key, 0.0) + result.net_profit_usd
            )

    def get_summary(self) -> PnLSummary:
        """Get current P&L summary."""
        executed = [r for r in self._results if r.executed]
        dry_runs = [r for r in self._results if not r.executed and r.error == "dry_run"]
        errors = [r for r in self._results if r.error and r.error not in ("dry_run",)]

        profits = [r.net_profit_usd for r in self._results if r.executed]
        wins = [p for p in profits if p > 0]

        return PnLSummary(
            total_opportunities=self._opportunities_seen,
            total_executed=len(executed),
            total_dry_runs=len(dry_runs),
            total_profit_usd=sum(r.actual_profit_usd for r in self._results),
            total_gas_usd=sum(r.gas_used_usd for r in self._results),
            total_errors=len(errors),
            by_chain=dict(self._chain_profit),
            by_type=dict(self._type_profit),
            best_trade_usd=max(profits) if profits else 0.0,
            worst_trade_usd=min(profits) if profits else 0.0,
            win_rate=(len(wins) / len(profits)) if profits else 0.0,
        )

    def reset(self) -> None:
        """Reset all tracking data."""
        self._results.clear()
        self._opportunities_seen = 0
        self._chain_profit.clear()
        self._type_profit.clear()
