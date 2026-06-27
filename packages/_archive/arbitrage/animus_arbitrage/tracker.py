"""P&L tracking and reporting for arbitrage trades."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from animus_arbitrage.models import TradeResult
from animus_arbitrage.store import ArbitrageStore

logger = logging.getLogger("animus_arbitrage.tracker")


class PnLTracker:
    """Tracks profit and loss across all arbitrage trades.

    Provides summary statistics and reporting.
    """

    def __init__(self, store: ArbitrageStore):
        self.store = store

    def record_result(self, result: TradeResult) -> None:
        """Record a trade result."""
        self.store.record_result(result)

    def get_summary(self) -> dict[str, Any]:
        """Generate a P&L summary across all recorded trades."""
        results = self.store.load_results()
        opportunities = self.store.load_opportunities()

        if not results:
            return {
                "total_trades": 0,
                "executed_trades": 0,
                "dry_run_trades": 0,
                "total_profit_usd": 0.0,
                "total_opportunities": len(opportunities),
                "success_rate": 0.0,
                "avg_profit_usd": 0.0,
                "avg_slippage_pct": 0.0,
                "best_trade_usd": 0.0,
                "worst_trade_usd": 0.0,
            }

        executed = [r for r in results if r.executed]
        dry_runs = [r for r in results if not r.executed]
        successful = [r for r in executed if r.success]
        profits = [r.actual_profit_usd for r in results]

        total_profit = sum(profits)
        success_rate = len(successful) / len(executed) if executed else 0.0
        avg_profit = total_profit / len(results) if results else 0.0
        slippages = [r.slippage_pct for r in results if r.slippage_pct != 0.0]
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0.0

        return {
            "total_trades": len(results),
            "executed_trades": len(executed),
            "dry_run_trades": len(dry_runs),
            "successful_trades": len(successful),
            "total_profit_usd": round(total_profit, 4),
            "total_opportunities": len(opportunities),
            "success_rate": round(success_rate, 4),
            "avg_profit_usd": round(avg_profit, 4),
            "avg_slippage_pct": round(avg_slippage, 4),
            "best_trade_usd": round(max(profits), 4) if profits else 0.0,
            "worst_trade_usd": round(min(profits), 4) if profits else 0.0,
        }

    def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get the most recent trade results."""
        results = self.store.load_results()
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return [r.to_dict() for r in results[:limit]]

    def get_daily_summary(self) -> dict[str, dict[str, Any]]:
        """Get P&L broken down by day."""
        results = self.store.load_results()
        days: dict[str, list[TradeResult]] = {}

        for result in results:
            day_key = result.timestamp.strftime("%Y-%m-%d")
            if day_key not in days:
                days[day_key] = []
            days[day_key].append(result)

        summary: dict[str, dict[str, Any]] = {}
        for day, day_results in sorted(days.items()):
            profits = [r.actual_profit_usd for r in day_results]
            summary[day] = {
                "trades": len(day_results),
                "profit_usd": round(sum(profits), 4),
                "best_usd": round(max(profits), 4) if profits else 0.0,
                "worst_usd": round(min(profits), 4) if profits else 0.0,
            }

        return summary

    def total_profit(self) -> float:
        """Total cumulative profit in USD."""
        results = self.store.load_results()
        return sum(r.actual_profit_usd for r in results)

    def total_opportunities_detected(self) -> int:
        """Total number of opportunities ever detected."""
        return self.store.opportunity_count()

    def format_report(self) -> str:
        """Generate a human-readable P&L report."""
        s = self.get_summary()
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            f"=== Arbitrage P&L Report ({now}) ===",
            f"Total Opportunities: {s['total_opportunities']}",
            f"Total Trades: {s['total_trades']} (executed: {s['executed_trades']}, "
            f"dry-run: {s['dry_run_trades']})",
            f"Total Profit: ${s['total_profit_usd']:.4f}",
            f"Avg Profit/Trade: ${s['avg_profit_usd']:.4f}",
            f"Success Rate: {s['success_rate']:.1%}",
            f"Avg Slippage: {s['avg_slippage_pct']:.2f}%",
            f"Best Trade: ${s['best_trade_usd']:.4f}",
            f"Worst Trade: ${s['worst_trade_usd']:.4f}",
        ]
        return "\n".join(lines)
